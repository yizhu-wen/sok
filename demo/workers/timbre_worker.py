"""
Timbre watermarking worker (runs in the torch-2.7 venv, CPU).

Loads the vendored TimbreWatermarking model + checkpoint from
repos/TimbreWatermarking/watermarking_model and serves embed/decode over
the JSON-lines protocol.  The payload length is fixed by the trained
checkpoint (10 bits); the model operates at 22.05 kHz.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
REPO_DIR = PROJECT_DIR / "repos" / "TimbreWatermarking" / "watermarking_model"

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_DIR))
# distortions/mel_transform.py does a bare `from frequency import ...`,
# so the distortions dir itself must be importable too.
sys.path.insert(0, str(REPO_DIR / "distortions"))
# Repo code opens config/hifigan/checkpoint files with relative paths.
os.chdir(str(REPO_DIR))

import librosa  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import _protocol  # noqa: E402

_protocol.hijack_stdout()  # before any model code prints to stdout

DEVICE = torch.device("cpu")

# torch>=2.6 defaults torch.load(weights_only=True); the upstream Timbre and
# hifigan checkpoints predate that and were saved from GPU tensors. They are
# vendored in this repo (trusted); this worker always runs on CPU.
_orig_torch_load = torch.load


def _load_full(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    kwargs.setdefault("map_location", "cpu")
    return _orig_torch_load(*args, **kwargs)


torch.load = _load_full

from model.conv2_mel_modules import Decoder, Encoder  # noqa: E402


def _build_model():
    process_config = yaml.safe_load(open("config/process.yaml"))
    model_config = yaml.safe_load(open("config/model.yaml"))
    train_config = yaml.safe_load(open("config/train.yaml"))

    msg_length = train_config["watermark"]["length"]
    win_dim = process_config["audio"]["win_len"]
    embedding_dim = model_config["dim"]["embedding"]

    encoder = Encoder(process_config, model_config, msg_length, win_dim,
                      embedding_dim,
                      model_config["layer"]["nlayers_encoder"],
                      attention_heads=model_config["layer"]["attention_heads_encoder"],
                      ).to(DEVICE)
    decoder = Decoder(process_config, model_config, msg_length, win_dim,
                      embedding_dim,
                      model_config["layer"]["nlayers_decoder"],
                      attention_heads=model_config["layer"]["attention_heads_decoder"],
                      ).to(DEVICE)

    ckpt_files = sorted(Path("results/ckpt/pth").glob("*.pth.tar"))
    assert ckpt_files, "No Timbre checkpoint found in results/ckpt/pth/"
    ckpt = torch.load(str(ckpt_files[0]), map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"], strict=False)
    encoder.eval()
    decoder.eval()
    decoder.robust = False

    model_sr = process_config["audio"]["sample_rate"]
    return encoder, decoder, msg_length, model_sr


ENCODER, DECODER, N_BITS, MODEL_SR = _build_model()
print(f"[timbre_worker] model loaded (bits={N_BITS}, sr={MODEL_SR})",
      file=sys.stderr, flush=True)


def _read_resampled(path):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr != MODEL_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=MODEL_SR)
    return y


def handle_ping(req):
    return {"n_bits": N_BITS, "sr": MODEL_SR, "method": "timbre"}


def handle_embed(req):
    bits = np.asarray(req["bits"], dtype=np.int64)
    if len(bits) != N_BITS:
        raise ValueError(f"Timbre payload must be exactly {N_BITS} bits")
    y = _read_resampled(req["in_wav"])
    msg = torch.from_numpy(np.array([[bits]])).float() * 2 - 1
    t = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        encoded, _ = ENCODER.test_forward(t, msg.to(DEVICE))
    out = encoded.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    sf.write(req["out_wav"], out, MODEL_SR, subtype="FLOAT")
    return {"out_wav": req["out_wav"], "sr": MODEL_SR}


def handle_decode(req):
    y = _read_resampled(req["in_wav"])
    t = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        decoded = DECODER.test_forward(t)
    bits = (decoded.cpu().numpy() >= 0).astype(int).reshape(-1)[:N_BITS]
    return {"bits": [int(b) for b in bits]}


if __name__ == "__main__":
    _protocol.serve({"ping": handle_ping, "embed": handle_embed,
                     "decode": handle_decode})
