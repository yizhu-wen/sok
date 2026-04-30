import gc
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import yaml


def wait_for_gpu(device, min_free_mb, poll_s, label="", prefix="timbre"):
    if device.type != "cuda":
        return
    while True:
        torch.cuda.empty_cache()
        free_mb = torch.cuda.mem_get_info()[0] // (1024 * 1024)
        if free_mb >= min_free_mb:
            return
        print(
            f"  [{prefix}{' ' + label if label else ''}] GPU only {free_mb}MB free, "
            f"waiting {poll_s}s for {min_free_mb}MB ...",
            flush=True,
        )
        time.sleep(poll_s)


def load_timbre_model(project_dir, device):
    from model.conv2_mel_modules import Decoder, Encoder

    process_config = yaml.safe_load(open("config/process.yaml"))
    model_config = yaml.safe_load(open("config/model.yaml"))
    train_config = yaml.safe_load(open("config/train.yaml"))

    out_dir = project_dir / "outputs" / "timbre"
    for key in list(train_config["path"]):
        train_config["path"][key] = str(out_dir / "tmp")
    train_config["path"]["ckpt"] = "results/ckpt"
    train_config["path"]["log_path"] = str(out_dir / "log")

    msg_length = train_config["watermark"]["length"]
    win_dim = process_config["audio"]["win_len"]
    embedding_dim = model_config["dim"]["embedding"]
    nlayers_enc = model_config["layer"]["nlayers_encoder"]
    nlayers_dec = model_config["layer"]["nlayers_decoder"]
    attn_heads_enc = model_config["layer"]["attention_heads_encoder"]
    attn_heads_dec = model_config["layer"]["attention_heads_decoder"]

    encoder = Encoder(
        process_config,
        model_config,
        msg_length,
        win_dim,
        embedding_dim,
        nlayers_enc,
        attn_heads_enc,
    ).to(device)
    decoder = Decoder(
        process_config,
        model_config,
        msg_length,
        win_dim,
        embedding_dim,
        nlayers_dec,
        attn_heads_dec,
    ).to(device)

    ckpt_files = sorted(Path("results/ckpt/pth").glob("*.pth.tar"))
    assert ckpt_files, "No Timbre checkpoint found in results/ckpt/pth/"
    ckpt = torch.load(str(ckpt_files[0]), map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"], strict=False)
    encoder.eval()
    decoder.eval()
    decoder.robust = False

    with open("results/wmpool.txt") as f:
        wm_bits = eval(f.readlines()[0])

    model_sr = process_config["audio"]["sample_rate"]
    return encoder, decoder, wm_bits, model_sr


def build_timbre_embed_decode(
    encoder,
    decoder,
    wm_bits,
    model_sr,
    device,
    decode_batch,
    min_free_mb,
    poll_s,
    max_retry,
    prefix="timbre",
    force_cpu_decode=False,
):
    wm_np = np.array(wm_bits)
    msg = torch.from_numpy(np.array([[wm_bits]])).float() * 2 - 1
    msg = msg.to(device)
    decode_state = {
        "device": torch.device("cpu") if force_cpu_decode else device,
        "cpu_fallback_used": force_cpu_decode,
    }

    def _decoder_device():
        return next(decoder.parameters()).device

    def _move_decoder(target_device):
        if _decoder_device() == target_device:
            return
        decoder.to(target_device)
        decoder.eval()
        decoder.robust = False
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if decode_state["device"] != _decoder_device():
        _move_decoder(decode_state["device"])

    def _resample(y, sr):
        y_m = librosa.resample(y, orig_sr=sr, target_sr=model_sr) if sr != model_sr else y
        return np.asarray(y_m, dtype=np.float32)

    def _chunk_to_batch(chunk, sr):
        resampled = [_resample(y, sr) for y in chunk]
        max_len = max(len(y_m) for y_m in resampled)
        padded = []
        for y_m in resampled:
            if len(y_m) < max_len:
                y_m = np.pad(y_m, (0, max_len - len(y_m)))
            padded.append(y_m)
        return np.stack(padded)

    def _decode_chunk(chunk, sr, target_device):
        batch = _chunk_to_batch(chunk, sr)
        t = torch.from_numpy(batch).unsqueeze(1).to(target_device)
        with torch.inference_mode():
            decoded = decoder.test_forward(t)
        decoded_np = decoded.cpu().numpy()
        bits_batch = (decoded_np >= 0).astype(int).reshape(decoded_np.shape[0], -1)
        scores = []
        for bits in bits_batch:
            n = min(len(bits), len(wm_np))
            scores.append(float(np.mean(bits[:n] == wm_np[:n])))
        return scores

    def embed(y, sr):
        for attempt in range(max_retry):
            try:
                wait_for_gpu(device, min_free_mb, poll_s, prefix=prefix)
                y_m = _resample(y, sr)
                t = torch.from_numpy(y_m).float().unsqueeze(0).unsqueeze(0).to(device)
                with torch.inference_mode():
                    encoded, _ = encoder.test_forward(t, msg)
                return encoded.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32), model_sr
            except RuntimeError as e:
                if attempt < max_retry - 1 and ("cudnn" in str(e).lower() or "cuda" in str(e).lower()):
                    torch.cuda.empty_cache()
                    time.sleep(poll_s)
                else:
                    raise

    def decode(ys, sr):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        results = []
        bs = decode_batch
        i = 0
        while i < len(ys):
            chunk = ys[i : i + bs]
            attempt = 0
            while True:
                target_device = decode_state["device"]
                try:
                    if target_device.type == "cuda":
                        wait_for_gpu(target_device, min_free_mb, poll_s, label="decode", prefix=prefix)
                        torch.cuda.empty_cache()
                    _move_decoder(target_device)
                    results.extend(_decode_chunk(chunk, sr, target_device))
                    i += len(chunk)
                    if decode_state["device"].type == "cuda":
                        bs = decode_batch
                    else:
                        bs = 1
                    break
                except RuntimeError as e:
                    s = str(e).lower()
                    oom_like = "out of memory" in s or "cudnn" in s
                    if target_device.type == "cuda" and oom_like:
                        torch.cuda.empty_cache()
                        if bs > 1 and attempt < max_retry:
                            bs = max(1, bs // 2)
                            attempt += 1
                            print(f"    [{prefix}] OOM on decode, batch→{bs}", flush=True)
                            continue
                        if not decode_state["cpu_fallback_used"]:
                            decode_state["device"] = torch.device("cpu")
                            decode_state["cpu_fallback_used"] = True
                            bs = 1
                            print(
                                f"    [{prefix}] GPU decode still OOM at batch=1; "
                                f"switching decoder to CPU for remaining decode work",
                                flush=True,
                            )
                            continue
                    raise

        if device.type == "cuda":
            torch.cuda.empty_cache()
        return results

    return embed, decode
