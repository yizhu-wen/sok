"""
RobustDNN (DNN-WM) watermarking worker (runs in the TensorFlow-2.12 venv, CPU).

Uses the vendored pretrained SavedModels from repos/dnn-audio-watermarking.
The message is a fixed-length 512-bit vector; audio is processed at 16 kHz in
33216-sample chunks (same constants as scripts/13_large_dnn.py).  Resampling
uses soxr to avoid pulling numba/librosa into the old-numpy TF environment.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import soxr  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
REPO_DIR = PROJECT_DIR / "repos" / "dnn-audio-watermarking"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _protocol  # noqa: E402

_protocol.hijack_stdout()  # before any model code prints to stdout

import tensorflow as tf  # noqa: E402

SR = 16000
STEP_SIZE = 33216
HOP_LENGTH = 511
WINDOW_LEN = 1023
N_BITS = 512

EMBEDDER = tf.keras.models.load_model(str(REPO_DIR / "embedder_model"))
DETECTOR = tf.keras.models.load_model(str(REPO_DIR / "detector_model"))
print(f"[dnn_worker] SavedModels loaded (bits={N_BITS}, sr={SR})",
      file=sys.stderr, flush=True)


def _read_resampled(path):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr != SR:
        y = soxr.resample(y, sr, SR).astype(np.float32)
    return y


def _stft_input(chunk):
    chunk_tf = tf.constant(chunk.astype(np.float32))[tf.newaxis]
    stft = tf.transpose(
        tf.signal.stft(chunk_tf, WINDOW_LEN, HOP_LENGTH, WINDOW_LEN),
        perm=[0, 2, 1])
    return tf.stack([tf.math.real(stft), tf.math.imag(stft)], axis=-1)


def _chunks(y):
    pad_len = STEP_SIZE - (len(y) % STEP_SIZE) if len(y) % STEP_SIZE else 0
    padded = np.pad(y, (0, pad_len))
    return [padded[s:s + STEP_SIZE] for s in range(0, len(padded), STEP_SIZE)]


def handle_ping(req):
    return {"n_bits": N_BITS, "sr": SR, "method": "dnn"}


def handle_embed(req):
    bits = np.asarray(req["bits"], dtype=np.float32)
    if len(bits) != N_BITS:
        raise ValueError(f"RobustDNN payload must be exactly {N_BITS} bits")
    y = _read_resampled(req["in_wav"])
    n = len(y)
    msg_tiled = np.tile(bits.reshape(1, 1, N_BITS), (16, 2, 1))
    msg_tf = tf.constant(msg_tiled, dtype=tf.float32)[tf.newaxis]
    out_chunks = []
    for chunk in _chunks(y):
        output = EMBEDDER([_stft_input(chunk), msg_tf])
        stft_out = tf.complex(output[:, :, :, 0], output[:, :, :, 1])
        recovered = tf.signal.inverse_stft(
            tf.transpose(stft_out, perm=[0, 2, 1]),
            WINDOW_LEN, HOP_LENGTH, WINDOW_LEN,
            window_fn=tf.signal.inverse_stft_window_fn(HOP_LENGTH))
        out_chunks.append(recovered.numpy()[0])
    wmed = np.concatenate(out_chunks)[:n].astype(np.float32)
    sf.write(req["out_wav"], wmed, SR, subtype="FLOAT")
    return {"out_wav": req["out_wav"], "sr": SR}


def handle_decode(req):
    y = _read_resampled(req["in_wav"])
    outs = [DETECTOR(_stft_input(c)).numpy() for c in _chunks(y)]
    out_all = np.concatenate(outs, axis=0)
    bits_per_chunk = (out_all >= 0.5).astype(int)
    bits = (np.mean(bits_per_chunk, axis=0) >= 0.5).astype(int).reshape(-1)
    return {"bits": [int(b) for b in bits[:N_BITS]]}


if __name__ == "__main__":
    _protocol.serve({"ping": handle_ping, "embed": handle_embed,
                     "decode": handle_decode})
