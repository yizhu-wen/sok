"""
AWARE watermarking worker (runs in the torch-2.7 venv, CPU).

AWARE has no pretrained embedder: it optimizes a small adversarial
perturbation against its detection network per clip (400 iterations by
default), so embedding is much slower than the other methods — several
minutes per clip on CPU.  SOK_AWARE_ITERS can lower the iteration count for
faster (weaker) embedding.  The payload is fixed at 20 bits; audio must be
16 kHz mono and contain speech-like content (AWARE refuses silence).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_DIR / "repos" / "aware" / "src"))

import librosa  # noqa: E402

import _protocol  # noqa: E402

_protocol.hijack_stdout()  # before any model code prints to stdout

from aware.service import detect_watermark, embed_watermark  # noqa: E402
from aware.utils.models import load  # noqa: E402

SR = 16000

_loaded = load(name="AWARE")
if _loaded is None:
    raise RuntimeError("aware.utils.models.load() returned None — see stderr log")
EMBEDDER, DETECTOR = _loaded
N_BITS = int(DETECTOR.detection_net.output_length)

_iters = os.environ.get("SOK_AWARE_ITERS", "").strip()
if _iters:
    EMBEDDER.num_iterations = int(_iters)
print(f"[aware_worker] loaded (bits={N_BITS}, iterations={EMBEDDER.num_iterations})",
      file=sys.stderr, flush=True)


def _read_resampled(path):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def handle_ping(req):
    return {"n_bits": N_BITS, "sr": SR, "method": "aware"}


def handle_embed(req):
    bits = np.asarray(req["bits"], dtype=np.int32)
    if len(bits) != N_BITS:
        raise ValueError(f"AWARE payload must be exactly {N_BITS} bits")
    y = _read_resampled(req["in_wav"])
    wmed = embed_watermark(y, SR, bits, EMBEDDER)
    wmed = np.asarray(wmed, dtype=np.float32).reshape(-1)
    n = min(len(wmed), len(y))
    sf.write(req["out_wav"], wmed[:n], SR, subtype="FLOAT")
    return {"out_wav": req["out_wav"], "sr": SR}


def handle_decode(req):
    y = _read_resampled(req["in_wav"])
    detected = detect_watermark(y, SR, DETECTOR)
    bits = detected[0] if isinstance(detected, tuple) else detected
    bits = np.asarray(bits).reshape(-1)
    if bits.size == 0:
        return {"bits": None}
    return {"bits": [int(b) for b in bits[:N_BITS]]}


if __name__ == "__main__":
    _protocol.serve({"ping": handle_ping, "embed": handle_embed,
                     "decode": handle_decode})
