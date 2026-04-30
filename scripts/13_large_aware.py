"""
AWARE large-scale benchmark across all 7 datasets (embed on-the-fly).
Run: envs/aware/bin/python scripts/13_large_aware.py
Output: results/benchmark/{dataset_key}/aware.json
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import librosa
from pathlib import Path

# GPU via LD_LIBRARY_PATH from silent310 conda env (PyTorch lazy-loads CUDA)
from config import NVIDIA_BASE as _NVIDIA_BASE
if _NVIDIA_BASE is not None and _NVIDIA_BASE.exists():
    _nvidia_libs = ":".join(str(p) for p in _NVIDIA_BASE.glob("*/lib") if p.is_dir())
    os.environ["LD_LIBRARY_PATH"] = _nvidia_libs + ":" + os.environ.get("LD_LIBRARY_PATH", "")

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(PROJECT_DIR / "repos" / "aware" / "src"))

import large_scale_utils as lu

from aware.utils.models import load
from aware.service import embed_watermark, detect_watermark

ALGO      = "aware"
SR        = 16000
WATERMARK = np.array([1,0,1,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,0,0], dtype=np.int32)


def main():
    embedder, detector = load(name="AWARE")
    print("AWARE model loaded", flush=True)

    def _detect_bits(y16):
        detected = detect_watermark(y16, SR, detector)
        bits = detected[0] if isinstance(detected, tuple) else detected
        bits = np.array(bits)
        n = min(len(bits), len(WATERMARK))
        return bits[:n], n

    def embed(y, sr):
        y16 = librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
        wmed = embed_watermark(y16.astype(np.float32), SR, WATERMARK, embedder)
        wmed = np.array(wmed, dtype=np.float32)
        n = min(len(wmed), len(y16))
        return wmed[:n], SR

    def decode(ys, sr):
        """Sequential decode (AWARE API is per-sample)."""
        results = []
        for y in ys:
            y16 = librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
            bits, n = _detect_bits(y16)
            results.append(float("nan") if n == 0 else float(np.mean(bits == WATERMARK[:n])))
        return results

    for dataset_key, category, folder in lu.DATASETS:
        lu.run_dataset_benchmark(dataset_key, category, folder, ALGO, embed, decode)


if __name__ == "__main__":
    main()
