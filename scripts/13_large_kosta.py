"""
FSVC / Patchwork / NormSpace large-scale benchmark across all 7 datasets.
Run: envs/kosta/bin/python scripts/13_large_kosta.py
Output: results/benchmark/{dataset_key}/{fsvc,patchwork,normspace}.json
"""
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

PROJECT_DIR = Path(__file__).parent.parent
REPO_DIR = PROJECT_DIR / "repos" / "audio-watermarking"

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO_DIR))

import large_scale_utils as lu
from fsvc_watermarking import fsvc_watermark_embedding, fsvc_watermark_detection
from norm_space_watermarking import (
    norm_space_watermark_embedding,
    norm_space_watermark_detection,
)
from patchwork_multylayer_watermarking import (
    patchwork_multilayer_watermark_embedding,
    patchwork_multilayer_watermark_detection,
)

WATERMARK = np.array(
    [
        1, 0, 1, 0, 1, 1, 0, 0, 1, 0,
        1, 1, 0, 1, 0, 0, 1, 0, 1, 1,
        0, 0, 1, 1, 0, 1, 0, 1, 1, 0,
        0, 1, 0, 1, 0, 1, 1, 0, 0, 1,
    ],
    dtype=int,
)
WM_LENGTH = 40
SR_MODEL = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_algo(algo):
    def embed(y, sr):
        y16 = librosa.resample(y, orig_sr=sr, target_sr=SR_MODEL) if sr != SR_MODEL else y
        if algo == "fsvc":
            wmed = fsvc_watermark_embedding(y16, WATERMARK, SR_MODEL, device=DEVICE)
        elif algo == "patchwork":
            wmed = patchwork_multilayer_watermark_embedding(y16, WATERMARK, SR_MODEL, device=DEVICE)
        elif algo == "normspace":
            wmed = norm_space_watermark_embedding(y16, WATERMARK, device=DEVICE)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
        return np.array(wmed, dtype=np.float32), SR_MODEL

    def decode(ys, sr):
        """Sequential decode (frequency-domain algorithms; no native batch API)."""
        results = []
        for y in ys:
            y16 = librosa.resample(y, orig_sr=sr, target_sr=SR_MODEL) if sr != SR_MODEL else y
            if algo == "fsvc":
                bits = fsvc_watermark_detection(y16, WM_LENGTH, SR_MODEL, device=DEVICE)
            elif algo == "patchwork":
                bits = patchwork_multilayer_watermark_detection(y16, WM_LENGTH, SR_MODEL, device=DEVICE)
            elif algo == "normspace":
                bits = norm_space_watermark_detection(y16, WM_LENGTH, device=DEVICE)
            else:
                results.append(0.0)
                continue
            n = min(len(bits), WM_LENGTH)
            results.append(float(np.mean(bits[:n] == WATERMARK[:n])))
        return results

    for dataset_key, category, folder in lu.DATASETS:
        lu.run_dataset_benchmark(dataset_key, category, folder, algo, embed, decode)


def main():
    print(f"Kosta device: {DEVICE}", flush=True)
    for algo in ["fsvc", "patchwork", "normspace"]:
        print(f"\n{'=' * 60}", flush=True)
        print(f"=== {algo.upper()} ===", flush=True)
        print(f"{'=' * 60}", flush=True)
        run_algo(algo)


if __name__ == "__main__":
    main()
