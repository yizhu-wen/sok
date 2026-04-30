"""
Timing benchmark for FSVC / Patchwork / NormSpace (embed + decode).
Run: envs/kosta/bin/python scripts/14_timing_kosta.py
Output: results/timing/{fsvc,patchwork,normspace}.json
"""
import sys, time, json
import numpy as np
import librosa
import torch
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPO_DIR    = PROJECT_DIR / "repos" / "audio-watermarking"

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO_DIR))
import benchmark_utils as bu
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
    [1,0,1,0,1,1,0,0,1,0, 1,1,0,1,0,0,1,0,1,1, 0,0,1,1,0,1,0,1,1,0, 0,1,0,1,0,1,1,0,0,1],
    dtype=int,
)
WM_LENGTH     = 40
SR_MODEL      = 16000
N_TIMING      = 10
N_DISTORTIONS = 116
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

out_dir = PROJECT_DIR / "results" / "timing"
out_dir.mkdir(parents=True, exist_ok=True)


def time_algo(algo: str, files: list):
    print(f"\n{'='*50}\n  {algo.upper()}\n{'='*50}", flush=True)

    def _embed(y, sr):
        y16 = librosa.resample(y, orig_sr=sr, target_sr=SR_MODEL) if sr != SR_MODEL else y
        if algo == "fsvc":
            wmed = fsvc_watermark_embedding(y16, WATERMARK, SR_MODEL, device=DEVICE)
        elif algo == "patchwork":
            wmed = patchwork_multilayer_watermark_embedding(y16, WATERMARK, SR_MODEL, device=DEVICE)
        else:  # normspace
            wmed = norm_space_watermark_embedding(y16, WATERMARK, device=DEVICE)
        return np.array(wmed, dtype=np.float32)

    def _decode_one(y, sr):
        y16 = librosa.resample(y, orig_sr=sr, target_sr=SR_MODEL) if sr != SR_MODEL else y
        if algo == "fsvc":
            bits = fsvc_watermark_detection(y16, WM_LENGTH, SR_MODEL, device=DEVICE)
        elif algo == "patchwork":
            bits = patchwork_multilayer_watermark_detection(y16, WM_LENGTH, SR_MODEL, device=DEVICE)
        else:
            bits = norm_space_watermark_detection(y16, WM_LENGTH, device=DEVICE)
        n = min(len(bits), WM_LENGTH)
        return float(np.mean(bits[:n] == WATERMARK[:n]))

    # Warmup
    y0, sr0 = bu.load_audio_clipped(str(files[0]), stem=files[0].stem)
    _embed(y0, sr0)
    _decode_one(y0, sr0)
    print("  Warmup done.", flush=True)

    embed_times, decode1_times, decode116_times, file_names = [], [], [], []

    for path in files:
        stem = path.stem
        y, sr = bu.load_audio_clipped(str(path), stem=stem)

        t0 = time.perf_counter()
        wmed = _embed(y, sr)
        embed_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _decode_one(wmed, SR_MODEL)
        decode1_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        for _ in range(N_DISTORTIONS):
            _decode_one(wmed, SR_MODEL)
        decode116_times.append(time.perf_counter() - t0)

        file_names.append(stem)
        print(f"  {stem}: embed={embed_times[-1]:.3f}s  "
              f"decode1={decode1_times[-1]:.3f}s  "
              f"decode116={decode116_times[-1]:.3f}s", flush=True)

    result = {
        "algo": algo,
        "device": str(DEVICE),
        "n_files": len(embed_times),
        "audio_duration_s": bu.MAX_AUDIO_DURATION,
        "n_distortions": N_DISTORTIONS,
        "model_load_s": 0.0,  # no separate model load for these algorithms
        "embed_times_s":     [round(x, 4) for x in embed_times],
        "decode1_times_s":   [round(x, 4) for x in decode1_times],
        "decode116_times_s": [round(x, 4) for x in decode116_times],
        "mean_embed_s":      round(float(np.mean(embed_times)),    4),
        "std_embed_s":       round(float(np.std(embed_times)),     4),
        "mean_decode1_s":    round(float(np.mean(decode1_times)),  4),
        "std_decode1_s":     round(float(np.std(decode1_times)),   4),
        "mean_decode116_s":  round(float(np.mean(decode116_times)), 4),
        "std_decode116_s":   round(float(np.std(decode116_times)),  4),
        "files": file_names,
    }

    out_path = out_dir / f"{algo}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → {out_path}")
    print(f"  embed:      {result['mean_embed_s']:.3f} ± {result['std_embed_s']:.3f} s")
    print(f"  decode×1:   {result['mean_decode1_s']:.3f} ± {result['std_decode1_s']:.3f} s")
    print(f"  decode×116: {result['mean_decode116_s']:.3f} ± {result['std_decode116_s']:.3f} s")


def main():
    print(f"Kosta device: {DEVICE}", flush=True)
    files = lu.sample_files("speech", "daps", n=N_TIMING)

    for algo in ["fsvc", "patchwork", "normspace"]:
        time_algo(algo, files)


if __name__ == "__main__":
    main()
