"""
Timing benchmark for TimbreWatermarking (embed + decode).
Run: envs/timbre/bin/python scripts/14_timing_timbre.py
Output: results/timing/timbre.json
"""
import os, sys, time, json

# Match the benchmark scripts: cap Numba/OpenMP before importing librosa.
os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import torch
import yaml
import librosa
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPO_DIR    = PROJECT_DIR / "repos" / "TimbreWatermarking" / "watermarking_model"

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "distortions"))
os.chdir(str(REPO_DIR))

import benchmark_utils as bu
import large_scale_utils as lu
from timbre_eval_utils import build_timbre_embed_decode, load_timbre_model

ALGO          = "timbre"
DECODE_BATCH  = 8
N_TIMING      = 10
N_DISTORTIONS = 116

out_dir = PROJECT_DIR / "results" / "timing"
out_dir.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    t0 = time.perf_counter()
    encoder, decoder, wm_bits, model_sr = load_timbre_model(PROJECT_DIR, DEVICE)
    model_load_s = time.perf_counter() - t0
    print(f"Model loaded in {model_load_s:.2f}s  device={DEVICE}  model_sr={model_sr}", flush=True)
    _embed, _decode_batch = build_timbre_embed_decode(
        encoder,
        decoder,
        wm_bits,
        model_sr,
        DEVICE,
        DECODE_BATCH,
        0,
        0,
        1,
    )

    files = lu.sample_files("speech", "daps", n=N_TIMING)

    # Warmup
    y0, sr0 = bu.load_audio_clipped(str(files[0]), stem=files[0].stem)
    _embed(y0, sr0)
    _decode_batch([y0], sr0)
    print("Warmup done.", flush=True)

    embed_times, decode1_times, decode116_times, file_names = [], [], [], []

    for path in files:
        stem = path.stem
        y, sr = bu.load_audio_clipped(str(path), stem=stem)

        t0 = time.perf_counter()
        wmed, _ = _embed(y, sr)
        embed_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _decode_batch([wmed], model_sr)
        decode1_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _decode_batch([wmed] * N_DISTORTIONS, model_sr)
        decode116_times.append(time.perf_counter() - t0)

        file_names.append(stem)
        print(f"  {stem}: embed={embed_times[-1]:.3f}s  "
              f"decode1={decode1_times[-1]:.3f}s  "
              f"decode116={decode116_times[-1]:.3f}s", flush=True)

    result = {
        "algo": ALGO,
        "device": str(DEVICE),
        "n_files": len(embed_times),
        "audio_duration_s": bu.MAX_AUDIO_DURATION,
        "n_distortions": N_DISTORTIONS,
        "model_load_s": round(model_load_s, 4),
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

    out_path = out_dir / f"{ALGO}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out_path}")
    print(f"  embed:      {result['mean_embed_s']:.3f} ± {result['std_embed_s']:.3f} s")
    print(f"  decode×1:   {result['mean_decode1_s']:.3f} ± {result['std_decode1_s']:.3f} s")
    print(f"  decode×116: {result['mean_decode116_s']:.3f} ± {result['std_decode116_s']:.3f} s")


if __name__ == "__main__":
    main()
