"""
Timing benchmark for AWARE (embed + decode).
Run: envs/aware/bin/python scripts/14_timing_aware.py
Output: results/timing/aware.json
"""
import os, sys, time, json
import numpy as np
import librosa
from pathlib import Path

from config import NVIDIA_BASE as _NVIDIA_BASE
if _NVIDIA_BASE is not None and _NVIDIA_BASE.exists():
    _nvidia_libs = ":".join(str(p) for p in _NVIDIA_BASE.glob("*/lib") if p.is_dir())
    os.environ["LD_LIBRARY_PATH"] = _nvidia_libs + ":" + os.environ.get("LD_LIBRARY_PATH", "")

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(PROJECT_DIR / "repos" / "aware" / "src"))
import benchmark_utils as bu
import large_scale_utils as lu

from aware.utils.models import load
from aware.service import embed_watermark, detect_watermark

ALGO          = "aware"
SR            = 16000
WATERMARK     = np.array([1,0,1,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,0,0], dtype=np.int32)
N_TIMING      = 10
N_DISTORTIONS = 116

out_dir = PROJECT_DIR / "results" / "timing"
out_dir.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.perf_counter()
    embedder, detector = load(name="AWARE")
    model_load_s = time.perf_counter() - t0
    print(f"Model loaded in {model_load_s:.2f}s", flush=True)

    def _embed(y, sr):
        y16 = librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
        wmed = embed_watermark(y16.astype(np.float32), SR, WATERMARK, embedder)
        wmed = np.array(wmed, dtype=np.float32)
        return wmed[:min(len(wmed), len(y16))]

    def _decode_one(y, sr):
        y16 = librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
        detected, _ = detect_watermark(y16, SR, detector)
        return float(np.mean(np.array(detected) == WATERMARK))

    files = lu.sample_files("speech", "daps", n=N_TIMING)

    # Warmup
    y0, sr0 = bu.load_audio_clipped(str(files[0]), stem=files[0].stem)
    wmed0 = _embed(y0, sr0)
    _decode_one(wmed0, SR)
    print("Warmup done.", flush=True)

    embed_times, decode1_times, decode116_times, file_names = [], [], [], []

    for path in files:
        stem = path.stem
        y, sr = bu.load_audio_clipped(str(path), stem=stem)

        t0 = time.perf_counter()
        wmed = _embed(y, sr)
        embed_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _decode_one(wmed, SR)
        decode1_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        for _ in range(N_DISTORTIONS):
            _decode_one(wmed, SR)
        decode116_times.append(time.perf_counter() - t0)

        file_names.append(stem)
        print(f"  {stem}: embed={embed_times[-1]:.3f}s  "
              f"decode1={decode1_times[-1]:.3f}s  "
              f"decode116={decode116_times[-1]:.3f}s", flush=True)

    result = {
        "algo": ALGO,
        "device": "cuda",
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
