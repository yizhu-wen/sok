"""
Timing benchmark for audiowmark CLI (embed + decode).
Run: envs/viz/bin/python scripts/14_timing_audiowmark.py
Output: results/timing/audiowmark.json
"""
import os, sys, subprocess, tempfile, time, json
import numpy as np
import soundfile as sf
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from config import AUDIOWMARK_BIN as AUDIOWMARK, AUDIOWMARK_LIB as _AW_LIB
import benchmark_utils as bu
import large_scale_utils as lu

ALGO          = "audiowmark"
PAYLOAD_HEX   = "0102030405060708090a0b0c0d0e0f10"
ENV           = {**os.environ, "LD_LIBRARY_PATH": str(_AW_LIB)}
N_TIMING      = 10
N_DISTORTIONS = 116

out_dir = PROJECT_DIR / "results" / "timing"
out_dir.mkdir(parents=True, exist_ok=True)


def _embed(y, sr):
    with tempfile.TemporaryDirectory() as tmp:
        in_wav  = Path(tmp) / "in.wav"
        out_wav = Path(tmp) / "out.wav"
        sf.write(str(in_wav), y, sr)
        result = subprocess.run(
            [AUDIOWMARK, "add", str(in_wav), str(out_wav), PAYLOAD_HEX],
            capture_output=True, env=ENV
        )
        if result.returncode != 0:
            raise RuntimeError(f"audiowmark add failed: {result.stderr.decode()[:300]}")
        wmed, wmed_sr = sf.read(str(out_wav))
    if wmed.ndim > 1:
        wmed = wmed.mean(axis=1)
    return wmed.astype(np.float32), wmed_sr


def _decode_one(y, sr):
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "test.wav"
        sf.write(str(wav_path), y, sr)
        result = subprocess.run(
            [AUDIOWMARK, "get", str(wav_path)],
            capture_output=True, text=True, env=ENV
        )
    return 1.0 if PAYLOAD_HEX in result.stdout else 0.0


def main():
    print(f"audiowmark binary: {AUDIOWMARK}", flush=True)

    files = lu.sample_files("speech", "daps", n=N_TIMING)

    # Warmup
    y0, sr0 = bu.load_audio_clipped(str(files[0]), stem=files[0].stem)
    wmed0, wmed_sr0 = _embed(y0, sr0)
    _decode_one(wmed0, wmed_sr0)
    print("Warmup done.", flush=True)

    # Model load time: N/A (CLI binary, no model to load)
    model_load_s = 0.0

    embed_times, decode1_times, decode116_times, file_names = [], [], [], []

    for path in files:
        stem = path.stem
        y, sr = bu.load_audio_clipped(str(path), stem=stem)

        t0 = time.perf_counter()
        wmed, wmed_sr = _embed(y, sr)
        embed_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _decode_one(wmed, wmed_sr)
        decode1_times.append(time.perf_counter() - t0)

        # Extrapolate decode×116 from decode×1 (CLI subprocess — linear scaling)
        decode116_times.append(decode1_times[-1] * N_DISTORTIONS)

        file_names.append(stem)
        print(f"  {stem}: embed={embed_times[-1]:.3f}s  "
              f"decode1={decode1_times[-1]:.3f}s  "
              f"decode116={decode116_times[-1]:.3f}s", flush=True)

    result = {
        "algo": ALGO,
        "device": "cpu (CLI)",
        "n_files": len(embed_times),
        "audio_duration_s": bu.MAX_AUDIO_DURATION,
        "n_distortions": N_DISTORTIONS,
        "decode116_extrapolated": True,  # CLI subprocess — linear scaling
        "model_load_s": model_load_s,
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
