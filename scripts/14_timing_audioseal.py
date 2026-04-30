"""
Timing benchmark for AudioSeal (embed + decode).
Run: envs/audioseal/bin/python scripts/14_timing_audioseal.py
Output: results/timing/audioseal.json
"""
import os, sys, time, json
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import numpy as np
import torch
import librosa
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
import benchmark_utils as bu
import large_scale_utils as lu

ALGO         = "audioseal"
MSG          = [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0]
SR           = 16000
N_TIMING     = 10
N_DISTORTIONS = 116
DECODE_BATCH = 32

out_dir = PROJECT_DIR / "results" / "timing"
out_dir.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("AudioSeal requires CUDA")
DEVICE = torch.device("cuda")


def main():
    from audioseal import AudioSeal

    t0 = time.perf_counter()
    generator = AudioSeal.load_generator("audioseal_wm_16bits").to(DEVICE)
    detector  = AudioSeal.load_detector("audioseal_detector_16bits").to(DEVICE)
    generator.eval(); detector.eval()
    model_load_s = time.perf_counter() - t0
    print(f"Model loaded in {model_load_s:.2f}s  device={DEVICE}", flush=True)

    msg_tensor = torch.tensor([MSG], dtype=torch.int32).to(DEVICE)
    msg_np     = np.array(MSG)

    def _embed(y, sr):
        y16 = librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
        t = torch.from_numpy(y16).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            wm  = generator.get_watermark(t, message=msg_tensor)
            out = (t + wm).squeeze(0).squeeze(0).cpu().numpy()
        return out.astype(np.float32)

    def _decode_batch(ys):
        bs = DECODE_BATCH
        i  = 0
        accs = []
        while i < len(ys):
            chunk   = ys[i : i + bs]
            max_len = max(len(y) for y in chunk)
            padded  = [np.pad(y, (0, max_len - len(y))) for y in chunk]
            t = torch.from_numpy(np.stack(padded)).unsqueeze(1).to(DEVICE)
            with torch.no_grad():
                _, msg_out = detector.detect_watermark(t)
            bits_batch = (msg_out.cpu().numpy() > 0.5).astype(int)
            accs.extend(float(np.mean(b == msg_np)) for b in bits_batch)
            i += bs
        return accs

    files = lu.sample_files("speech", "daps", n=N_TIMING)

    # Warmup (avoids CUDA JIT overhead on first timed call)
    y0, sr0 = bu.load_audio_clipped(str(files[0]), stem=files[0].stem)
    _embed(y0, sr0)
    _decode_batch([y0])
    print("Warmup done.", flush=True)

    embed_times, decode1_times, decode116_times, file_names = [], [], [], []

    for path in files:
        stem = path.stem
        y, sr = bu.load_audio_clipped(str(path), stem=stem)

        t0 = time.perf_counter()
        wmed = _embed(y, sr)
        embed_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _decode_batch([wmed])
        decode1_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _decode_batch([wmed] * N_DISTORTIONS)
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
        "embed_times_s":    [round(x, 4) for x in embed_times],
        "decode1_times_s":  [round(x, 4) for x in decode1_times],
        "decode116_times_s":[round(x, 4) for x in decode116_times],
        "mean_embed_s":       round(float(np.mean(embed_times)),   4),
        "std_embed_s":        round(float(np.std(embed_times)),    4),
        "mean_decode1_s":     round(float(np.mean(decode1_times)), 4),
        "std_decode1_s":      round(float(np.std(decode1_times)),  4),
        "mean_decode116_s":   round(float(np.mean(decode116_times)), 4),
        "std_decode116_s":    round(float(np.std(decode116_times)),  4),
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
