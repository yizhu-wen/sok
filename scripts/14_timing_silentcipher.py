"""
Timing benchmark for SilentCipher (embed + decode).
Run: envs/silentcipher/bin/python scripts/14_timing_silentcipher.py
Output: results/timing/silentcipher.json
"""
import ctypes, sys, time, json
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from config import NVIDIA_BASE as _NVIDIA_BASE
import benchmark_utils as bu
import large_scale_utils as lu

ALGO          = "silentcipher"
MESSAGE       = [123, 45, 67, 89, 12]
N_TIMING      = 10
N_DISTORTIONS = 116

out_dir = PROJECT_DIR / "results" / "timing"
out_dir.mkdir(parents=True, exist_ok=True)


def _ensure_symlink(target, link_path):
    if link_path.is_symlink() or link_path.exists():
        if link_path.resolve() == target.resolve():
            return
        link_path.unlink()
    link_path.symlink_to(target)


def _configure_cuda():
    if _NVIDIA_BASE is None or not _NVIDIA_BASE.exists():
        return
    import os
    lib_dirs = [str(p) for p in _NVIDIA_BASE.glob("*/lib") if p.is_dir()]
    nvrtc_lib = _NVIDIA_BASE / "cuda_nvrtc" / "lib" / "libnvrtc.so.12"
    compat_dir = Path("/tmp/silentcipher_cuda_compat")
    compat_dir.mkdir(parents=True, exist_ok=True)
    if nvrtc_lib.exists():
        _ensure_symlink(nvrtc_lib, compat_dir / "libnvrtc.so")
        ctypes.CDLL(str(compat_dir / "libnvrtc.so"), mode=ctypes.RTLD_GLOBAL)
        lib_dirs.insert(0, str(compat_dir))
    current = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([current] if current else []))


def main():
    _configure_cuda()
    import torch
    import silentcipher

    if not torch.cuda.is_available():
        raise RuntimeError("SilentCipher requires CUDA")

    t0 = time.perf_counter()
    model = silentcipher.get_model(model_type="44.1k", device="cuda")
    model_load_s = time.perf_counter() - t0
    print(f"Model loaded in {model_load_s:.2f}s  device=cuda", flush=True)

    msg_np = np.array(MESSAGE)

    def _embed(y, sr):
        encoded, _ = model.encode_wav(y, sr, MESSAGE, calc_sdr=False)
        encoded = np.array(encoded, dtype=np.float32)
        return encoded[0] if encoded.ndim > 1 else encoded

    def _decode_one(y, sr):
        result = model.decode_wav(y, sr, phase_shift_decoding=False)
        if result.get("status") and result["messages"]:
            return float(np.mean(np.array(result["messages"][0]) == msg_np))
        return 0.0

    files = lu.sample_files("speech", "daps", n=N_TIMING)

    # Warmup
    y0, sr0 = bu.load_audio_clipped(str(files[0]), stem=files[0].stem)
    _embed(y0, sr0)
    _decode_one(y0, sr0)
    print("Warmup done.", flush=True)

    embed_times, decode1_times, decode116_times, file_names = [], [], [], []

    for path in files:
        stem = path.stem
        y, sr = bu.load_audio_clipped(str(path), stem=stem)

        t0 = time.perf_counter()
        wmed = _embed(y, sr)
        embed_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _decode_one(wmed, sr)
        decode1_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        for _ in range(N_DISTORTIONS):
            _decode_one(wmed, sr)
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
