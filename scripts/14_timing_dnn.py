"""
Timing benchmark for DNN watermark (embed + decode).
Run: envs/dnn_wm/bin/python scripts/14_timing_dnn.py
Output: results/timing/dnn_watermark.json

Uses the same re-exec pattern as 13_large_dnn.py to set LD_LIBRARY_PATH
before TensorFlow loads.
"""
import os, sys
from pathlib import Path

from config import NVIDIA_BASE as _CONFIG_NVIDIA_BASE, PTXAS_DIR as _PTXAS_DIR_CFG


def _candidate_nvidia_bases():
    import site as _site
    candidates = []
    env_base = Path(_site.getsitepackages()[0]) / "nvidia"
    candidates.append(env_base)
    if _CONFIG_NVIDIA_BASE is not None:
        candidates.append(Path(_CONFIG_NVIDIA_BASE))
    seen, ordered = set(), []
    for base in candidates:
        if base not in seen:
            seen.add(base); ordered.append(base)
    return ordered


def _pick_nvidia_base():
    for base in _candidate_nvidia_bases():
        if base.exists():
            return base
    return None


def _reexec_with_cuda_env():
    if "DNN_CUDA_LIBS_SET" in os.environ:
        return
    base = _pick_nvidia_base()
    if base is None:
        return
    lib_dirs = [str(p) for p in base.glob("*/lib") if p.is_dir()]
    if not lib_dirs:
        return
    ptxas_dir = str(_PTXAS_DIR_CFG) if _PTXAS_DIR_CFG is not None else ""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    if ptxas_dir:
        env["PATH"] = ptxas_dir + ":" + env.get("PATH", "")
    env["DNN_CUDA_LIBS_SET"] = "1"
    env["DNN_NVIDIA_BASE"]   = str(base)
    env["TF_XLA_FLAGS"]      = "--tf_xla_auto_jit=0"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


_reexec_with_cuda_env()

import time, json
import numpy as np

PROJECT_DIR = Path(__file__).parent.parent
REPO_DIR    = PROJECT_DIR / "repos" / "dnn-audio-watermarking"

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO_DIR))
import benchmark_utils as bu
import large_scale_utils as lu

ALGO          = "dnn_watermark"
SR            = 16000
STEP_SIZE     = 33216
HOP_LENGTH    = 511
WINDOW_LEN    = 1023
N_TIMING      = 10
N_DISTORTIONS = 116

out_dir = PROJECT_DIR / "results" / "timing"
out_dir.mkdir(parents=True, exist_ok=True)


def _require_gpu(tf):
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("DNN-WM requires GPU but TF found no GPU devices")
    print(f"DNN GPUs: {gpus}", flush=True)


def main():
    import os as _os
    _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import librosa
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    _require_gpu(tf)

    t0 = time.perf_counter()
    embedder = tf.keras.models.load_model(str(REPO_DIR / "embedder_model"))
    detector = tf.keras.models.load_model(str(REPO_DIR / "detector_model"))
    pool_path = REPO_DIR / "samples" / "message_pool.npy"
    if not pool_path.exists():
        pool_path = REPO_DIR / "dataset" / "message_pool.npy"
    message     = np.load(str(pool_path))[0].astype(np.float32)
    message_int = (message >= 0.5).astype(int)
    model_load_s = time.perf_counter() - t0
    print(f"Model loaded in {model_load_s:.2f}s  bits={len(message)}", flush=True)

    def _stft_input(chunk):
        chunk_tf = tf.constant(chunk.astype(np.float32))[tf.newaxis]
        stft = tf.transpose(tf.signal.stft(chunk_tf, WINDOW_LEN, HOP_LENGTH, WINDOW_LEN), perm=[0, 2, 1])
        return tf.stack([tf.math.real(stft), tf.math.imag(stft)], axis=-1)

    msg_tiled = np.tile(message.reshape(1, 1, 512), (16, 2, 1))
    msg_tf = tf.constant(msg_tiled, dtype=tf.float32)[tf.newaxis]

    def _embed(y, sr):
        y16 = librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
        n = len(y16)
        pad_len = STEP_SIZE - (n % STEP_SIZE) if n % STEP_SIZE != 0 else 0
        padded = np.pad(y16, (0, pad_len))
        out_chunks = []
        for start in range(0, len(padded), STEP_SIZE):
            chunk = padded[start:start + STEP_SIZE]
            stft_in = _stft_input(chunk)
            output = embedder([stft_in, msg_tf])
            stft_out = tf.complex(output[:, :, :, 0], output[:, :, :, 1])
            recovered = tf.signal.inverse_stft(
                tf.transpose(stft_out, perm=[0, 2, 1]),
                WINDOW_LEN, HOP_LENGTH, WINDOW_LEN,
                window_fn=tf.signal.inverse_stft_window_fn(HOP_LENGTH),
            )
            out_chunks.append(recovered.numpy()[0])
        return np.concatenate(out_chunks)[:n].astype(np.float32)

    def _decode_batch(ys, sr):
        all_chunks, chunk_counts = [], []
        for y in ys:
            y16 = librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
            pad_len = STEP_SIZE - (len(y16) % STEP_SIZE) if len(y16) % STEP_SIZE != 0 else 0
            padded = np.pad(y16, (0, pad_len))
            for start in range(0, len(padded), STEP_SIZE):
                all_chunks.append(_stft_input(padded[start:start + STEP_SIZE]))
            chunk_counts.append(len(padded) // STEP_SIZE)
        batch_in = tf.concat(all_chunks, axis=0)
        out_all  = detector(batch_in).numpy()
        results, offset = [], 0
        for count in chunk_counts:
            bits = (np.mean((out_all[offset:offset + count] >= 0.5).astype(int), axis=0) >= 0.5).astype(int)
            results.append(float(np.mean(bits == message_int[:len(bits)])))
            offset += count
        return results

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
        wmed = _embed(y, sr)
        embed_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _decode_batch([wmed], SR)
        decode1_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _decode_batch([wmed] * N_DISTORTIONS, SR)
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
