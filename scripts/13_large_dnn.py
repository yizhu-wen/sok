"""
DNN watermark large-scale benchmark across all 7 datasets (embed on-the-fly).
Run: envs/dnn_wm/bin/python scripts/13_large_dnn.py
Output: results/benchmark/{dataset_key}/dnn_watermark.json
"""
import os
import sys
from pathlib import Path

import numpy as np

from config import NVIDIA_BASE as _CONFIG_NVIDIA_BASE, PTXAS_DIR as _PTXAS_DIR_CFG


def _candidate_nvidia_bases():
    import site as _site
    candidates = []
    # Use site.getsitepackages() — resolves to the venv's site-packages even when
    # the Python binary symlinks to a different conda env (py38).
    env_base = Path(_site.getsitepackages()[0]) / "nvidia"
    candidates.append(env_base)
    if _CONFIG_NVIDIA_BASE is not None:
        candidates.append(Path(_CONFIG_NVIDIA_BASE))

    seen = set()
    ordered = []
    for base in candidates:
        if base in seen:
            continue
        seen.add(base)
        ordered.append(base)
    return ordered


def _pick_nvidia_base():
    for base in _candidate_nvidia_bases():
        if base.exists():
            return base
    return None


def _reexec_with_cuda_env() -> None:
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
    env["DNN_NVIDIA_BASE"] = str(base)
    env["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


_reexec_with_cuda_env()

PROJECT_DIR = Path(__file__).parent.parent
REPO_DIR = PROJECT_DIR / "repos" / "dnn-audio-watermarking"

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO_DIR))
import large_scale_utils as lu

ALGO = "dnn_watermark"
SR = 16000
STEP_SIZE = 33216
HOP_LENGTH = 511
WINDOW_LEN = 1023
DECODE_BATCH = 32   # max STFT chunks per detector call — caps GPU memory


def _require_gpu(tf) -> None:
    build = tf.sysconfig.get_build_info()
    print(f"  TF {tf.__version__}  |  CUDA {build.get('cuda_version')}  "
          f"|  cuDNN {build.get('cudnn_version')}  "
          f"|  NVIDIA_BASE={os.environ.get('DNN_NVIDIA_BASE', 'N/A')}", flush=True)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"DNN GPUs: {gpus}", flush=True)
        return
    raise RuntimeError(
        "DNN-WM GPU requested, but TensorFlow did not register any GPU devices. "
        f"TF build expects CUDA {build.get('cuda_version')} / cuDNN {build.get('cudnn_version')}. "
        f"Launcher used NVIDIA_BASE={os.environ.get('DNN_NVIDIA_BASE', 'N/A')!r}. "
        "Check that nvidia-cu11 packages are installed in envs/dnn_wm and that "
        "the re-exec set LD_LIBRARY_PATH correctly."
    )


def main():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    import librosa
    import tensorflow as tf

    # Allow TF to grow GPU memory on demand instead of grabbing all 47 GiB at startup.
    # Without this TensorFlow claims the entire GPU, starving every other process.
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    _require_gpu(tf)

    embedder = tf.keras.models.load_model(str(REPO_DIR / "embedder_model"))
    detector = tf.keras.models.load_model(str(REPO_DIR / "detector_model"))

    pool_path = REPO_DIR / "samples" / "message_pool.npy"
    if not pool_path.exists():
        pool_path = REPO_DIR / "dataset" / "message_pool.npy"
    message = np.load(str(pool_path))[0].astype(np.float32)
    message_int = (message >= 0.5).astype(int)
    print(f"DNN model loaded on GPU, message bits: {len(message)}", flush=True)

    def _stft_input(chunk):
        chunk_tf = tf.constant(chunk.astype(np.float32))[tf.newaxis]
        stft = tf.transpose(tf.signal.stft(chunk_tf, WINDOW_LEN, HOP_LENGTH, WINDOW_LEN), perm=[0, 2, 1])
        return tf.stack([tf.math.real(stft), tf.math.imag(stft)], axis=-1)

    def embed(y, sr):
        y16 = librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
        n = len(y16)
        pad_len = STEP_SIZE - (n % STEP_SIZE) if n % STEP_SIZE != 0 else 0
        padded = np.pad(y16, (0, pad_len))
        out_chunks = []
        msg_tiled = np.tile(message.reshape(1, 1, 512), (16, 2, 1))
        msg_tf = tf.constant(msg_tiled, dtype=tf.float32)[tf.newaxis]
        for start in range(0, len(padded), STEP_SIZE):
            chunk = padded[start:start + STEP_SIZE]
            stft_in = _stft_input(chunk)
            output = embedder([stft_in, msg_tf])
            stft_out = tf.complex(output[:, :, :, 0], output[:, :, :, 1])
            recovered = tf.signal.inverse_stft(
                tf.transpose(stft_out, perm=[0, 2, 1]),
                WINDOW_LEN,
                HOP_LENGTH,
                WINDOW_LEN,
                window_fn=tf.signal.inverse_stft_window_fn(HOP_LENGTH),
            )
            out_chunks.append(recovered.numpy()[0])
        wmed = np.concatenate(out_chunks)[:n].astype(np.float32)
        return wmed, SR

    def decode(ys, sr):
        if not ys:
            return []
        # Collect chunks as numpy on CPU — no GPU tensors allocated yet
        all_np_chunks, chunk_counts = [], []
        for y in ys:
            y16 = librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
            pad_len = STEP_SIZE - (len(y16) % STEP_SIZE) if len(y16) % STEP_SIZE != 0 else 0
            padded  = np.pad(y16, (0, pad_len))
            for start in range(0, len(padded), STEP_SIZE):
                all_np_chunks.append(padded[start:start + STEP_SIZE].astype(np.float32))
            chunk_counts.append(len(padded) // STEP_SIZE)
        # Run detector in DECODE_BATCH-sized sub-batches to cap GPU memory
        out_parts = []
        for i in range(0, len(all_np_chunks), DECODE_BATCH):
            batch_stft = [_stft_input(c) for c in all_np_chunks[i:i + DECODE_BATCH]]
            batch_in   = tf.concat(batch_stft, axis=0)
            out_parts.append(detector(batch_in).numpy())
        out_all = np.concatenate(out_parts, axis=0)
        results, offset = [], 0
        for count in chunk_counts:
            bits_for_audio = (out_all[offset:offset + count] >= 0.5).astype(int)
            bits = (np.mean(bits_for_audio, axis=0) >= 0.5).astype(int)
            results.append(float(np.mean(bits == message_int[:len(bits)])))
            offset += count
        return results

    for dataset_key, category, folder in lu.DATASETS:
        lu.run_dataset_benchmark(dataset_key, category, folder, ALGO, embed, decode)


if __name__ == "__main__":
    main()
