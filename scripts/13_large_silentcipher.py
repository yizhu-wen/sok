"""
SilentCipher large-scale benchmark across all 7 datasets (embed on-the-fly).
Run: envs/silentcipher/bin/python scripts/13_large_silentcipher.py
Output: results/benchmark/{dataset_key}/silentcipher.json
"""
import ctypes
import os
import sys
from pathlib import Path

import numpy as np

from config import NVIDIA_BASE as _NVIDIA_BASE

sys.path.insert(0, str(Path(__file__).parent))
import large_scale_utils as lu

ALGO = "silentcipher"
MESSAGE = [123, 45, 67, 89, 12]


def _ensure_symlink(target: Path, link_path: Path) -> None:
    if link_path.is_symlink() or link_path.exists():
        if link_path.resolve() == target.resolve():
            return
        link_path.unlink()
    link_path.symlink_to(target)


def _configure_cuda_runtime() -> None:
    if _NVIDIA_BASE is None or not _NVIDIA_BASE.exists():
        return

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


def load_model():
    _configure_cuda_runtime()

    import torch
    import silentcipher

    if not torch.cuda.is_available():
        raise RuntimeError("SilentCipher GPU requested, but CUDA is unavailable in envs/silentcipher")

    print("SilentCipher device: cuda", flush=True)
    return silentcipher.get_model(model_type="44.1k", device="cuda")


def main():
    model = load_model()
    msg_np = np.array(MESSAGE)

    def embed(y, sr):
        encoded, _ = model.encode_wav(y, sr, MESSAGE, calc_sdr=False)
        encoded = np.array(encoded, dtype=np.float32)
        if encoded.ndim > 1:
            encoded = encoded[0]
        return encoded, sr

    def decode(ys, sr):
        """Sequential decode (SilentCipher API is per-sample)."""
        results = []
        for y in ys:
            try:
                result = model.decode_wav(y, sr, phase_shift_decoding=False)
                if result.get("status") and result["messages"]:
                    results.append(float(np.mean(np.array(result["messages"][0]) == msg_np)))
                else:
                    results.append(float("nan"))
            except Exception:
                results.append(float("nan"))
        return results

    for dataset_key, category, folder in lu.DATASETS:
        lu.run_dataset_benchmark(dataset_key, category, folder, ALGO, embed, decode)


if __name__ == "__main__":
    main()
