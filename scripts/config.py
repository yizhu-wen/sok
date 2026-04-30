"""
Machine-local configuration for the open-science release.

Prefer setting paths with environment variables. If you prefer, edit this file
once on your machine and keep the rest of the pipeline unchanged.

Relevant environment variables:
  SOK_STORAGE_DIR
  SOK_DATASET_DIR
  SOK_NOISE_DIR
  SOK_RIR_DIR
  SOK_AUDIOWMARK_BIN
  SOK_AUDIOWMARK_LIB
  SOK_NVIDIA_BASE
  SOK_PTXAS_DIR

After editing, verify paths exist:
    python scripts/config.py
"""
from pathlib import Path
import os
import sys


def _path_or_none(env_name, default=None):
    raw = os.environ.get(env_name, "").strip()
    if raw:
        return Path(os.path.expanduser(raw))
    if default is None:
        return None
    return Path(os.path.expanduser(default))


STORAGE_DIR = _path_or_none("SOK_STORAGE_DIR")

DATASET_DIR = _path_or_none(
    "SOK_DATASET_DIR",
    str(STORAGE_DIR / "dataset") if STORAGE_DIR is not None else "/path/to/dataset_root",
)
NOISE_DIR = _path_or_none(
    "SOK_NOISE_DIR",
    str(STORAGE_DIR / "background_noise") if STORAGE_DIR is not None else "/path/to/background_noise",
)
RIR_DIR = _path_or_none(
    "SOK_RIR_DIR",
    str(STORAGE_DIR / "reverberation") if STORAGE_DIR is not None else "/path/to/reverberation",
)

AUDIOWMARK_BIN = _path_or_none("SOK_AUDIOWMARK_BIN", "/usr/local/bin/audiowmark")
AUDIOWMARK_LIB = _path_or_none("SOK_AUDIOWMARK_LIB", "/usr/local/lib")

# Optional. Set these only when the CUDA libraries or ptxas binary are not
# discoverable through your environment manager or system PATH.
NVIDIA_BASE = _path_or_none("SOK_NVIDIA_BASE")
PTXAS_DIR = _path_or_none("SOK_PTXAS_DIR")


if __name__ == "__main__":
    checks = {
        "STORAGE_DIR": STORAGE_DIR,
        "DATASET_DIR": DATASET_DIR,
        "NOISE_DIR": NOISE_DIR,
        "RIR_DIR": RIR_DIR,
        "AUDIOWMARK_BIN": AUDIOWMARK_BIN,
        "AUDIOWMARK_LIB": AUDIOWMARK_LIB,
        "NVIDIA_BASE": NVIDIA_BASE,
        "PTXAS_DIR": PTXAS_DIR,
    }
    all_ok = True
    for name, path in checks.items():
        if path is None:
            print(f"  [OPTIONAL] {name}: None")
            continue
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"  [{status}]  {name}: {path}")
        if not exists and name in {"DATASET_DIR", "NOISE_DIR", "RIR_DIR", "AUDIOWMARK_BIN"}:
            all_ok = False
    sys.exit(0 if all_ok else 1)
