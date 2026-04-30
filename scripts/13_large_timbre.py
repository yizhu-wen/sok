"""
TimbreWatermarking large-scale benchmark across all 7 datasets (embed on-the-fly).
Run: envs/timbre/bin/python scripts/13_large_timbre.py
Output: results/benchmark/{dataset_key}/timbre.json
"""
import os, sys, time

# large_scale_utils also sets these, but Timbre imports librosa before that module.
# librosa loads numba on import in this env, so the caps must be in place first.
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

# Must chdir before importing repo-relative modules; use resolve() for absolute paths
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "distortions"))
os.chdir(str(REPO_DIR))

import large_scale_utils as lu
from timbre_eval_utils import build_timbre_embed_decode, load_timbre_model, wait_for_gpu

ALGO         = "timbre"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DECODE_BATCH = 8    # conservative for the larger Timbre decoder; halved on OOM
_MIN_FREE_MB = 4000   # wait until this much GPU memory is free before loading / embedding
_POLL_S      = 60     # how often to re-check GPU memory (seconds)
_MAX_RETRY   = 10     # max retries on cuDNN error before giving up on a file
FORCE_CPU_DECODE = os.environ.get("SOK_FORCE_CPU_DECODE", "0") == "1"


def main():
    # Retry load_model until GPU has enough headroom (DNN may still be using 12+ GiB)
    for attempt in range(_MAX_RETRY):
        wait_for_gpu(DEVICE, _MIN_FREE_MB, _POLL_S, label="startup")
        try:
            encoder, decoder, wm_bits, model_sr = load_timbre_model(PROJECT_DIR, DEVICE)
            break
        except RuntimeError as e:
            if attempt < _MAX_RETRY - 1 and ("cuda" in str(e).lower() or "memory" in str(e).lower()):
                print(f"  [timbre] OOM during load_model (attempt {attempt+1}/{_MAX_RETRY}): {e}. "
                      f"Clearing cache, waiting {_POLL_S}s ...", flush=True)
                torch.cuda.empty_cache()
                time.sleep(_POLL_S)
            else:
                raise
    print(f"Timbre model_sr={model_sr}  device={DEVICE}", flush=True)
    embed, decode = build_timbre_embed_decode(
        encoder,
        decoder,
        wm_bits,
        model_sr,
        DEVICE,
        DECODE_BATCH,
        _MIN_FREE_MB,
        _POLL_S,
        _MAX_RETRY,
        force_cpu_decode=FORCE_CPU_DECODE,
    )

    for dataset_key, category, folder in lu.DATASETS:
        lu.run_dataset_benchmark(dataset_key, category, folder, ALGO, embed, decode)


if __name__ == "__main__":
    main()
