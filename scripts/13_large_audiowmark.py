"""
audiowmark large-scale benchmark across all 7 datasets (embed on-the-fly via CLI).
Run: envs/viz/bin/python scripts/13_large_audiowmark.py
Output: results/benchmark/{dataset_key}/audiowmark.json
"""
import os, sys, subprocess, tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import large_scale_utils as lu

ALGO        = "audiowmark"
from config import AUDIOWMARK_BIN as AUDIOWMARK, AUDIOWMARK_LIB as _AW_LIB
LD_PATH = ":".join([str(_AW_LIB), os.environ.get("LD_LIBRARY_PATH", "")]).strip(":")
PAYLOAD_HEX = "0102030405060708090a0b0c0d0e0f10"
ENV         = {**os.environ, "LD_LIBRARY_PATH": LD_PATH}


def _payload_found(text):
    return PAYLOAD_HEX.lower() in text.lower()


def embed(y, sr):
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


def decode(ys, sr):
    """Sequential decode via CLI (audiowmark has no batch API)."""
    results = []
    for y in ys:
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "test.wav"
            sf.write(str(wav_path), y, sr)
            result = subprocess.run(
                [AUDIOWMARK, "get", str(wav_path)],
                capture_output=True, text=True, env=ENV
            )
        results.append(1.0 if _payload_found(result.stdout) else 0.0)
    return results


def main():
    print(f"audiowmark binary: {AUDIOWMARK}", flush=True)
    for dataset_key, category, folder in lu.DATASETS:
        lu.run_dataset_benchmark(dataset_key, category, folder, ALGO, embed, decode)


if __name__ == "__main__":
    main()
