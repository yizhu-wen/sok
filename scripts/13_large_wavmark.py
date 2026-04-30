"""
wavmark large-scale benchmark across all 7 datasets (embed on-the-fly).
Run: envs/wavmark/bin/python scripts/13_large_wavmark.py
Output: results/benchmark/{dataset_key}/wavmark.json
"""
import sys
import numpy as np
import librosa
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import large_scale_utils as lu

ALGO    = "wavmark"
PAYLOAD = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0], dtype=np.int32)
SR      = 16000


def main():
    import wavmark
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("WavMark GPU required but CUDA is unavailable in envs/wavmark")
    model = wavmark.load_model().to(torch.device("cuda"))
    model.eval()
    print("wavmark model loaded", flush=True)

    def embed(y, sr):
        y16 = librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
        wmed, _ = wavmark.encode_watermark(model, y16, PAYLOAD, show_progress=False)
        if wmed is None:
            raise RuntimeError("wavmark encode_watermark returned None (audio too short?)")
        return wmed.astype(np.float32), SR

    def decode(ys, sr):
        """Sequential decode (wavmark API is per-sample; no native batch support)."""
        results = []
        for y in ys:
            y16 = librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
            decoded, _ = wavmark.decode_watermark(model, y16, show_progress=False)
            results.append(float("nan") if decoded is None else float(np.mean(decoded == PAYLOAD)))
        return results

    for dataset_key, category, folder in lu.DATASETS:
        lu.run_dataset_benchmark(dataset_key, category, folder, ALGO, embed, decode)


if __name__ == "__main__":
    main()
