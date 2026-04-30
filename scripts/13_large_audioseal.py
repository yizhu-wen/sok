"""
AudioSeal large-scale benchmark across all 7 datasets (embed on-the-fly).
Run: envs/audioseal/bin/python scripts/13_large_audioseal.py
Output: results/benchmark/{dataset_key}/audioseal.json
"""
import os, sys
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import numpy as np
import torch
import librosa
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import large_scale_utils as lu

ALGO         = "audioseal"
MSG          = [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0]
SR           = 16000
DECODE_BATCH = 32   # pad+stack tensors; halved on OOM

if not torch.cuda.is_available():
    raise RuntimeError("AudioSeal GPU required but CUDA is unavailable in envs/audioseal")
DEVICE = torch.device("cuda")


def main():
    from audioseal import AudioSeal
    generator = AudioSeal.load_generator("audioseal_wm_16bits").to(DEVICE)
    detector  = AudioSeal.load_detector("audioseal_detector_16bits").to(DEVICE)
    generator.eval(); detector.eval()
    msg_tensor = torch.tensor([MSG], dtype=torch.int32).to(DEVICE)
    msg_np = np.array(MSG)
    print(f"Device: {DEVICE}", flush=True)

    def embed(y, sr):
        y16 = librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
        t = torch.from_numpy(y16).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            wm = generator.get_watermark(t, message=msg_tensor)
            out = (t + wm).squeeze(0).squeeze(0).cpu().numpy()
        return out.astype(np.float32), SR

    def decode(ys, sr):
        """Batch decode: pad to max len, one GPU forward pass per DECODE_BATCH samples."""
        results = []
        bs = DECODE_BATCH
        i  = 0
        while i < len(ys):
            chunk = ys[i : i + bs]
            try:
                max_len = max(len(y) for y in chunk)
                padded = []
                for y in chunk:
                    y16 = librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
                    if len(y16) < max_len:
                        y16 = np.pad(y16, (0, max_len - len(y16)))
                    padded.append(y16.astype(np.float32))
                t = torch.from_numpy(np.stack(padded)).unsqueeze(1).to(DEVICE)  # (B,1,T)
                with torch.no_grad():
                    _, msg_out = detector.detect_watermark(t)
                bits_batch = (msg_out.cpu().numpy() > 0.5).astype(int)  # (B, 16)
                for b in bits_batch:
                    results.append(float(np.mean(b == msg_np)))
                i += len(chunk)
                bs = DECODE_BATCH   # reset after success
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and bs > 1:
                    torch.cuda.empty_cache()
                    bs = max(1, bs // 2)
                    print(f"    [audioseal] OOM on decode, batch→{bs}", flush=True)
                else:
                    raise
        return results

    for dataset_key, category, folder in lu.DATASETS:
        lu.run_dataset_benchmark(dataset_key, category, folder, ALGO, embed, decode)


if __name__ == "__main__":
    main()
