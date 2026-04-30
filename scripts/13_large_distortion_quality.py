"""
Compute audio quality metrics (SI-SNR, PESQ, ESTOI) for each distortion setting
applied to the original (un-watermarked) audio, for each large-scale dataset.

Uses the same 10-file sample (seed=42) as the benchmark scripts.
ViSQOL is skipped (NaN) to keep the run fast; the Excel report shows "-" for it.

Run: envs/viz/bin/python scripts/13_large_distortion_quality.py
Output: results/benchmark/{dataset_key}/distortion_quality.json  (7 files)
"""
import json, sys
import numpy as np
import librosa
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import benchmark_utils as bu
import large_scale_utils as lu


def compute_metrics(ref: np.ndarray, deg: np.ndarray, sr: int) -> dict:
    n = min(len(ref), len(deg))
    ref, deg = ref[:n].astype(np.float32), deg[:n].astype(np.float32)
    return {
        "si_snr": bu.compute_si_snr(ref, deg),
        "pesq":   bu.compute_pesq(ref, deg, sr),
        "estoi":  bu.compute_estoi(ref, deg, sr),
        "visqol": float("nan"),   # skipped for speed; shows as "-" in Excel
    }


def avg_metrics(metric_list):
    keys = ["si_snr", "pesq", "estoi", "visqol"]
    return {k: float(np.nanmean([m[k] for m in metric_list])) for k in keys}


def run_dataset(dataset_key, category, folder):
    files = lu.sample_files(category, folder, n=lu.N_SAMPLES)
    print(f"\n  === {dataset_key}  ({len(files)} files) ===", flush=True)

    # Load all original files (clipped to 20 s)
    originals = []
    for path in files:
        stem = path.stem
        try:
            y, sr = bu.load_audio_clipped(str(path), stem=stem)
            originals.append((y, sr, stem))
        except Exception as e:
            print(f"    SKIP load {stem}: {e}", flush=True)

    if not originals:
        print(f"  ERROR: no files loaded for {dataset_key}", flush=True)
        return

    results = {}

    # ── No distortion ────────────────────────────────────────────────────────
    print("  [no_distortion]", flush=True)
    metrics = [compute_metrics(y, y, sr) for y, sr, _ in originals]
    results["no_distortion"] = avg_metrics(metrics)
    m = results["no_distortion"]
    print(f"    si_snr={m['si_snr']:7.2f}  pesq={m['pesq']:.3f}  "
          f"estoi={m['estoi']:.3f}  visqol={m['visqol']:.3f}", flush=True)

    # ── Distortions ──────────────────────────────────────────────────────────
    for dist_name, settings in bu.DISTORTIONS.items():
        print(f"  [{dist_name}]", flush=True)
        results[dist_name] = {}
        for setting in settings:
            label = bu.setting_label(setting)
            metrics = []
            for y, sr, stem in originals:
                try:
                    dist = bu.apply_distortion(y, sr, dist_name, setting)
                    metrics.append(compute_metrics(y, dist, sr))
                except Exception as e:
                    print(f"    SKIP {stem}: {e}", flush=True)
                    metrics.append({"si_snr": float("nan"), "pesq": float("nan"),
                                    "estoi": float("nan"), "visqol": float("nan")})
            results[dist_name][label] = avg_metrics(metrics)
            m = results[dist_name][label]
            print(f"    {label:>20}  si_snr={m['si_snr']:7.2f}  pesq={m['pesq']:.3f}  "
                  f"estoi={m['estoi']:.3f}", flush=True)

    out_path = lu.get_results_path(dataset_key, "distortion_quality")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {out_path}", flush=True)


def main():
    import sys
    # Optional: pass a single dataset_key as argument to process only that dataset
    # e.g.  envs/viz/bin/python scripts/13_large_distortion_quality.py speech_daps
    if len(sys.argv) > 1:
        key = sys.argv[1]
        match = [(k, c, f) for k, c, f in lu.DATASETS if k == key]
        if not match:
            print(f"Unknown dataset key: {key}. Valid keys: {[k for k,_,_ in lu.DATASETS]}")
            sys.exit(1)
        for k, c, f in match:
            run_dataset(k, c, f)
    else:
        for dataset_key, category, folder in lu.DATASETS:
            run_dataset(dataset_key, category, folder)


if __name__ == "__main__":
    main()
