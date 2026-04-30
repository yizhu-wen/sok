"""
Aggregate timing results from results/timing/*.json and print a detailed report.
Run: envs/viz/bin/python scripts/14_timing_report.py
     (or any Python env with json/pathlib)

Output:
  - Console: per-algorithm timing table + per-dataset estimated total runtimes
  - results/timing/timing_report.json  (machine-readable summary)

Timing breakdown per file:
  embed        — watermark embedding (GPU forward pass)
  decode×116   — decode all 116 distorted copies (GPU/CLI)
  metrics×116  — SI-SNR + PESQ + ESTOI + ViSQOL for all 116 distortions
                  (parallel, N_DIST_WORKERS threads; Timbre/DNN skip ViSQOL)
  ─────────────────────────────────────────────────────
  TOTAL        — embed + decode×116 + metrics×116
"""
import json
from pathlib import Path
from typing import Optional

PROJECT_DIR = Path(__file__).resolve().parent.parent
TIMING_DIR  = PROJECT_DIR / "results" / "timing"

# Canonical algorithm order and display names
ALGO_ORDER = [
    ("audioseal",    "AudioSeal   (ICML'24)"),
    ("wavmark",      "WavMark     (ICLR'24)"),
    ("silentcipher", "SilentCipher(IS'24)  "),
    ("timbre",       "Timbre      (NDSS'24)"),
    ("dnn_watermark","DNN-WM     (DSP'22)  "),
    ("audiowmark",   "audiowmark  (CLI)    "),
    ("aware",        "AWARE       (arXiv'25)"),
    ("fsvc",         "FSVC        (TASLP'21)"),
    ("patchwork",    "Patchwork   (TASLP'17)"),
    ("normspace",    "NormSpace   (EURSP'19)"),
]

# All algorithms now run ViSQOL (NUMBA_THREADING_LAYER=omp fixed the Python 3.8 hang)
SKIP_VISQOL = set()

# Dataset file counts (from CLAUDE.md)
DATASETS = [
    ("speech_daps",       "Speech-DAPS",       200),
    ("speech_gigaspeech", "Speech-GigaSpeech", 6750),
    ("speech_ljspeech",   "Speech-LJSpeech",   13100),
    ("music_m4singer",    "Music-M4Singer",     20896),
    ("music_moisesdb",    "Music-MoisesDB",     240),
    ("event_clotho",      "Event-Clotho",       5925),
    ("event_esc50",       "Event-ESC50",        2000),
]


def fmt_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 0:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{int(s):02d}s"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h{int(m):02d}m"


def load_json(name: str) -> Optional[dict]:
    path = TIMING_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    # ── Load metric timing ───────────────────────────────────────────────────
    metrics = load_json("metrics")
    has_metrics = metrics is not None

    if has_metrics:
        n_workers   = metrics["n_dist_workers"]
        n_dist      = metrics["n_distortions"]
        mean_sisnr  = metrics["mean_sisnr_s"]
        mean_pesq   = metrics["mean_pesq_s"]
        mean_estoi  = metrics["mean_estoi_s"]
        mean_visqol = metrics["mean_visqol_s"]
        # Wall-time measured by the parallel-simulation run (with ViSQOL)
        metrics_par_full   = metrics["metrics_parallel_per_file_s"]
        # Without ViSQOL: scale the sequential total by same speedup factor
        seq_full    = metrics["metrics_sequential_per_file_s"]
        seq_no_vq   = n_dist * (mean_sisnr + mean_pesq + mean_estoi)
        speedup     = seq_full / metrics_par_full  # actual observed speedup
        metrics_par_novq = seq_no_vq / speedup     # apply same speedup ratio
    else:
        metrics_par_full = metrics_par_novq = None

    WIDTH = 100
    print("\n" + "=" * WIDTH)
    print("  AUDIO WATERMARKING — ALGORITHM TIMING REPORT")
    print("=" * WIDTH)
    print(f"  Source: {TIMING_DIR}")
    print(f"  Audio: 20 s clips, speech_daps dataset")
    print(f"  Decode×116 : decode all 116 distortion conditions per file")
    if has_metrics:
        print(f"  Metrics×116: SI-SNR(sync) + PESQ({mean_pesq:.1f}s) + "
              f"ESTOI({mean_estoi:.2f}s) + ViSQOL({mean_visqol:.1f}s) "
              f"— parallel, {n_workers} workers")
        print(f"  Measured parallel metric wall time: {metrics_par_full:.1f}s/file "
              f"(full) / {metrics_par_novq:.1f}s/file (no ViSQOL)")
    else:
        print(f"  [metrics.json not found — run 14_timing_metrics.py to include metric costs]")
    print("=" * WIDTH)

    # ── Per-algorithm timing table ──────────────────────────────────────────
    if has_metrics:
        header = (
            f"\n{'Algorithm':<28} {'Device':<10} {'Load':>6}  "
            f"{'Embed':>9}  {'Dec×1':>9}  {'Dec×116':>10}  "
            f"{'Metrics':>8}  {'Total/file':>10}"
        )
        sep = "-" * WIDTH
        print(header)
        print(sep)
    else:
        header = (
            f"\n{'Algorithm':<28} {'Device':<10} {'Load':>6}  "
            f"{'Embed':>9}  {'Dec×1':>9}  {'Dec×116':>10}  {'Per-file':>10}"
        )
        sep = "-" * 90
        print(header)
        print(sep)

    summary_rows = []

    for algo, label in ALGO_ORDER:
        d = load_json(algo)
        if d is None:
            print(f"  {label}   [NOT RUN — missing {algo}.json]")
            summary_rows.append({"algo": algo, "label": label.strip(), "available": False})
            continue

        embed_s  = d["mean_embed_s"]
        dec1_s   = d["mean_decode1_s"]
        dec116_s = d["mean_decode116_s"]
        load_s   = d.get("model_load_s", 0.0)
        device   = d.get("device", "?")[:9]
        extrap   = d.get("decode116_extrapolated", False)
        flag     = "*" if extrap else " "

        # Metric cost for this algo
        if has_metrics:
            skip_vq = algo in SKIP_VISQOL
            met_s   = metrics_par_novq if skip_vq else metrics_par_full
            total_s = embed_s + dec116_s + met_s
            vq_note = "†" if skip_vq else " "
        else:
            met_s   = None
            total_s = embed_s + dec116_s

        if has_metrics:
            print(
                f"  {label}  {device:<10} {load_s:>5.1f}  "
                f"{embed_s:>7.3f}±{d['std_embed_s']:.3f}  "
                f"{dec1_s:>7.3f}±{d['std_decode1_s']:.3f}  "
                f"{dec116_s:>8.2f}{flag}  "
                f"{met_s:>7.1f}{vq_note}  "
                f"{total_s:>10.1f}"
            )
        else:
            print(
                f"  {label}  {device:<10} {load_s:>5.1f}  "
                f"{embed_s:>7.3f}±{d['std_embed_s']:.3f}  "
                f"{dec1_s:>7.3f}±{d['std_decode1_s']:.3f}  "
                f"{dec116_s:>8.2f}{flag}  "
                f"{embed_s + dec116_s:>10.1f}"
            )

        summary_rows.append({
            "algo": algo, "label": label.strip(),
            "available": True,
            "device": d.get("device", "?"),
            "skip_visqol": algo in SKIP_VISQOL,
            "model_load_s":     load_s,
            "mean_embed_s":     embed_s,
            "std_embed_s":      d["std_embed_s"],
            "mean_decode1_s":   dec1_s,
            "std_decode1_s":    d["std_decode1_s"],
            "mean_decode116_s": dec116_s,
            "std_decode116_s":  d["std_decode116_s"],
            "decode116_extrapolated": extrap,
            "metrics_par_s":    met_s,
            "mean_total_per_file_s": total_s,
        })

    print(sep)
    if has_metrics:
        print(f"  All times in seconds.  Load = one-time startup cost.")
        print(f"  * Dec×116 extrapolated as {n_dist} × Dec×1 (sequential/CLI decode).")
        print(f"  † ViSQOL skipped (Python 3.8 env); metric cost uses PESQ+ESTOI only.")
    else:
        print(f"  Per-file = embed + decode×116 (quality metrics not included).")
        print(f"  * Dec×116 extrapolated as N_DISTORTIONS × Dec×1 (sequential/CLI decode).")

    # ── Per-dataset estimated total runtime ─────────────────────────────────
    print(f"\n\n{'='*WIDTH}")
    label_what = "embed + decode×116 + metrics×116" if has_metrics else "embed + decode×116 only"
    print(f"  ESTIMATED TOTAL RUNTIME PER DATASET  ({label_what})")
    print(f"  (mean_total_per_file × n_files; excludes model load, file I/O,")
    print(f"   distortion application time, and ViSQOL for † algos)")
    print(f"{'='*WIDTH}\n")

    avail = [r for r in summary_rows if r["available"]]

    ds_labels = [lbl for _, lbl, _ in DATASETS]
    col_w     = max(16, max(len(lbl) for lbl in ds_labels) + 2)
    algo_col  = 30
    print(" " * algo_col + "".join(f"{lbl:>{col_w}}" for lbl in ds_labels))
    print("-" * (algo_col + col_w * len(DATASETS)))

    for row in avail:
        per_file = row["mean_total_per_file_s"]
        line     = f"  {row['label']:<{algo_col - 2}}"
        for _, _, n_files in DATASETS:
            line += f"{fmt_time(per_file * n_files):>{col_w}}"
        print(line)

    print("-" * (algo_col + col_w * len(DATASETS)))

    if has_metrics:
        print(f"\n  Shared metric overhead per file (all algos): "
              f"{metrics_par_full:.0f}s (with ViSQOL) / {metrics_par_novq:.0f}s (no ViSQOL)")
        print(f"  Per-metric timings:  SI-SNR {mean_sisnr*1000:.1f}ms  "
              f"PESQ {mean_pesq:.2f}s  ESTOI {mean_estoi:.3f}s  ViSQOL {mean_visqol:.2f}s")
        print(f"  Parallelism: {n_workers} ThreadPoolExecutor workers per file.")

    # ── Save machine-readable report ────────────────────────────────────────
    report = {
        "metrics_timing": metrics,
        "algorithms": summary_rows,
        "datasets": [
            {
                "key": ds_key, "label": ds_label, "n_files": n_files,
                "estimated_total_s": {
                    row["algo"]: round(row["mean_total_per_file_s"] * n_files, 1)
                    for row in avail
                },
            }
            for ds_key, ds_label, n_files in DATASETS
        ],
    }
    out_path = TIMING_DIR / "timing_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Machine-readable report → {out_path}\n")


if __name__ == "__main__":
    main()
