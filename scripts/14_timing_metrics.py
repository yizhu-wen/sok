"""
Measure per-call wall time for quality metrics: SI-SNR, PESQ, ESTOI, ViSQOL.
These are algorithm-independent — same cost for every watermarking method.

Run: envs/viz/bin/python scripts/14_timing_metrics.py
Output: results/timing/metrics.json

Model for how metrics are executed in the actual benchmark pipeline
(large_scale_utils.py):
  - SI-SNR:  synchronous in main thread, one call per distortion
  - PESQ, ESTOI, ViSQOL: submitted to ThreadPoolExecutor(N_DIST_WORKERS)
    → 3 tasks × 116 distortions = 348 tasks, 8 workers in parallel
  - Effective wall time ≈ 116 × (SI-SNR + max_task) / 1
                        + (PESQ+ESTOI+ViSQOL) total / N_DIST_WORKERS
    Simplified: 116 × (PESQ + ESTOI + ViSQOL) / N_DIST_WORKERS
"""
import sys, time, json
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
import benchmark_utils as bu
import large_scale_utils as lu

N_DISTORTIONS   = 116
N_DIST_WORKERS  = min(8, __import__("os").cpu_count() or 4)
N_AUDIO_SAMPLES = 5   # number of (ref, deg) audio pairs to average over

out_dir = PROJECT_DIR / "results" / "timing"
out_dir.mkdir(parents=True, exist_ok=True)


def make_audio_pairs(n):
    """Load n audio files; use the watermark-free file as both ref and deg
    (sufficient to benchmark metric runtime — signal shape matters, not value)."""
    files = lu.sample_files("speech", "daps", n=n)
    pairs = []
    for path in files:
        y, sr = bu.load_audio_clipped(str(path), stem=path.stem)
        # slight distortion for deg (avoid identical-signal edge cases in PESQ/ViSQOL)
        deg = (y + np.random.RandomState(0).normal(0, 0.001, len(y))).astype(np.float32)
        pairs.append((y, deg, sr))
    return pairs


def time_calls(fn, pairs, n_reps=3):
    """Return list of per-call wall times (seconds), averaged over n_reps per pair."""
    times = []
    for ref, deg, sr in pairs:
        for _ in range(n_reps):
            t0 = time.perf_counter()
            fn(ref, deg, sr)
            times.append(time.perf_counter() - t0)
    return times


def time_parallel_116(pairs, mode="speech"):
    """
    Simulate the actual benchmark: submit 116 × (PESQ+ESTOI+ViSQOL) tasks
    to ThreadPoolExecutor(N_DIST_WORKERS) and measure total wall time.
    Uses the first audio pair.
    """
    ref, deg, sr = pairs[0]

    def _run_all():
        futs = []
        with ThreadPoolExecutor(max_workers=N_DIST_WORKERS) as ex:
            for _ in range(N_DISTORTIONS):
                futs.append(ex.submit(bu.compute_pesq,   ref.copy(), deg.copy(), sr))
                futs.append(ex.submit(bu.compute_estoi,  ref.copy(), deg.copy(), sr))
                futs.append(ex.submit(bu.compute_visqol, ref.copy(), deg.copy(), sr, mode))
            # SI-SNR is synchronous — run it here too (measures real overlap cost)
            for _ in range(N_DISTORTIONS):
                bu.compute_si_snr(ref, deg)
        return [f.result() for f in futs]   # collect all

    # Two timed runs, take the second (first may have JIT / cold-cache overhead)
    _run_all()
    t0 = time.perf_counter()
    _run_all()
    return time.perf_counter() - t0


def main():
    print(f"N_DIST_WORKERS = {N_DIST_WORKERS}", flush=True)
    pairs = make_audio_pairs(N_AUDIO_SAMPLES)
    print(f"Loaded {len(pairs)} audio pairs ({bu.MAX_AUDIO_DURATION:.0f}s each)", flush=True)

    # ── Per-call timings ──────────────────────────────────────────────────────
    print("\nMeasuring SI-SNR ...", flush=True)
    sisnr_times = time_calls(lambda r, d, sr: bu.compute_si_snr(r, d), pairs)

    print("Measuring PESQ ...", flush=True)
    pesq_times  = time_calls(bu.compute_pesq, pairs)

    print("Measuring ESTOI ...", flush=True)
    estoi_times = time_calls(bu.compute_estoi, pairs)

    print("Measuring ViSQOL (speech mode) ...", flush=True)
    visqol_times = time_calls(
        lambda r, d, sr: bu.compute_visqol(r, d, sr, "speech"), pairs)

    # ── Simulated full-pipeline parallel run ──────────────────────────────────
    print(f"\nSimulating full parallel run ({N_DISTORTIONS} distortions, "
          f"{N_DIST_WORKERS} workers) ...", flush=True)
    wall_parallel = time_parallel_116(pairs, mode="speech")

    # Derived estimates
    mean_sisnr  = float(np.mean(sisnr_times))
    mean_pesq   = float(np.mean(pesq_times))
    mean_estoi  = float(np.mean(estoi_times))
    mean_visqol = float(np.mean(visqol_times))

    # Sequential total per file (all metrics for all 116 distortions)
    seq_total = N_DISTORTIONS * (mean_sisnr + mean_pesq + mean_estoi + mean_visqol)
    # Effective parallel time (per file, with N_DIST_WORKERS)
    # The measured wall_parallel is the ground truth for this
    par_total = wall_parallel

    print(f"\n{'='*60}")
    print(f"  SI-SNR (sync):    {mean_sisnr*1000:.2f} ms/call")
    print(f"  PESQ   (thread):  {mean_pesq:.3f} s/call")
    print(f"  ESTOI  (thread):  {mean_estoi:.3f} s/call")
    print(f"  ViSQOL (thread):  {mean_visqol:.3f} s/call")
    print(f"  ─────────────────────────────────────")
    print(f"  Sequential total per file  (116 dists): {seq_total:.1f} s")
    print(f"  Parallel wall time per file ({N_DIST_WORKERS} workers): {par_total:.1f} s")
    print(f"{'='*60}")

    result = {
        "n_distortions":    N_DISTORTIONS,
        "n_dist_workers":   N_DIST_WORKERS,
        "n_audio_samples":  len(pairs),
        "audio_duration_s": bu.MAX_AUDIO_DURATION,
        "sisnr_times_s":    [round(x, 6) for x in sisnr_times],
        "pesq_times_s":     [round(x, 4) for x in pesq_times],
        "estoi_times_s":    [round(x, 4) for x in estoi_times],
        "visqol_times_s":   [round(x, 4) for x in visqol_times],
        "mean_sisnr_s":     round(mean_sisnr,  6),
        "std_sisnr_s":      round(float(np.std(sisnr_times)), 6),
        "mean_pesq_s":      round(mean_pesq,   4),
        "std_pesq_s":       round(float(np.std(pesq_times)),  4),
        "mean_estoi_s":     round(mean_estoi,  4),
        "std_estoi_s":      round(float(np.std(estoi_times)), 4),
        "mean_visqol_s":    round(mean_visqol, 4),
        "std_visqol_s":     round(float(np.std(visqol_times)), 4),
        # per-file totals
        "metrics_sequential_per_file_s": round(seq_total, 2),
        "metrics_parallel_per_file_s":   round(par_total, 2),
    }

    out_path = out_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
