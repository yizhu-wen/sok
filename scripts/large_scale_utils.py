"""
Shared utilities for large-scale dataset benchmarking (scripts/13_large_*.py).

For each dataset, embeds the watermark into audio files then evaluates robustness
by applying every distortion setting to the watermarked audio and measuring:
  - bit_accuracy  : fraction of watermark bits correctly recovered
  - si_snr        : Scale-Invariant SNR vs clean original
  - pesq          : PESQ wideband score vs clean original
  - estoi         : ESTOI score vs clean original
  - visqol        : ViSQOL MOS-LQO vs clean original (~4 s/call)
  - secs          : speaker encoder cosine similarity vs clean original

Streams one file at a time — O(1) audio memory regardless of dataset size.
Results saved to: results/benchmark/{dataset_key}/{algo}.json

Resume support
--------------
If a run is interrupted (power loss, crash), simply re-run the same script.
- Completed datasets are skipped automatically (final JSON already exists).
- Partially-completed datasets resume from the last checkpoint:
    results/benchmark/{dataset_key}/{algo}.checkpoint.json
  Checkpoints are written every _LOG_INTERVAL successfully embedded files.
  The checkpoint stores per-distortion sums+counts (compact, constant size)
  plus the set of fully-processed file stems; appending new values to the
  reconstructed accumulators gives the mathematically correct grand mean.

Parallel PESQ / ESTOI / ViSQOL
-------------------------------
PESQ (~3 s/call) and ViSQOL (~4 s/call) — and ESTOI (~0.3 s/call) — are all
submitted to a ThreadPoolExecutor, so they run concurrently while the next
distortion is being applied in the main thread.  SI-SNR stays synchronous
(negligible cost).  N_DIST_WORKERS controls concurrency; tune down if running
many algo scripts simultaneously.  decode_fn always runs in the main thread
(GPU-safe).
"""
import hashlib
import json, os, sys
import subprocess
import tempfile
# Both pystoi (ESTOI) and visqol-python (gammatone) use Numba JIT and are called
# concurrently from ThreadPoolExecutor workers.  "workqueue" crashes under concurrent
# access; "tbb" requires Intel TBB which is not installed in all venvs; "omp" (OpenMP
# via libgomp, always present with GCC on Linux) is thread-safe and works with both
# Numba 0.59 and 0.64.
os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")
# Prevent OMP thread explosion: each of the N_DIST_WORKERS ThreadPoolExecutor workers
# would otherwise spawn its own OMP pool, creating hundreds of competing threads.
# Setting OMP/NUMBA threads to 1 keeps each worker serial; parallelism comes from
# the worker pool itself.
os.environ.setdefault("OMP_NUM_THREADS",   "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import librosa
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
import benchmark_utils as bu

from config import DATASET_DIR
N_SAMPLES      = None   # None = use all files in the dataset; set to int for a fixed sample
_LOG_INTERVAL  = 100                            # print progress and checkpoint every N files
N_DIST_WORKERS = min(8, os.cpu_count() or 4)   # parallel PESQ/ESTOI/ViSQOL workers per file
_SKIP_VISQOL          = False  # set to True in caller scripts where visqol-python hangs (Python 3.8 envs)
_SAFE_RESULT_TIMEOUT  = 120    # seconds to wait per ViSQOL/PESQ/ESTOI future; increase for slow envs
_SECS_PYTHON          = os.environ.get(
    "SOK_SECS_PYTHON",
    str(PROJECT_DIR / "envs" / "timbre" / "bin" / "python"),
)
_SECS_SCRIPT          = Path(__file__).with_name("resemblyzer_secs_server.py")

# (dataset_key, category_folder, dataset_folder)
DATASETS = [
    ("speech_daps",       "speech", "daps"),
    ("speech_gigaspeech", "speech", "gigaspeech"),
    ("speech_ljspeech",   "speech", "LJSpeech-1.1"),
    ("music_m4singer",    "music",  "m4singer"),
    ("music_moisesdb",    "music",  "moisesdb"),
    ("event_clotho",      "event",  "Clotho"),
    ("event_esc50",       "event",  "ESC-50-master"),
]


def sample_files(category, folder, n=N_SAMPLES, seed=42):
    """Return WAV paths from the dataset. n=None returns all files; otherwise samples n."""
    dataset_path = DATASET_DIR / category / folder
    all_wavs = sorted([
        p for p in dataset_path.rglob("*.wav")
        if not p.name.startswith("._")
    ])
    if not all_wavs:
        raise FileNotFoundError(f"No WAV files found in {dataset_path}")
    if n is None or n >= len(all_wavs):
        return all_wavs
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(all_wavs), size=n, replace=False))
    return [all_wavs[i] for i in idx]


def get_results_path(dataset_key, algo):
    out_dir = PROJECT_DIR / "results" / "benchmark" / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{algo}.json"


def _get_ckpt_path(dataset_key, algo):
    return get_results_path(dataset_key, algo).with_suffix(".checkpoint.json")


def resample_if_needed(y, orig_sr, target_sr):
    if orig_sr == target_sr:
        return y
    return librosa.resample(y.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)


def _nanmean(lst):
    return float(np.nanmean(lst)) if lst else float("nan")


def _ncount(lst):
    return sum(1 for v in lst if v == v)  # non-NaN count


def _distortions_hash(distortions=None):
    """Compact fingerprint of the DISTORTIONS config for skip invalidation."""
    distortions = bu.DISTORTIONS if distortions is None else distortions
    h = hashlib.md5()
    for name, settings in distortions.items():
        h.update(f"{name}:{settings}".encode())
    return h.hexdigest()[:12]


# ── Checkpoint helpers ────────────────────────────────────────────────────────

_METRICS = ("bit", "si_snr", "pesq", "estoi", "visqol", "secs")


def _serialize_acc(acc, dist_keys):
    """Convert list-based acc → compact {n, s, nan_n} dict for checkpoint storage."""
    def _summarize(lst):
        s, n, nan_n = 0.0, 0, 0
        for v in lst:
            if v == v:  # not NaN
                s += v; n += 1
            else:
                nan_n += 1
        return {"s": s, "n": n, "nan_n": nan_n}

    def _entry(a):
        return {m: _summarize(a[m]) for m in _METRICS}

    data = {"no_distortion": _entry(acc["no_distortion"])}
    for key, dist_name, setting, label in dist_keys:
        data.setdefault(dist_name, {})[label] = _entry(acc[key])
    return data


def _deserialize_acc(serialized, dist_keys):
    """Reconstruct list-based acc from compact checkpoint data.

    Each metric list is rebuilt as [mean]*n_valid + [nan]*nan_n so that
    appending further values and calling _nanmean gives the correct grand mean.
    """
    def _reconstruct(d):
        result = {}
        for m in _METRICS:
            e = d.get(m, {})
            n_valid = e.get("n", 0)
            s       = e.get("s", 0.0)
            nan_n   = e.get("nan_n", 0)
            mean_v  = s / n_valid if n_valid > 0 else float("nan")
            result[m] = [mean_v] * n_valid + [float("nan")] * nan_n
        return result

    acc = {"no_distortion": _reconstruct(serialized.get("no_distortion", {}))}
    for key, dist_name, setting, label in dist_keys:
        row = serialized.get(dist_name, {}).get(label, {})
        acc[key] = _reconstruct(row)
    return acc


def _save_checkpoint(ckpt_path, n_embedded, files_done, acc, dist_keys):
    data = {
        "n_embedded": n_embedded,
        "files_done": sorted(files_done),
        "acc":        _serialize_acc(acc, dist_keys),
    }
    # Write atomically: temp file → rename
    tmp = ckpt_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.replace(ckpt_path)


def _load_checkpoint(ckpt_path, dist_keys):
    with open(ckpt_path) as f:
        data = json.load(f)
    acc = _deserialize_acc(data.get("acc", {}), dist_keys)
    return data["n_embedded"], set(data["files_done"]), acc


# ── Main benchmark loop ───────────────────────────────────────────────────────

class _SecsClient:
    """Persistent JSON-lines client for the Resemblyzer SECS worker."""

    def __init__(self):
        env = os.environ.copy()
        # Keep the SECS worker on CPU to avoid contending with watermark models.
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
        self._proc = subprocess.Popen(
            [_SECS_PYTHON, str(_SECS_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

    def close(self):
        if getattr(self, "_proc", None) is None:
            return
        if self._proc.poll() is None:
            try:
                self._request({"cmd": "shutdown"})
            except Exception:
                pass
            try:
                self._proc.terminate()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None

    def _request(self, payload):
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("SECS worker is not available")
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            err = ""
            if self._proc.stderr is not None:
                err = self._proc.stderr.read().strip()
            raise RuntimeError(f"SECS worker exited unexpectedly: {err}")
        reply = json.loads(line)
        if not reply.get("ok", False):
            raise RuntimeError(reply.get("error", "unknown SECS worker error"))
        return reply

    def compare_many(self, ref_wav, deg_items, sr):
        """Return cosine similarities for deg_items against ref_wav."""
        with tempfile.TemporaryDirectory(prefix="secs_") as tmp_dir:
            tmp = Path(tmp_dir)
            ref_path = tmp / "ref.wav"
            bu.sf.write(str(ref_path), np.asarray(ref_wav, dtype=np.float32), sr, subtype="FLOAT")
            deg_paths = []
            for idx, wav in enumerate(deg_items):
                deg_path = tmp / f"deg_{idx:03d}.wav"
                bu.sf.write(str(deg_path), np.asarray(wav, dtype=np.float32), sr, subtype="FLOAT")
                deg_paths.append(str(deg_path))
            reply = self._request({
                "cmd": "compare_many",
                "ref_path": str(ref_path),
                "deg_paths": deg_paths,
            })
        return [
            float(score) if score is not None else float("nan")
            for score in reply.get("scores", [])
        ]


def run_dataset_benchmark(dataset_key, category, folder, algo,
                          embed_fn, decode_fn, n_samples=N_SAMPLES,
                          distortions=None, compute_quality=True,
                          compute_secs=False):
    """
    For each file: embed watermark → apply every distortion to the watermarked audio
    → measure bit_accuracy + selected quality metrics vs clean original.

    Streams one file at a time — O(1) audio memory regardless of dataset size.
    Resumes automatically from checkpoint if a previous run was interrupted.

    embed_fn(y: np.ndarray, sr: int) -> (watermarked: np.ndarray, wmed_sr: int)
    decode_fn(ys: list[np.ndarray], sr: int) -> list[float]  (bit accuracy per audio)

    decode_fn receives ALL distorted arrays for one file in a single call so that
    GPU algorithms can batch them.  For non-GPU algorithms, a plain loop is fine.

    Saves to results/benchmark/{dataset_key}/{algo}.json
    """
    distortions = bu.DISTORTIONS if distortions is None else distortions

    out_path  = get_results_path(dataset_key, algo)
    ckpt_path = _get_ckpt_path(dataset_key, algo)

    # ── Skip if already complete ──────────────────────────────────────────────
    if out_path.exists():
        _skip_complete = True
        try:
            with open(out_path) as _f:
                _existing = json.load(_f)
            _saved_hash = _existing.get("_meta", {}).get("distortions_hash", "")
            _curr_hash  = _distortions_hash(distortions)
            if _saved_hash and _saved_hash != _curr_hash:
                print(f"\n  [WARN] {algo}  {dataset_key} — distortions config changed "
                      f"(saved={_saved_hash!r} current={_curr_hash!r}). "
                      f"Delete {out_path.name} to re-run.", flush=True)
            if compute_secs:
                _secs_done = _existing.get("no_distortion", {}).get("n_secs", 0)
                _n_done = _existing.get("_meta", {}).get("n_processed", 0)
                if not _secs_done or _secs_done < _n_done:
                    _skip_complete = False
                    print(f"\n  [RERUN] {algo}  {dataset_key} — existing result lacks SECS; recomputing.", flush=True)
        except Exception:
            pass
        if _skip_complete:
            print(f"\n  [SKIP] {algo}  {dataset_key} — already complete ({out_path.name})",
                  flush=True)
            return

    all_files   = sample_files(category, folder, n=n_samples)
    visqol_mode = "speech" if dataset_key.startswith("speech") else "audio"

    # Build flat distortion list once (needed by checkpoint helpers too)
    dist_keys = []
    for dist_name, settings in distortions.items():
        for setting in settings:
            label = bu.setting_label(setting)
            dist_keys.append(((dist_name, label), dist_name, setting, label))

    def _new_lists():
        return {m: [] for m in _METRICS}

    # ── Resume from checkpoint or start fresh ─────────────────────────────────
    if ckpt_path.exists():
        n_embedded, files_done, acc = _load_checkpoint(ckpt_path, dist_keys)
        files = [f for f in all_files if f.stem not in files_done]
        print(f"\n  === {algo}  {dataset_key}  "
              f"(resuming: {n_embedded} done, {len(files)} remaining) ===", flush=True)
    else:
        n_embedded = 0
        files_done = set()
        acc = {"no_distortion": _new_lists()}
        for key, *_ in dist_keys:
            acc[key] = _new_lists()
        files = all_files
        print(f"\n  === {algo}  {dataset_key}  ({len(files)} files) ===", flush=True)

    # ── Per-file processing ───────────────────────────────────────────────────
    # (dist_name, label) → cumulative error count across all files
    n_dist_errors = {}
    # distortion_key → count of decode refusals (NaN returned by decode_fn)
    n_decode_refusals = {}

    def _safe_result(fut, timeout=_SAFE_RESULT_TIMEOUT):
        """Return fut.result() or NaN on any failure / None future / timeout."""
        if fut is None:
            return float("nan")
        try:
            return fut.result(timeout=timeout)
        except Exception:
            return float("nan")

    _secs_client = _SecsClient() if compute_secs else None
    try:
        with ThreadPoolExecutor(max_workers=N_DIST_WORKERS) as _executor:
            for path in files:
                stem = path.stem
                try:
                    orig, sr = bu.load_audio_clipped(str(path), stem=stem)
                    wmed, wmed_sr = embed_fn(orig, sr)
                    wmed    = np.array(wmed, dtype=np.float32)
                    orig_rs = resample_if_needed(orig, sr, wmed_sr)
                    del orig
                    n       = min(len(orig_rs), len(wmed))
                    orig_rs = orig_rs[:n]
                    wmed_m  = wmed[:n]   # metric-aligned view; wmed kept full for decode
                except Exception as e:
                    print(f"    EMBED SKIP {stem}: {type(e).__name__}: {e}", flush=True)
                    continue

                n_embedded += 1
                if n_embedded == 1 or n_embedded % _LOG_INTERVAL == 0:
                    print(f"    [{n_embedded}/{len(all_files)}] embedded {stem}  "
                          f"({n/wmed_sr:.1f}s @ {wmed_sr}Hz)", flush=True)

                # ── Baseline metrics (bit_acc deferred to batch decode) ──────────
                _baseline_ok = False
                _bsn_pesq_fut = _bsn_estoi_fut = _bsn_visqol_fut = None
                try:
                    if compute_quality:
                        acc["no_distortion"]["si_snr"].append(
                            bu.compute_si_snr(orig_rs, wmed_m))
                        _bsn_pesq_fut = _executor.submit(
                            bu.compute_pesq, orig_rs.copy(), wmed_m.copy(), wmed_sr)
                        _bsn_estoi_fut = _executor.submit(
                            bu.compute_estoi, orig_rs.copy(), wmed_m.copy(), wmed_sr)
                        _bsn_visqol_fut = (None if _SKIP_VISQOL else _executor.submit(
                            bu.compute_visqol, orig_rs.copy(), wmed_m.copy(), wmed_sr, visqol_mode))
                    else:
                        acc["no_distortion"]["si_snr"].append(float("nan"))
                    _baseline_ok = True
                except Exception as e:
                    print(f"    SKIP baseline {stem}: {type(e).__name__}: {e}", flush=True)
                    acc["no_distortion"]["si_snr"].append(float("nan"))

                # ── Phase 1: apply distortions, SI-SNR sync, submit PESQ/ESTOI/ViSQOL ──
                dist_batch     = []   # [(key, dist_array_or_None)]
                pesq_futures   = []   # [(key, future_or_None)]
                estoi_futures  = []
                visqol_futures = []

                for key, dist_name, setting, label in dist_keys:
                    try:
                        dist   = bu.apply_distortion(wmed, wmed_sr, dist_name, setting, stem=stem)
                        dn     = min(len(dist), len(orig_rs))
                        dist_m = dist[:dn]
                        ref_dn = orig_rs[:dn]
                        if compute_quality:
                            acc[key]["si_snr"].append(bu.compute_si_snr(ref_dn, dist_m))
                            pesq_futures.append((key, _executor.submit(
                                bu.compute_pesq, ref_dn.copy(), dist_m.copy(), wmed_sr)))
                            estoi_futures.append((key, _executor.submit(
                                bu.compute_estoi, ref_dn.copy(), dist_m.copy(), wmed_sr)))
                            visqol_futures.append((key, None if _SKIP_VISQOL else _executor.submit(
                                bu.compute_visqol, ref_dn.copy(), dist_m.copy(), wmed_sr, visqol_mode)))
                        else:
                            acc[key]["si_snr"].append(float("nan"))
                            pesq_futures.append((key, None))
                            estoi_futures.append((key, None))
                            visqol_futures.append((key, None))
                        dist_batch.append((key, dist))   # full dist for decode
                    except Exception as e:
                        dk  = (dist_name, label)
                        cnt = n_dist_errors.get(dk, 0)
                        if cnt == 0:
                            print(f"    DIST ERR {stem} [{dist_name}/{label}]: "
                                  f"{type(e).__name__}: {e}", flush=True)
                        n_dist_errors[dk] = cnt + 1
                        acc[key]["si_snr"].append(float("nan"))
                        pesq_futures.append((key, None))
                        estoi_futures.append((key, None))
                        visqol_futures.append((key, None))
                        dist_batch.append((key, None))

                # ── Phase 1b: speaker-encoder cosine similarity ────────────────
                if compute_secs:
                    secs_keys = []
                    secs_wavs = []
                    if _baseline_ok:
                        secs_keys.append("no_distortion")
                        secs_wavs.append(wmed_m)
                    for key, dist in dist_batch:
                        if dist is not None:
                            dn = min(len(dist), len(orig_rs))
                            secs_keys.append(key)
                            secs_wavs.append(dist[:dn])
                    try:
                        secs_scores = _secs_client.compare_many(orig_rs, secs_wavs, wmed_sr)
                    except Exception as e:
                        print(f"    SECS ERR {stem}: {type(e).__name__}: {e}", flush=True)
                        secs_scores = [float('nan')] * len(secs_wavs)
                    for key, score in zip(secs_keys, secs_scores):
                        acc[key]["secs"].append(score)
                if not compute_secs:
                    acc["no_distortion"]["secs"].append(float("nan"))
                    for key, _dist in dist_batch:
                        acc[key]["secs"].append(float("nan"))
                else:
                    if not _baseline_ok:
                        acc["no_distortion"]["secs"].append(float("nan"))
                    for key, dist in dist_batch:
                        if dist is None:
                            acc[key]["secs"].append(float("nan"))

                # ── Phase 2: batch decode (baseline + all distortions) ────────────
                _decode_keys = []
                _decode_ys   = []
                if _baseline_ok:
                    _decode_keys.append("no_distortion")
                    _decode_ys.append(wmed)
                for key, dist in dist_batch:
                    if dist is not None:
                        _decode_keys.append(key)
                        _decode_ys.append(dist)

                del orig_rs, wmed, wmed_m   # free originals; dist copies held in dist_batch

                if _decode_ys:
                    try:
                        _bit_accs = decode_fn(_decode_ys, wmed_sr)
                        for key, ba in zip(_decode_keys, _bit_accs):
                            if ba != ba:  # NaN — decode refusal
                                n_decode_refusals[key] = n_decode_refusals.get(key, 0) + 1
                                acc[key]["bit"].append(0.0)
                            else:
                                acc[key]["bit"].append(ba)
                    except Exception as e:
                        print(f"    DECODE_BATCH ERR {stem}: {type(e).__name__}: {e}", flush=True)
                        for key in _decode_keys:
                            acc[key]["bit"].append(float("nan"))

                # NaN for baseline failure or any distortion that failed to apply
                if not _baseline_ok:
                    acc["no_distortion"]["bit"].append(float("nan"))
                for key, dist in dist_batch:
                    if dist is None:
                        acc[key]["bit"].append(float("nan"))

                del dist_batch   # free distorted audio arrays

                # ── Phase 3: collect baseline + distortion PESQ/ESTOI/ViSQOL ────
                if _baseline_ok and compute_quality:
                    acc["no_distortion"]["pesq"].append(_safe_result(_bsn_pesq_fut))
                    acc["no_distortion"]["estoi"].append(_safe_result(_bsn_estoi_fut))
                    acc["no_distortion"]["visqol"].append(_safe_result(_bsn_visqol_fut))
                else:
                    for k in ("pesq", "estoi", "visqol"):
                        acc["no_distortion"][k].append(float("nan"))

                for (key, pfut), (_, efut), (_, vfut) in zip(
                        pesq_futures, estoi_futures, visqol_futures):
                    acc[key]["pesq"].append(_safe_result(pfut))
                    acc[key]["estoi"].append(_safe_result(efut))
                    acc[key]["visqol"].append(_safe_result(vfut))

                # Mark file as fully processed and checkpoint periodically
                files_done.add(stem)
                if n_embedded % _LOG_INTERVAL == 0:
                    _save_checkpoint(ckpt_path, n_embedded, files_done, acc, dist_keys)
    finally:
        if _secs_client is not None:
            _secs_client.close()

    # ── Finalise ──────────────────────────────────────────────────────────────
    if n_embedded == 0:
        print(f"  ERROR: no files embedded for {dataset_key}", flush=True)
        return

    def _row(key, setting_val):
        a = acc[key]
        return {
            "setting":      setting_val,
            "bit_accuracy": _nanmean(a["bit"]),
            "si_snr":       _nanmean(a["si_snr"]),
            "pesq":         _nanmean(a["pesq"]),
            "estoi":        _nanmean(a["estoi"]),
            "visqol":       _nanmean(a["visqol"]),
            "secs":         _nanmean(a["secs"]),
            "n":            _ncount(a["bit"]),
            "n_si_snr":     _ncount(a["si_snr"]),
            "n_pesq":       _ncount(a["pesq"]),
            "n_estoi":      _ncount(a["estoi"]),
            "n_visqol":     _ncount(a["visqol"]),
            "n_secs":       _ncount(a["secs"]),
        }

    results = {"no_distortion": _row("no_distortion", "none")}
    r = results["no_distortion"]
    print(f"\n  [no_distortion]  bit_acc={r['bit_accuracy']:.3f}  "
          f"si_snr={r['si_snr']:.2f}  pesq={r['pesq']:.3f}  "
          f"estoi={r['estoi']:.3f}  visqol={r['visqol']:.3f}  "
          f"secs={r['secs']:.3f}", flush=True)

    for dist_name, settings in distortions.items():
        results[dist_name] = {}
        for setting in settings:
            label = bu.setting_label(setting)
            results[dist_name][label] = _row((dist_name, label), setting)

    # Pipeline metadata: counts + any distortion error summary
    results["_meta"] = {
        "n_processed":      n_embedded,
        "n_total":          len(all_files),
        "n_failed_embed":   len(all_files) - n_embedded,
        "distortions_hash": _distortions_hash(distortions),
        "clip_duration_s":  bu.MAX_AUDIO_DURATION,
        "dist_errors":      {
            f"{dn}/{lbl}": cnt
            for (dn, lbl), cnt in sorted(n_dist_errors.items())
        },
        "decode_refusals":  {
            (f"{k[0]}/{k[1]}" if isinstance(k, tuple) else k): cnt
            for k, cnt in sorted(n_decode_refusals.items(), key=str)
        },
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {out_path}  ({n_embedded}/{len(all_files)} files)", flush=True)

    # Remove checkpoint now that final result is saved
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"  Checkpoint removed.", flush=True)
