#!/usr/bin/env python3
"""Add reviewer-page metrics to demo_audio_clips/manifest.json.

Metrics added per row:
  - bit_accuracy: decoded source-file bit recovery for physical/AI demos, or
    benchmark mean across available algorithms for digital distortion settings.
  - visqol: ViSQOL MOS-LQO against the best available reference. Speech rows use
    speech mode; music rows use audio mode.
  - secs: speaker encoder cosine similarity for AI-induced speech demos.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENT_ROOT = Path("/home/yizhu/Desktop/Production/PycharmProjects/sok_experiment")
DEFAULT_DATASET_ROOT = Path("/home/yizhu/Storage/sok_experiment/dataset")
DEFAULT_LJS_ROOT = Path("/media/yizhu/Data/LJSpeech-1.1/wavs")
DEFAULT_LJS_ALT_ROOT = Path("/media/yizhu/Storage/sok_experiment/dataset__/speech/LJSpeech-1.1")
DEFAULT_SECS_PYTHON = Path("/home/yizhu/Production/anaconda3/envs/timbrewatermark/bin/python")

MANIFEST_PATH = PROJECT_DIR / "demo_audio_clips" / "manifest.json"
BIT_PATH = PROJECT_DIR / "demo_audio_clips" / "bit_accuracy.json"
CACHE_PATH = PROJECT_DIR / "demo_audio_clips" / "metrics_cache.json"
SUMMARY_PATH = PROJECT_DIR / "demo_audio_clips" / "metrics_summary.json"

DATASET_KEYS = {
    "LJSpeech-1.1": "speech_ljspeech",
    "daps": "speech_daps",
    "gigaspeech": "speech_gigaspeech",
    "m4singer": "music_m4singer",
    "moisesdb": "music_moisesdb",
}

PHYSICAL_ALGO_MAP = {
    "fvsc": "fsvc",
    "fsvc": "fsvc",
    "norm_space": "normspace",
    "normspace": "normspace",
    "dnn_watermark": "dnn_watermark",
    "robustdnn": "dnn_watermark",
    "audioseal": "audioseal",
    "wavmark": "wavmark",
    "silentcipher": "silentcipher",
    "timbre": "timbre",
    "audiowmark": "audiowmark",
    "audiowavmark": "audiowmark",
    "patchwork": "patchwork",
}


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def rounded(value, digits=3):
    return None if not finite(value) else round(float(value), digits)


def load_audio(path: Path):
    import soundfile as sf

    y, sr = sf.read(str(path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32), int(sr)


def compute_visqol(ref_path: Path, deg_path: Path, mode: str) -> float:
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))
    import benchmark_utils as bu

    ref, ref_sr = load_audio(ref_path)
    deg, deg_sr = load_audio(deg_path)
    if deg_sr != ref_sr:
        import librosa

        deg = librosa.resample(deg, orig_sr=deg_sr, target_sr=ref_sr).astype(np.float32)
    return bu.compute_visqol(ref, deg, ref_sr, mode=mode)


def cache_key(kind: str, *paths_or_values) -> str:
    parts = [kind]
    for value in paths_or_values:
        if isinstance(value, Path):
            try:
                stat = value.stat()
                parts.append(f"{value}|{stat.st_size}|{int(stat.st_mtime)}")
            except OSError:
                parts.append(str(value))
        else:
            parts.append(str(value))
    return "||".join(parts)


def read_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {"visqol": {}, "secs": {}}


def write_cache(cache: dict) -> None:
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    tmp.replace(CACHE_PATH)


def read_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def write_manifest(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, indent=2) + "\n")
    tmp.replace(path)


def read_bit_scores() -> dict:
    if not BIT_PATH.exists():
        return {}
    with BIT_PATH.open() as f:
        return json.load(f).get("sources", {})


def setting_json_key(setting_dir: str) -> str:
    if setting_dir.startswith("plus"):
        return setting_dir[4:]
    return setting_dir


def digital_components(source: str):
    parts = Path(source).parts
    # distortion_demo_audio/{domain_attack}/{distortion}/{setting}/{dataset}/{file...}
    if len(parts) < 6 or parts[0] != "distortion_demo_audio":
        return None
    domain_attack, dist_key, setting_dir, dataset = parts[1:5]
    domain = domain_attack.split("_", 1)[0]
    relative_file = Path(*parts[4:])
    return domain, domain_attack, dist_key, setting_dir, dataset, relative_file


def physical_relative_suffix(path_value: str) -> str | None:
    normalized = path_value.replace("\\", "/")
    marker = "/physical/"
    if marker not in normalized:
        return None
    return "physical/" + normalized.split(marker, 1)[1]


def load_digital_visqol_csvs() -> dict[str, float]:
    values = {}
    for csv_path in (PROJECT_DIR / "distortion_demo_audio").glob("*/visqol_audio_per_file.csv"):
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                value = row.get("visqol_audio")
                if not finite(value):
                    continue
                distorted = row.get("distorted_file", "").replace("\\", "/")
                marker = "/dataset/"
                if marker not in distorted:
                    continue
                suffix = distorted.split(marker, 1)[1]
                values[suffix] = float(value)
    return values


def load_physical_visqol(experiment_root: Path) -> dict[str, dict]:
    candidates = [
        experiment_root / "results" / "custom" / "physical_visqol_speech_9algos.csv",
        PROJECT_DIR / "results" / "custom" / "physical_visqol_speech_9algos.csv",
    ]
    mapping = {}
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        return mapping
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            suffix = physical_relative_suffix(row.get("target_file", ""))
            if not suffix:
                continue
            mapping[suffix] = {
                "visqol": rounded(row.get("visqol_speech")),
                "reference": row.get("reference_file") or "",
                "match_method": row.get("match_method") or "",
            }
    return mapping


def load_filelist() -> list[str]:
    filelist = PROJECT_DIR / "ai_distortions_code" / "tts" / "YourTTS" / "ljs_audio_text_test_filelist.txt"
    if not filelist.exists():
        filelist = PROJECT_DIR / "ai_distortions_code" / "vc" / "ljs_audio_text_test_filelist.txt"
    entries = []
    with filelist.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            entries.append(raw.split("|", 1)[0])
    return entries


def first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def digital_reference(row: dict, dataset_root: Path) -> Path | None:
    components = digital_components(row["source"])
    if components is None:
        return None
    domain, _domain_attack, _dist_key, _setting_dir, dataset, relative_file = components
    filename = Path(row["selected_file"])
    candidates = [
        dataset_root / domain / dataset / filename,
        Path("/media/yizhu/Storage/sok_experiment/dataset") / domain / dataset / filename,
        Path("/media/yizhu/Storage/sok_experiment/dataset__") / domain / dataset / filename,
    ]
    if dataset == "LJSpeech-1.1":
        candidates.extend([
            DEFAULT_LJS_ROOT / filename,
            DEFAULT_LJS_ALT_ROOT / filename,
        ])
    return first_existing(candidates)


def ai_reference(row: dict, filelist: list[str]) -> Path | None:
    try:
        index = int(Path(row["selected_file"]).stem)
    except ValueError:
        return None
    if index < 1 or index > len(filelist):
        return None
    filename = filelist[index - 1]
    candidates = [
        DEFAULT_LJS_ROOT / filename,
        DEFAULT_LJS_ALT_ROOT / filename,
        DEFAULT_DATASET_ROOT / "speech" / "LJSpeech-1.1" / filename,
    ]
    return first_existing(candidates)


def physical_reference_from_csv(info: dict) -> Path | None:
    raw = info.get("reference", "")
    if not raw:
        return None
    path = Path(raw)
    if path.exists():
        return path
    suffix = physical_relative_suffix(raw)
    if suffix:
        local = PROJECT_DIR / suffix
        if local.exists():
            return local
    return None


def visqol_mode(row: dict) -> str:
    if row["category"] == "Digital-level":
        components = digital_components(row["source"])
        if components and components[0] == "music":
            return "audio"
    return "speech"


def benchmark_bit_accuracy(row: dict, results_root: Path) -> dict:
    components = digital_components(row["source"])
    if components is None:
        return {"value": None, "n": 0}
    _domain, _domain_attack, dist_key, setting_dir, dataset, _relative_file = components
    dataset_key = DATASET_KEYS.get(dataset)
    if not dataset_key:
        return {"value": None, "n": 0}
    setting_key = setting_json_key(setting_dir)
    values = []
    for json_path in sorted((results_root / "benchmark" / dataset_key).glob("*.json")):
        try:
            data = json.loads(json_path.read_text())
            metric = data.get(dist_key, {}).get(setting_key, {}).get("bit_accuracy")
        except Exception:
            continue
        if finite(metric):
            values.append(float(metric))
    if not values:
        return {"value": None, "n": 0}
    return {
        "value": round(float(np.mean(values)), 6),
        "n": len(values),
        "min": round(float(np.min(values)), 6),
        "max": round(float(np.max(values)), 6),
    }


class SecsWorker:
    def __init__(self, python: Path):
        self.proc = None
        self.python = python

    def __enter__(self):
        if not self.python.exists():
            return self
        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
        self.proc = subprocess.Popen(
            [str(self.python), str(PROJECT_DIR / "scripts" / "resemblyzer_secs_server.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.proc and self.proc.poll() is None:
            try:
                self.request({"cmd": "shutdown"})
            except Exception:
                pass
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def request(self, payload: dict) -> dict:
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("SECS worker unavailable")
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"SECS worker exited: {err.strip()}")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "unknown SECS error"))
        return reply

    def compare_many(self, ref: Path, degs: list[Path]) -> list[float | None]:
        if self.proc is None:
            return [None] * len(degs)
        reply = self.request({
            "cmd": "compare_many",
            "ref_path": str(ref),
            "deg_paths": [str(path) for path in degs],
        })
        return reply.get("scores", [])


def enrich_rows(args) -> tuple[list[dict], dict]:
    rows = read_manifest(args.manifest)
    cache = read_cache()
    bit_scores = read_bit_scores()
    digital_csv_visqol = load_digital_visqol_csvs()
    physical_visqol = load_physical_visqol(args.experiment_root)
    filelist = load_filelist()
    results_root = args.experiment_root / "results"

    summary = defaultdict(int)
    secs_groups: dict[Path, list[tuple[int, Path, str]]] = defaultdict(list)
    visqol_tasks: list[tuple[int, str, str, Path, Path]] = []

    for idx, row in enumerate(rows):
        source_rel = row["source"]
        source_path = PROJECT_DIR / source_rel
        metrics = {
            "bit_accuracy": None,
            "bit_accuracy_scope": "",
            "bit_accuracy_n": None,
            "visqol": None,
            "visqol_mode": visqol_mode(row),
            "visqol_reference": "",
            "secs": None,
            "secs_reference": "",
        }

        if row["category"] == "Digital-level":
            bit = benchmark_bit_accuracy(row, results_root)
            metrics["bit_accuracy"] = bit["value"]
            metrics["bit_accuracy_n"] = bit["n"]
            metrics["bit_accuracy_scope"] = (
                f"benchmark mean across {bit['n']} available algorithms"
                if bit["n"] else "benchmark metric unavailable"
            )
        else:
            bit = bit_scores.get(source_rel, {})
            metrics["bit_accuracy"] = bit.get("bit_accuracy")
            metrics["bit_accuracy_n"] = 1 if bit.get("bit_accuracy") is not None else 0
            metrics["bit_accuracy_scope"] = (
                "decoded from displayed source WAV"
                if bit.get("bit_accuracy") is not None else "source decode unavailable"
            )

        reference = None
        if row["category"] == "Digital-level":
            components = digital_components(source_rel)
            if components is not None:
                suffix = "/".join(Path(source_rel).parts[1:])
                if metrics["visqol_mode"] == "audio" and suffix in digital_csv_visqol:
                    metrics["visqol"] = rounded(digital_csv_visqol[suffix])
                reference = digital_reference(row, args.dataset_root)
        elif row["category"] == "Physical-level":
            info = physical_visqol.get(source_rel)
            if info:
                metrics["visqol"] = info["visqol"]
                reference = physical_reference_from_csv(info)
            if reference is None:
                # Fall back to a direct clean_watermarked filename match.
                parts = Path(source_rel).parts
                if len(parts) >= 5:
                    raw_algo = PHYSICAL_ALGO_MAP.get(parts[3], "")
                    stem = Path(row["selected_file"]).stem.rsplit("_", 1)[0]
                    candidates = [
                        PROJECT_DIR / "physical" / "clean_watermarked" / raw_algo / f"{stem}.wav",
                        PROJECT_DIR / "physical" / "clean_watermarked" / parts[-2] / f"{stem}.wav",
                    ]
                    reference = first_existing(candidates)
        elif row["category"] == "AI-induced":
            reference = ai_reference(row, filelist)
            if reference is not None:
                metrics["secs_reference"] = str(reference)
                secs_key = cache_key("secs", reference, source_path)
                cached = cache.get("secs", {}).get(secs_key)
                if cached is not None:
                    metrics["secs"] = rounded(cached)
                else:
                    secs_groups[reference].append((idx, source_path, secs_key))

        if reference is not None:
            metrics["visqol_reference"] = str(reference)
        if metrics["visqol"] is None and reference is not None and source_path.exists():
            vkey = cache_key("visqol", metrics["visqol_mode"], reference, source_path)
            cached = cache.get("visqol", {}).get(vkey)
            if cached is not None:
                metrics["visqol"] = rounded(cached)
            else:
                visqol_tasks.append((idx, vkey, metrics["visqol_mode"], reference, source_path))

        row["metrics"] = metrics
        summary[f"{row['category']} rows"] += 1
        if metrics["bit_accuracy"] is not None:
            summary[f"{row['category']} bit"] += 1
        if metrics["visqol"] is not None:
            summary[f"{row['category']} visqol"] += 1
        if metrics["secs"] is not None:
            summary[f"{row['category']} secs"] += 1

    if visqol_tasks:
        print(
            f"Computing {len(visqol_tasks)} missing ViSQOL scores "
            f"with {args.visqol_workers} workers...",
            flush=True,
        )
        completed = 0
        with ProcessPoolExecutor(max_workers=args.visqol_workers) as executor:
            futures = {
                executor.submit(compute_visqol, ref, source, mode): (row_idx, key)
                for row_idx, key, mode, ref, source in visqol_tasks
            }
            for future in as_completed(futures):
                row_idx, key = futures[future]
                try:
                    score = future.result()
                except Exception:
                    score = float("nan")
                cache.setdefault("visqol", {})[key] = None if score != score else float(score)
                rows[row_idx]["metrics"]["visqol"] = rounded(score)
                if score == score:
                    summary[f"{rows[row_idx]['category']} visqol"] += 1
                completed += 1
                if completed % 25 == 0 or completed == len(visqol_tasks):
                    write_cache(cache)
                    print(f"  ViSQOL {completed}/{len(visqol_tasks)}", flush=True)

    pending_secs = sum(len(items) for items in secs_groups.values())
    if pending_secs and not args.skip_secs:
        with SecsWorker(args.secs_python) as worker:
            for ref, items in secs_groups.items():
                scores = worker.compare_many(ref, [item[1] for item in items])
                for (row_idx, _path, key), score in zip(items, scores):
                    cache.setdefault("secs", {})[key] = None if score is None else float(score)
                    rows[row_idx]["metrics"]["secs"] = rounded(score)
                    if score is not None:
                        summary[f"{rows[row_idx]['category']} secs"] += 1
                write_cache(cache)

    return rows, dict(summary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=MANIFEST_PATH, type=Path)
    parser.add_argument("--experiment-root", default=DEFAULT_EXPERIMENT_ROOT, type=Path)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, type=Path)
    parser.add_argument("--secs-python", default=DEFAULT_SECS_PYTHON, type=Path)
    parser.add_argument("--skip-secs", action="store_true")
    parser.add_argument("--visqol-workers", type=int, default=min(6, os.cpu_count() or 2))
    args = parser.parse_args()

    rows, summary = enrich_rows(args)
    write_manifest(args.manifest, rows)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
