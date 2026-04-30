"""
Pre-flight checks for the large-scale watermarking benchmark.

Run from the project root with any environment (envs/viz recommended):
    envs/viz/bin/python scripts/preflight.py

Checks:
  1. config.py paths (STORAGE_DIR, DATASET_DIR, AUDIOWMARK_BIN, AUDIOWMARK_LIB,
     NVIDIA_BASE, PTXAS_DIR)
  2. Each of the 7 datasets — directory exists and contains WAV files
  3. Background noise (DEMAND) — spot-checks first folder
  4. Reverberation (AIR) — spot-checks first folder
  5. ffmpeg binary is on PATH
  6. audiowmark binary is executable
  7. benchmark_utils + large_scale_utils import cleanly
  8. Metric packages: pesq, pystoi, visqol
  9. Results output directory is writable
"""
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PASS = "  [OK]  "
WARN = "  [WARN]"
FAIL = "  [FAIL]"

issues = []


def ok(msg):
    print(f"{PASS} {msg}", flush=True)


def warn(msg):
    print(f"{WARN} {msg}", flush=True)
    issues.append(("WARN", msg))


def fail(msg):
    print(f"{FAIL} {msg}", flush=True)
    issues.append(("FAIL", msg))


# ── 1. config.py paths ────────────────────────────────────────────────────────
print("\n=== 1. config.py paths ===")
try:
    from config import (STORAGE_DIR, DATASET_DIR, AUDIOWMARK_BIN,
                        AUDIOWMARK_LIB, NVIDIA_BASE, PTXAS_DIR)

    for name, path in [
        ("STORAGE_DIR",    STORAGE_DIR),
        ("DATASET_DIR",    DATASET_DIR),
        ("AUDIOWMARK_BIN", AUDIOWMARK_BIN),
        ("AUDIOWMARK_LIB", AUDIOWMARK_LIB),
    ]:
        if path is None:
            warn(f"{name} is None (not configured)")
        elif Path(path).exists():
            ok(f"{name} = {path}")
        else:
            fail(f"{name} not found: {path}")

    for name, path in [("NVIDIA_BASE", NVIDIA_BASE), ("PTXAS_DIR", PTXAS_DIR)]:
        if path is None:
            warn(f"{name} is None (GPU-only scripts may fail)")
        elif Path(path).exists():
            ok(f"{name} = {path}")
        else:
            warn(f"{name} not found: {path}  (GPU-only scripts may fail)")

except Exception as e:
    fail(f"config.py import failed: {e}")
    DATASET_DIR = None
    AUDIOWMARK_BIN = None
    NVIDIA_BASE = None


# ── 2. Datasets ───────────────────────────────────────────────────────────────
print("\n=== 2. Datasets ===")
try:
    from large_scale_utils import DATASETS

    for dataset_key, category, folder in DATASETS:
        if DATASET_DIR is None:
            fail(f"{dataset_key}: DATASET_DIR not set")
            continue
        path = Path(DATASET_DIR) / category / folder
        if not path.exists():
            fail(f"{dataset_key}: directory not found — {path}")
        else:
            wavs = list(path.rglob("*.wav"))
            n = len(wavs)
            if n == 0:
                fail(f"{dataset_key}: no WAV files in {path}")
            else:
                ok(f"{dataset_key}: {n} WAV files")
except Exception as e:
    fail(f"DATASETS check failed: {e}")


# ── 3. Background noise (DEMAND) ──────────────────────────────────────────────
print("\n=== 3. Background noise (DEMAND) ===")
try:
    from config import NOISE_DIR
    if NOISE_DIR is None or not Path(NOISE_DIR).exists():
        fail(f"NOISE_DIR not found: {NOISE_DIR}")
    else:
        folders = sorted(Path(NOISE_DIR).iterdir())
        n_found = sum(1 for f in folders if f.is_dir())
        if n_found == 0:
            fail(f"NOISE_DIR empty: {NOISE_DIR}")
        else:
            # spot-check first folder has ch01.wav
            first = next(f for f in folders if f.is_dir())
            ch01 = first / "ch01.wav"
            if ch01.exists():
                ok(f"DEMAND noise OK — {n_found} folders, spot-check {first.name}/ch01.wav found")
            else:
                warn(f"DEMAND noise folder {first.name} missing ch01.wav")
except Exception as e:
    fail(f"DEMAND noise check failed: {e}")


# ── 4. Reverberation (AIR) ───────────────────────────────────────────────────
print("\n=== 4. Reverberation (AIR) ===")
try:
    from config import RIR_DIR
    if RIR_DIR is None or not Path(RIR_DIR).exists():
        fail(f"RIR_DIR not found: {RIR_DIR}")
    else:
        mat_files = list(Path(RIR_DIR).rglob("*.mat"))
        if not mat_files:
            fail(f"RIR_DIR contains no .mat files: {RIR_DIR}")
        else:
            ok(f"AIR reverberation OK — {len(mat_files)} .mat files")
except Exception as e:
    fail(f"AIR RIR check failed: {e}")


# ── 5. ffmpeg ─────────────────────────────────────────────────────────────────
print("\n=== 5. ffmpeg ===")
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path:
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        ver_line = result.stdout.splitlines()[0] if result.stdout else "?"
        ok(f"ffmpeg found at {ffmpeg_path} — {ver_line}")
    except Exception as e:
        fail(f"ffmpeg found but failed to run: {e}")
else:
    fail("ffmpeg not on PATH — mp3_compression distortion will fail")


# ── 6. audiowmark ─────────────────────────────────────────────────────────────
print("\n=== 6. audiowmark ===")
try:
    aw_bin = Path(AUDIOWMARK_BIN) if AUDIOWMARK_BIN else None
    if aw_bin is None:
        fail("AUDIOWMARK_BIN is None")
    elif not aw_bin.exists():
        fail(f"audiowmark binary not found: {aw_bin}")
    elif not aw_bin.stat().st_mode & 0o111:
        fail(f"audiowmark binary not executable: {aw_bin}")
    else:
        ok(f"audiowmark binary OK: {aw_bin}")
except Exception as e:
    fail(f"audiowmark check failed: {e}")


# ── 7. Core imports ───────────────────────────────────────────────────────────
print("\n=== 7. Core imports ===")
for mod in ("benchmark_utils", "large_scale_utils"):
    try:
        importlib.import_module(mod)
        ok(f"{mod} imports cleanly")
    except Exception as e:
        fail(f"{mod} import failed: {e}")


# ── 8. Metric packages ────────────────────────────────────────────────────────
print("\n=== 8. Metric packages ===")
for pkg, label in [("pesq", "pesq"), ("pystoi", "pystoi"), ("visqol", "visqol-python")]:
    try:
        importlib.import_module(pkg)
        ok(f"{label} importable")
    except ImportError:
        fail(f"{label} not installed in this env — quality metrics will be NaN")


# ── 9. Output directory writable ─────────────────────────────────────────────
print("\n=== 9. Output directories ===")
PROJECT_DIR = Path(__file__).resolve().parent.parent
for sub in ["results/benchmark", "logs"]:
    out = PROJECT_DIR / sub
    out.mkdir(parents=True, exist_ok=True)
    test_file = out / ".preflight_write_test"
    try:
        test_file.write_text("ok")
        test_file.unlink()
        ok(f"{out} is writable")
    except Exception as e:
        fail(f"{out} not writable: {e}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
n_fail = sum(1 for level, _ in issues if level == "FAIL")
n_warn = sum(1 for level, _ in issues if level == "WARN")
if not issues:
    print("All checks passed.")
else:
    if n_fail:
        print(f"{n_fail} FAIL(s), {n_warn} WARN(s):")
    else:
        print(f"0 failures, {n_warn} WARN(s):")
    for level, msg in issues:
        print(f"  [{level}] {msg}")
sys.exit(1 if n_fail else 0)
