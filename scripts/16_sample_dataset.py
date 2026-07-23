"""
Reproduce the paper's dataset sampling and build the benchmark dataset tree.

Sampling counts:
  - LJSpeech    : randomly sample 2,000 of the 13,100 clips
  - LibriSpeech : randomly sample 2,000 clips
  - DAPS        : use the full dataset
  - M4Singer    : randomly sample 2,000 clips
  - MoisesDB    : randomly select 2,000 clips

The output tree matches what scripts/large_scale_utils.py expects under
SOK_DATASET_DIR:

    <out>/speech/LJSpeech-1.1/...
    <out>/speech/LibriSpeech/...
    <out>/speech/daps/...
    <out>/music/m4singer/...
    <out>/music/moisesdb/...

Sampling is deterministic (--seed, default 42) and recorded in
<out>/sample_manifest.json.  Non-WAV sources (e.g. LibriSpeech FLAC) are
converted to WAV, because the benchmark loader globs *.wav.

Usage:
    python3 scripts/16_sample_dataset.py \
        --raw ljspeech=/data/LJSpeech-1.1 \
        --raw librispeech=/data/LibriSpeech/test-clean \
        --raw daps=/data/daps \
        --raw m4singer=/data/m4singer \
        --raw moisesdb=/data/moisesdb \
        --out /data/sok_dataset \
        --zip sok_dataset_sample.zip

Any subset of --raw entries may be given; only those datasets are built.
The optional --zip archive is what you would upload for sharing (e.g. to
Google Drive).  Check each dataset's license before redistributing —
LJSpeech (public domain), LibriSpeech (CC BY 4.0) and DAPS are shareable;
M4Singer and MoisesDB have research licenses that may not permit
redistribution.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

# key -> (output subdir, sample count (None = all), source audio extensions)
DATASETS = {
    "ljspeech":    ("speech/LJSpeech-1.1", 2000, (".wav",)),
    "librispeech": ("speech/LibriSpeech",  2000, (".flac", ".wav")),
    "daps":        ("speech/daps",         None, (".wav",)),
    "m4singer":    ("music/m4singer",      2000, (".wav",)),
    "moisesdb":    ("music/moisesdb",      2000, (".wav",)),
}


def _find_audio(root: Path, exts) -> list[Path]:
    files = [p for p in sorted(root.rglob("*"))
             if p.suffix.lower() in exts and not p.name.startswith("._")]
    if not files:
        raise SystemExit(f"error: no {'/'.join(exts)} files found under {root}")
    return files


def _sample(files: list[Path], n, seed: int) -> list[Path]:
    if n is None or n >= len(files):
        return files
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(files), size=n, replace=False))
    return [files[i] for i in idx]


def _unique_name(src: Path, root: Path) -> str:
    """Flatten the source-relative path into a unique file name."""
    rel = src.relative_to(root)
    if len(rel.parts) == 1:
        return rel.name
    return "__".join(rel.parts[:-1]) + "__" + rel.name


def _copy_as_wav(src: Path, dst: Path) -> None:
    """dst always has a .wav suffix; non-WAV sources are converted."""
    if src.suffix.lower() == ".wav":
        shutil.copy2(src, dst)
        return
    import soundfile as sf
    y, sr = sf.read(str(src))
    sf.write(str(dst), y, sr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", action="append", default=[], metavar="KEY=DIR",
                    help=f"dataset root, key in {sorted(DATASETS)} (repeatable)")
    ap.add_argument("--out", required=True, help="output dataset root "
                    "(point SOK_DATASET_DIR here afterwards)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zip", default=None, metavar="ZIPFILE",
                    help="also pack the output tree into this zip archive")
    args = ap.parse_args()

    if not args.raw:
        ap.error("at least one --raw key=dir is required")

    raw_dirs = {}
    for spec in args.raw:
        key, _, path = spec.partition("=")
        if key not in DATASETS or not path:
            ap.error(f"bad --raw {spec!r}; expected key=dir with key in {sorted(DATASETS)}")
        raw_dirs[key] = Path(path).expanduser()
        if not raw_dirs[key].is_dir():
            ap.error(f"--raw {key}: {path} is not a directory")

    out_root = Path(args.out).expanduser()
    manifest = {"seed": args.seed, "datasets": {}}

    for key, root in raw_dirs.items():
        subdir, n, exts = DATASETS[key]
        dst_dir = out_root / subdir
        dst_dir.mkdir(parents=True, exist_ok=True)
        files = _find_audio(root, exts)
        chosen = _sample(files, n, args.seed)
        print(f"[{key}] {len(chosen)}/{len(files)} files -> {dst_dir}", flush=True)
        names = []
        for src in chosen:
            name = str(Path(_unique_name(src, root)).with_suffix(".wav"))
            _copy_as_wav(src, dst_dir / name)
            names.append(name)
        manifest["datasets"][key] = {
            "source_root": str(root), "n_available": len(files),
            "n_selected": len(chosen), "output_dir": subdir, "files": names,
        }

    manifest_path = out_root / "sample_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest -> {manifest_path}", flush=True)

    if args.zip:
        zip_path = Path(args.zip).expanduser()
        print(f"Packing {out_root} -> {zip_path} ...", flush=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for p in sorted(out_root.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(out_root))
        print(f"Done: {zip_path} "
              f"({zip_path.stat().st_size / 1e9:.2f} GB)", flush=True)

    print("\nNext: export SOK_DATASET_DIR=" + str(out_root), flush=True)


if __name__ == "__main__":
    main()
