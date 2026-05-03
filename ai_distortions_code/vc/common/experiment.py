import os
import shutil
from pathlib import Path


VC_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILELIST = VC_ROOT / "ljs_audio_text_test_filelist.txt"


def read_filelist(filelist_path=DEFAULT_FILELIST, limit=None):
    """Read rows formatted as filename|transcript and return (filename, text)."""
    entries = []
    path = Path(filelist_path)
    with path.open(encoding="utf-8") as file:
        for line_no, raw_line in enumerate(file, 1):
            line = raw_line.strip()
            if not line:
                continue
            if "|" not in line:
                raise ValueError(f"{path}:{line_no} is not formatted as filename|text")
            filename, text = line.split("|", 1)
            entries.append((filename.strip(), text.strip()))
            if limit is not None and len(entries) >= limit:
                break
    return entries


def existing_pairs(source_dir, target_dir, filelist_path=DEFAULT_FILELIST, limit=None):
    """Resolve paired source and target wavs by filelist filename."""
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory does not exist: {target_dir}")

    pairs = []
    missing = []
    for filename, text in read_filelist(filelist_path, limit):
        source_file = source_dir / filename
        target_file = target_dir / filename
        if not source_file.exists():
            missing.append(str(source_file))
        if not target_file.exists():
            missing.append(str(target_file))
        pairs.append((filename, text, source_file, target_file))

    if missing:
        preview = "\n".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
        raise FileNotFoundError(f"Missing {len(missing)} source/target wav files:\n{preview}{suffix}")
    return pairs


def wav_files(path):
    return sorted(str(item) for item in Path(path).iterdir() if item.suffix.lower() == ".wav")


def link_or_copy_tree_flat(source_dir, output_dir, copy_mode="symlink"):
    """Make a flat wav directory for training/preprocessing from clean victim recordings."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for src in sorted(source_dir.glob("*.wav")):
        dst = output_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if copy_mode == "symlink":
            os.symlink(os.path.relpath(src.resolve(), dst.parent.resolve()), dst)
        elif copy_mode == "hardlink":
            os.link(src, dst)
        elif copy_mode == "copy":
            shutil.copy2(src, dst)
        else:
            raise ValueError(f"Unsupported copy mode: {copy_mode}")
        copied.append(dst)
    if not copied:
        raise FileNotFoundError(f"No .wav files found under {source_dir}")
    return copied


def add_common_vc_args(parser):
    parser.add_argument("--source", "-s", required=True, help="watermarked source wav directory")
    parser.add_argument("--target", "-t", required=True, help="clean victim reference wav directory")
    parser.add_argument("--output", "-o", required=True, help="converted/<output> directory name")
    parser.add_argument("--filelist", default=str(DEFAULT_FILELIST), help="filename|text filelist used for conversion")
    parser.add_argument("--limit", type=int, help="limit number of utterances")
    return parser
