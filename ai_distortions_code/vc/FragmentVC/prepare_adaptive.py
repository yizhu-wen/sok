import argparse
import sys
from pathlib import Path


VC_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(VC_DIR))

from common.experiment import link_or_copy_tree_flat  # noqa: E402


def resolve_path(path):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return str(path)
    return str(SCRIPT_DIR / path)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare clean victim recordings for adaptive FragmentVC fine-tuning.")
    parser.add_argument("--clean-dir", required=True, help="directory containing clean victim wavs")
    parser.add_argument("--output-dir", required=True, help="output feature directory consumed by train.py")
    parser.add_argument("--work-dir", default="experiments/adaptive_clean", help="temporary speaker-organized wav directory")
    parser.add_argument("--copy-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--wav2vec-path", default="checkpoints/wav2vec_small.pt")
    parser.add_argument("--trim-method", choices=["librosa", "vad"], default="vad")
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--preemph", type=float, default=0.97)
    parser.add_argument("--hop-len", type=int, default=326)
    parser.add_argument("--win-len", type=int, default=1304)
    parser.add_argument("--n-fft", type=int, default=1304)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--f-min", type=int, default=80)
    return parser.parse_args()


def main():
    args = parse_args()
    from preprocess import main as preprocess_main

    speaker_dir = Path(args.work_dir) / "victim"
    link_or_copy_tree_flat(args.clean_dir, speaker_dir, args.copy_mode)

    preprocess_main(
        data_dirs=[str(Path(args.work_dir))],
        wav2vec_path=resolve_path(args.wav2vec_path),
        out_dir=args.output_dir,
        trim_method=args.trim_method,
        n_workers=args.n_workers,
        sample_rate=args.sample_rate,
        preemph=args.preemph,
        hop_len=args.hop_len,
        win_len=args.win_len,
        n_fft=args.n_fft,
        n_mels=args.n_mels,
        f_min=args.f_min,
    )


if __name__ == "__main__":
    main()
