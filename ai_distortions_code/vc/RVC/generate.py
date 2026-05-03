import argparse
import os
import sys
from pathlib import Path


VC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VC_DIR))

from common.experiment import DEFAULT_FILELIST, read_filelist, wav_files  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run RVC conversion with the RVC-Project WebUI backend.")
    parser.add_argument("--source", "-s", required=True, help="watermarked source wav directory")
    parser.add_argument("--target", "-t", required=True, help="clean victim recordings directory; used to document/validate the target speaker set")
    parser.add_argument("--output", "-o", required=True, help="converted/<output> directory name")
    parser.add_argument("--filelist", default=str(DEFAULT_FILELIST), help="filename|text filelist used for conversion")
    parser.add_argument("--limit", type=int, help="limit number of utterances")
    parser.add_argument("--rvc-repo", default=os.environ.get("RVC_REPO"), help="path to RVC-Project/Retrieval-based-Voice-Conversion-WebUI")
    parser.add_argument("--model-path", required=True, help="trained RVC .pth model for the clean victim voice")
    parser.add_argument("--index-path", help="optional RVC .index retrieval file for the clean victim voice")
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--f0-up-key", type=int, default=0)
    parser.add_argument("--f0-method", default="rmvpe", choices=["pm", "harvest", "dio", "crepe", "crepe-tiny", "rmvpe"])
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--filter-radius", type=int, default=3)
    parser.add_argument("--resample-sr", type=int, default=0)
    parser.add_argument("--rms-mix-rate", type=float, default=0.25)
    parser.add_argument("--protect", type=float, default=0.33)
    return parser.parse_args()


def require_rvc_repo(path):
    if not path:
        raise SystemExit("Pass --rvc-repo or set RVC_REPO to the RVC WebUI repository path.")
    repo = Path(path).resolve()
    expected = repo / "infer" / "modules" / "vc" / "modules.py"
    if not expected.exists():
        raise FileNotFoundError(f"Not an RVC WebUI checkout: missing {expected}")
    return repo


def load_rvc(repo, model_path, index_path):
    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Missing RVC model: {model_path}")
    index_path = Path(index_path).resolve() if index_path else None
    if index_path and not index_path.exists():
        raise FileNotFoundError(f"Missing RVC index: {index_path}")

    os.environ["weight_root"] = str(model_path.parent)
    index_root = index_path.parent if index_path else model_path.parent
    os.environ["index_root"] = str(index_root)
    os.environ["outside_index_root"] = str(index_root)

    os.chdir(repo)
    sys.path.insert(0, str(repo))

    from configs.config import Config
    from infer.modules.vc.modules import VC

    vc = VC(Config())
    vc.get_vc(model_path.name)
    return vc, str(index_path) if index_path else ""


def write_wav(path, sample_rate, audio):
    try:
        import soundfile as sf

        sf.write(path, audio, sample_rate, subtype="PCM_16")
    except ModuleNotFoundError:
        from scipy.io import wavfile

        wavfile.write(path, sample_rate, audio)


def main():
    args = parse_args()
    repo = require_rvc_repo(args.rvc_repo)
    source_dir = Path(args.source)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not wav_files(args.target):
        raise FileNotFoundError(f"No clean target wavs found under {args.target}")

    entries = read_filelist(args.filelist, args.limit)
    out_dir = Path("converted") / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir.resolve()

    old_cwd = Path.cwd()
    try:
        vc, index_path = load_rvc(repo, args.model_path, args.index_path)
        for index, (filename, _text) in enumerate(entries, 1):
            source_file = source_dir / filename
            if not source_file.exists():
                raise FileNotFoundError(f"Missing source wav: {source_file}")
            info, result = vc.vc_single(
                args.speaker_id,
                str(source_file.resolve()),
                args.f0_up_key,
                None,
                args.f0_method,
                index_path,
                "",
                args.index_rate,
                args.filter_radius,
                args.resample_sr,
                args.rms_mix_rate,
                args.protect,
            )
            sample_rate, audio = result
            if sample_rate is None or audio is None:
                raise RuntimeError(f"RVC failed for {source_file}: {info}")
            write_wav(out_dir / f"{index}.wav", sample_rate, audio)
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
