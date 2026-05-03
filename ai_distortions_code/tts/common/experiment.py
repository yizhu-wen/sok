import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


TTS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILELIST = TTS_ROOT / "YourTTS" / "ljs_audio_text_test_filelist.txt"


def read_filelist(filelist_path, limit=None):
    """Read repo filelists with rows formatted as filename|transcript."""
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


def _replace_existing_link(path):
    if path.is_symlink():
        path.unlink()


def _link_or_copy(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    _replace_existing_link(dst)

    if mode == "symlink":
        rel_src = os.path.relpath(src.resolve(), dst.parent.resolve())
        os.symlink(rel_src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported copy mode: {mode}")


def prepare_ljspeech_dataset(audio_dir, filelist_path, dataset_dir, copy_mode="symlink", limit=None):
    """Prepare a Coqui-compatible single-speaker LJSpeech-style dataset.

    The input filelist rows use the repository convention:
        LJ001-0001.wav|Transcript text

    Coqui's LJSpeech formatter expects:
        metadata.csv rows: LJ001-0001|raw text|normalized text
        wavs/LJ001-0001.wav
    """
    audio_dir = Path(audio_dir)
    dataset_dir = Path(dataset_dir)
    wavs_dir = dataset_dir / "wavs"
    metadata_path = dataset_dir / "metadata.csv"

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory does not exist: {audio_dir}")

    entries = read_filelist(filelist_path, limit=limit)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    missing = []
    for filename, text in entries:
        src = audio_dir / filename
        if not src.exists():
            missing.append(str(src))
            continue
        utt_id = Path(filename).stem
        dst = wavs_dir / f"{utt_id}.wav"
        _link_or_copy(src, dst, copy_mode)
        metadata_rows.append(f"{utt_id}|{text}|{text}")

    if missing:
        preview = "\n".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
        raise FileNotFoundError(f"Missing {len(missing)} wav files:\n{preview}{suffix}")

    with metadata_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(metadata_rows))
        file.write("\n")

    return {
        "dataset_dir": dataset_dir,
        "metadata_path": metadata_path,
        "num_items": len(metadata_rows),
    }


def run_command(command, cwd=None):
    printable = " ".join(str(part) for part in command)
    print(f"\n$ {printable}\n")
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def find_best_checkpoint(path):
    root = Path(path)
    if root.is_file():
        return root
    best = sorted(root.rglob("best_model*.pth"), key=lambda item: item.stat().st_mtime, reverse=True)
    if best:
        return best[0]
    checkpoints = sorted(root.rglob("checkpoint*.pth"), key=lambda item: item.stat().st_mtime, reverse=True)
    if checkpoints:
        return checkpoints[0]
    raise FileNotFoundError(f"No Coqui checkpoint found under {root}")


def find_config(path):
    root = Path(path)
    if root.is_file():
        return root
    configs = sorted(root.rglob("config.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    if configs:
        return configs[0]
    raise FileNotFoundError(f"No config.json found under {root}")


def load_tts_model(args):
    try:
        import torch
        from TTS.api import TTS
    except ModuleNotFoundError as exc:
        raise SystemExit("Install dependencies first, for example: pip install -r requirements.txt") from exc

    init_kwargs = {"progress_bar": False}
    if args.model_path:
        model_ref = Path(args.model_path)
        config_ref = args.config_path or (model_ref if model_ref.is_dir() else model_ref.parent)
        init_kwargs["model_path"] = str(find_best_checkpoint(model_ref))
        init_kwargs["config_path"] = str(find_config(config_ref))
    else:
        init_kwargs["model_name"] = args.model_name

    if args.vocoder_path:
        vocoder_ref = Path(args.vocoder_path)
        vocoder_config_ref = args.vocoder_config_path or (vocoder_ref if vocoder_ref.is_dir() else vocoder_ref.parent)
        init_kwargs["vocoder_path"] = str(find_best_checkpoint(vocoder_ref))
        init_kwargs["vocoder_config_path"] = str(find_config(vocoder_config_ref))
    elif args.vocoder_name:
        init_kwargs["vocoder_name"] = args.vocoder_name

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    with contextlib.redirect_stdout(None):
        return TTS(**init_kwargs).to(device)


def synthesize_filelist(args):
    entries = read_filelist(args.filelist, limit=args.limit)
    out_dir = Path("generated") / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    tts = load_tts_model(args)
    print("TTS model loaded")

    try:
        import tqdm
    except ModuleNotFoundError as exc:
        raise SystemExit("Install tqdm first, for example: pip install tqdm") from exc

    for index in tqdm.trange(len(entries), unit=" files"):
        filename, text = entries[index]
        output_file = out_dir / f"{index + 1}.wav"

        if args.voice_convert:
            if not args.target:
                raise ValueError("--voice-convert requires --target")
            speaker_wav = Path(args.target) / filename
            with contextlib.redirect_stdout(None):
                tts.tts_with_vc_to_file(text=text, speaker_wav=str(speaker_wav), file_path=str(output_file))
        else:
            with contextlib.redirect_stdout(None):
                tts.tts_to_file(text=text, file_path=str(output_file))

    print("\n")


def add_generation_args(parser, default_model_name=None):
    parser.add_argument("--filelist", default=str(DEFAULT_FILELIST), help="filename|text test filelist")
    parser.add_argument("--target", "-t", help="watermarked victim wav directory used as speaker reference in --voice-convert mode")
    parser.add_argument("--output", "-o", required=True, help="generated/<output> directory name")
    parser.add_argument("--model-name", default=default_model_name, help="Coqui model name for pretrained inference")
    parser.add_argument("--model-path", help="fine-tuned acoustic checkpoint or run directory")
    parser.add_argument("--config-path", help="fine-tuned acoustic config.json or run directory")
    parser.add_argument("--vocoder-name", help="Coqui vocoder model name")
    parser.add_argument("--vocoder-path", help="fine-tuned vocoder checkpoint or run directory for adaptive*")
    parser.add_argument("--vocoder-config-path", help="fine-tuned vocoder config.json or run directory for adaptive*")
    parser.add_argument("--voice-convert", action="store_true", help="synthesize with this backbone, then convert to --target speaker wav")
    parser.add_argument("--limit", type=int, help="limit number of test utterances")
    parser.add_argument("--cpu", action="store_true", help="force CPU inference")
    return parser
