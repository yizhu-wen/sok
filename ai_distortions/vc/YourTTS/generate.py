import torch
import tqdm

import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--source", "-s", help="source wav path")  # Gives speech
parser.add_argument("--target", "-t", help="target wav path")  # Gives voice
parser.add_argument("--output", "-o", help="output wav path")
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parent

with (BASE_DIR.parent / "ljs_audio_text_test_filelist.txt").open() as f:
    filelist = f.read().strip().splitlines()

print(f"\nPreparing to generate...")

OUT_DIR = BASE_DIR / "converted" / args.output
OUT_DIR.mkdir(parents=True, exist_ok=True)
use_cuda = "True" if torch.cuda.is_available() else "False"

for i in tqdm.trange(len(filelist), unit="file"):
    filename, text = filelist[i].split("|")

    source_file = str(Path(args.source) / filename)
    target_file = str(Path(args.target) / filename)
    output_file = str(OUT_DIR / f"{i + 1}.wav")

    subprocess.run(
        ["tts", "--model_name", "tts_models/multilingual/multi-dataset/your_tts", "--language_idx", "en",
            "--reference_wav", source_file, "--speaker_wav", target_file, "--out_path", output_file, "--use_cuda", use_cuda],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )

print("\n")
