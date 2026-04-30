import argparse
import contextlib
from pathlib import Path
import warnings

import torch
import tqdm
from TTS.api import TTS

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--target", "-t", help="target wav path")
parser.add_argument("--output", "-o", help="output wav path")
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parent

with (BASE_DIR / "ljs_audio_text_test_filelist.txt").open() as f:
    filelist = f.read().strip().splitlines()

print(f"\nGenerating synthesized audios...")

OUT_DIR = BASE_DIR / "generated" / args.output
OUT_DIR.mkdir(parents=True, exist_ok=True)

with contextlib.redirect_stdout(None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tts = TTS("tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)

print("TTS model loaded")

for i in tqdm.trange(len(filelist), unit=" files"):
    filename, text = filelist[i].split("|")

    target_file = str(Path(args.target) / filename)
    output_file = str(OUT_DIR / f"{i + 1}.wav")

    with contextlib.redirect_stdout(None):
        tts.tts_to_file(text, speaker_wav=target_file, language="en", file_path=output_file)

print("\n")
