import torch
import tqdm
from TTS.api import TTS

import argparse
import contextlib
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--target", "-t", help="target wav path")
parser.add_argument("--output", "-o", help="output wav path")
args = parser.parse_args()

with open("ljs_audio_text_test_filelist.txt") as f:
    filelist = f.read().strip().splitlines()

print(f"\nGenerating synthesized audios...")

OUT_DIR = os.path.join("generated", args.output)
os.makedirs(OUT_DIR, exist_ok=True)

with contextlib.redirect_stdout(None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tts = TTS("tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)

print("TTS model loaded")

for i in tqdm.trange(len(filelist), unit=" files"):
    filename, text = filelist[i].split("|")

    target_file = os.path.join(args.target, filename)
    output_file = os.path.join(OUT_DIR, f"{i + 1}.wav")

    with contextlib.redirect_stdout(None):
        tts.tts_to_file(text, speaker_wav=target_file, language="en", file_path=output_file)

print("\n")
