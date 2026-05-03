import tqdm

import argparse
import os
import subprocess
import sys
from pathlib import Path


VC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VC_DIR))

from common.experiment import add_common_vc_args, existing_pairs  # noqa: E402


parser = argparse.ArgumentParser()
add_common_vc_args(parser)
parser.add_argument("--model-name", default="tts_models/multilingual/multi-dataset/your_tts")
parser.add_argument("--language-idx", default="en")
parser.add_argument("--use-cuda", default="True", choices=["True", "False"])
args = parser.parse_args()

pairs = existing_pairs(args.source, args.target, args.filelist, args.limit)

print(f"\nPreparing to generate...")

OUT_DIR = os.path.join("converted", args.output)
os.makedirs(OUT_DIR, exist_ok=True)

for i in tqdm.trange(len(pairs), unit="file"):
    filename, text, source_file, target_file = pairs[i]
    output_file = os.path.join(OUT_DIR, f"{i + 1}.wav")

    subprocess.run(
        [
            "tts",
            "--model_name",
            args.model_name,
            "--language_idx",
            args.language_idx,
            "--reference_wav",
            str(source_file),
            "--speaker_wav",
            str(target_file),
            "--out_path",
            output_file,
            "--use_cuda",
            args.use_cuda,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        check=True,
    )

print("\n")
