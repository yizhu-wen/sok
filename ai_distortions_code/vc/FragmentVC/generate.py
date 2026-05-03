import yaml

import argparse
import datetime
import os
import random
import subprocess
import sys
from pathlib import Path


VC_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VC_DIR))

from common.experiment import DEFAULT_FILELIST, read_filelist, wav_files  # noqa: E402


def resolve_path(path):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return str(path)
    return str(SCRIPT_DIR / path)


parser = argparse.ArgumentParser()
parser.add_argument("--source", "-s", required=True, help="watermarked source wav directory")
parser.add_argument("--target", "-t", required=True, help="clean victim reference wav directory")
parser.add_argument("--output", "-o", required=True, help="converted/<output> directory name")
parser.add_argument("--filelist", default=str(DEFAULT_FILELIST), help="filename|text filelist used for conversion")
parser.add_argument("--limit", type=int, help="limit number of utterances")
parser.add_argument("--target-samples", type=int, default=10, help="clean reference utterances per conversion")
parser.add_argument("--batch-size", type=int, default=50)
parser.add_argument("--seed", type=int)
parser.add_argument("--ckpt-path", default="checkpoints/fragmentvc.pt", help="FragmentVC checkpoint; pass fine-tuned checkpoint for adaptive")
parser.add_argument("--wav2vec-path", default="checkpoints/wav2vec_small.pt")
parser.add_argument("--vocoder-path", default="checkpoints/vocoder.pt")
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)

filelist = read_filelist(args.filelist, args.limit)

print(f"\nPreparing to generate...")

OUT_DIR = Path("converted") / args.output
OUT_DIR.mkdir(parents=True, exist_ok=True)
output_dir_arg = str(OUT_DIR.resolve())

all_targets = wav_files(args.target)
if not all_targets:
    raise FileNotFoundError(f"No clean reference wav files found under {args.target}")
if args.target_samples < 1:
    raise ValueError("--target-samples must be >= 1")

filelist_batches = [filelist[i:i + args.batch_size] for i in range(0, len(filelist), args.batch_size)]
file_num = 0
start_time = datetime.datetime.now()
pairs_info_path = SCRIPT_DIR / "pairs_info.yaml"

for batch_num, batch in enumerate(filelist_batches, 1):
    pairs_info = {}
    for filename, text in batch:
        file_num += 1
        source_file = Path(args.source) / filename
        if not source_file.exists():
            raise FileNotFoundError(f"Missing source wav: {source_file}")

        target_count = min(args.target_samples, len(all_targets))
        pairs_info[file_num] = {
            "source": str(source_file),
            "target": random.sample(all_targets, target_count),
        }

    with open(pairs_info_path, "w") as f:
        yaml.dump(pairs_info, f)

    print(f"\nBATCH {batch_num}/{len(filelist_batches)}\n")

    batch_start_time = datetime.datetime.now()
    subprocess.run(
        [
            sys.executable,
            "convert_batch.py",
            str(pairs_info_path),
            output_dir_arg,
            "--ckpt_path",
            resolve_path(args.ckpt_path),
            "--wav2vec_path",
            resolve_path(args.wav2vec_path),
            "--vocoder_path",
            resolve_path(args.vocoder_path),
        ],
        cwd=SCRIPT_DIR,
        check=True,
    )
    batch_end_time = datetime.datetime.now()

    print(f"\nElapsed time: {batch_end_time - batch_start_time}")

try:
    os.remove(pairs_info_path)
except FileNotFoundError:
    pass

end_time = datetime.datetime.now()
print(f"\nFinished\nTotal elapsed time: {end_time - start_time}\n")
