import yaml

import argparse
import datetime
from pathlib import Path
import random
import subprocess
import sys

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

all_targets = [str(p) for p in Path(args.target).iterdir() if p.suffix == ".wav"]

batch_size = 50
filelist_batches = [filelist[i:i + batch_size] for i in range(0, len(filelist), batch_size)]
file_num = 0
start_time = datetime.datetime.now()

for batch_num, batch in enumerate(filelist_batches, 1):
    pairs_info = {}
    for entry in batch:
        filename, text = entry.split("|")

        file_num += 1
        pairs_info[file_num] = {
            "source": str(Path(args.source) / filename),
            "target": random.sample(all_targets, 10)
        }

    pairs_info_path = BASE_DIR / "pairs_info.yaml"
    with pairs_info_path.open("w") as f:
        yaml.dump(pairs_info, f)

    print(f"\nBATCH {batch_num}/{len(filelist_batches)}\n")

    batch_start_time = datetime.datetime.now()
    subprocess.run(
        [sys.executable, str(BASE_DIR / "convert_batch.py"), str(pairs_info_path), str(OUT_DIR)],
        cwd=str(BASE_DIR),
        check=True,
    )
    batch_end_time = datetime.datetime.now()

    print(f"\nElapsed time: {batch_end_time - batch_start_time}")

try:
    pairs_info_path.unlink()
except FileNotFoundError:
    pass

end_time = datetime.datetime.now()
print(f"\nFinished\nTotal elapsed time: {end_time - start_time}\n")
