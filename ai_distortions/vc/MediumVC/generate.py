import yaml

import argparse
import logging
from pathlib import Path
import shutil

from Any2Any.infer.infer import Solver

logging.getLogger().disabled = True

parser = argparse.ArgumentParser()
parser.add_argument("--source", "-s", help="source wav path")  # Gives speech
parser.add_argument("--target", "-t", help="target wav path")  # Gives voice
parser.add_argument("--output", "-o", help="output wav path")
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parent

with (BASE_DIR.parent / "ljs_audio_text_test_filelist.txt").open() as f:
    filelist = f.read().strip().splitlines()

print(f"\nPreparing to generate...")

with (BASE_DIR / "Any2Any" / "infer" / "infer_config.yaml").open() as f:
    config = yaml.load(f, yaml.FullLoader)

config["test_filelist"] = [i.split("|")[0] for i in filelist]
config["source_path"] = str(Path(args.source))
config["target_path"] = str(Path(args.target))
config["out_dir"] = str(BASE_DIR / "Any2Any" / "output")

solver = Solver(config)
print("VC model loaded")

solver.infer()

OUT_DIR = BASE_DIR / "converted" / args.output
shutil.copytree(BASE_DIR / "Any2Any" / "output" / "infer" / "voice", OUT_DIR, dirs_exist_ok=True)
shutil.rmtree(BASE_DIR / "Any2Any" / "output")

print("\n")
