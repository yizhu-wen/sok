import tqdm
import yaml

import argparse
from pathlib import Path
import warnings

from inference import Inferencer

warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--source", "-s", help="source wav path")  # Gives speech
parser.add_argument("--target", "-t", help="target wav path")  # Gives voice
parser.add_argument("--output", "-o", help="output wav path")
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parent

with (BASE_DIR.parent / "ljs_audio_text_test_filelist.txt").open() as f:
    filelist = f.read().strip().splitlines()

print(f"\nGenerating converted audios...")

OUT_DIR = BASE_DIR / "converted" / args.output
OUT_DIR.mkdir(parents=True, exist_ok=True)

inf_args = argparse.Namespace(
    attr="attr.pkl",
    config="config.yaml",
    model="vctk_model.ckpt",
    sample_rate=24000
)

with open(inf_args.config) as f:
    inf_config = yaml.load(f, yaml.FullLoader)

inferencer = Inferencer(config=inf_config, args=inf_args)
print("VC model loaded")

for i in tqdm.trange(len(filelist), unit="file"):
    filename, text = filelist[i].split("|")

    inferencer.args.source = str(Path(args.source) / filename)
    inferencer.args.target = str(Path(args.target) / filename)
    inferencer.args.output = str(OUT_DIR / f"{i + 1}.wav")

    inferencer.inference_from_path()

print("\n")
