import tqdm
import yaml

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

VC_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VC_DIR))

from common.experiment import add_common_vc_args, existing_pairs  # noqa: E402


def resolve_path(path):
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute() or path.exists():
        return str(path)
    return str(SCRIPT_DIR / path)


parser = argparse.ArgumentParser()
add_common_vc_args(parser)
parser.add_argument("--attr", default="attr.pkl", help="normalization attr.pkl")
parser.add_argument("--config", default="config.yaml", help="AdaIn-VC config")
parser.add_argument("--model", default="vctk_model.ckpt", help="AdaIn-VC checkpoint; pass fine-tuned checkpoint for adaptive")
parser.add_argument("--sample-rate", type=int, default=24000)
args = parser.parse_args()

from inference import Inferencer  # noqa: E402

pairs = existing_pairs(args.source, args.target, args.filelist, args.limit)

print(f"\nGenerating converted audios...")

OUT_DIR = os.path.join("converted", args.output)
os.makedirs(OUT_DIR, exist_ok=True)

inf_args = argparse.Namespace(
    attr=resolve_path(args.attr),
    config=resolve_path(args.config),
    model=resolve_path(args.model),
    sample_rate=args.sample_rate,
)

with open(inf_args.config) as f:
    inf_config = yaml.load(f, yaml.FullLoader)

inferencer = Inferencer(config=inf_config, args=inf_args)
print("VC model loaded")

for i in tqdm.trange(len(pairs), unit="file"):
    filename, text, source_file, target_file = pairs[i]
    inferencer.args.source = str(source_file)
    inferencer.args.target = str(target_file)
    inferencer.args.output = os.path.join(OUT_DIR, f"{i + 1}.wav")

    inferencer.inference_from_path()

print("\n")
