import yaml

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path


VC_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(VC_DIR))

from common.experiment import add_common_vc_args, existing_pairs  # noqa: E402
from Any2Any.infer.infer import Solver  # noqa: E402


def resolve_path(path):
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute() or path.exists():
        return str(path)
    return str(SCRIPT_DIR / path)


logging.getLogger().disabled = True

parser = argparse.ArgumentParser()
add_common_vc_args(parser)
parser.add_argument("--config", default="Any2Any/infer/infer_config.yaml")
parser.add_argument("--checkpoint", default="Any2Any/model/checkpoint-3900.pt", help="MediumVC checkpoint; pass fine-tuned checkpoint for adaptive")
parser.add_argument("--singlevc-checkpoint", help="optional any2one pretrain checkpoint override")
parser.add_argument("--hifi-model-path", help="optional HiFi-GAN checkpoint override")
parser.add_argument("--hifi-config-path", help="optional HiFi-GAN config override")
parser.add_argument("--scratch-out-dir", default="Any2Any/output", help="temporary MediumVC output directory")
args = parser.parse_args()

pairs = existing_pairs(args.source, args.target, args.filelist, args.limit)

print(f"\nPreparing to generate...")

with open(resolve_path(args.config)) as f:
    config = yaml.load(f, yaml.FullLoader)

config["test_filelist"] = [filename for filename, _, _, _ in pairs]
config["source_path"] = str(Path(args.source).resolve())
config["target_path"] = str(Path(args.target).resolve())
config["resume_path"] = resolve_path(args.checkpoint)
config["out_dir"] = resolve_path(args.scratch_out_dir)

if args.singlevc_checkpoint:
    config["singlevc_model_path"] = resolve_path(args.singlevc_checkpoint)
if args.hifi_model_path:
    config["hifi_model_path"] = resolve_path(args.hifi_model_path)
if args.hifi_config_path:
    config["hifi_config_path"] = resolve_path(args.hifi_config_path)

solver = Solver(config)
print("VC model loaded")

solver.infer()

OUT_DIR = Path("converted") / args.output
source_out_dir = Path(config["out_dir"]) / "infer" / "voice"
shutil.copytree(source_out_dir, OUT_DIR, dirs_exist_ok=True)
shutil.rmtree(config["out_dir"], ignore_errors=True)

print("\n")
