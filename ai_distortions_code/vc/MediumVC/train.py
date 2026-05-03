import argparse
import sys
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from Any2Any.solver import Solver  # noqa: E402


def resolve_path(path):
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute() or path.exists():
        return str(path)
    return str(SCRIPT_DIR / path)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune MediumVC on clean victim recordings.")
    parser.add_argument("--config", default="Any2Any/config.yaml")
    parser.add_argument("--feature-pkl", default="Any2Any/pre_feature/spk_emb_mel_label_adaptive.pkl")
    parser.add_argument("--resume-path", default="Any2Any/model/checkpoint-3900.pt", help="checkpoint to adapt from")
    parser.add_argument("--out-dir", default="Any2Any/output_adaptive")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(resolve_path(args.config)) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config["figure_mel_label_dir"] = resolve_path(args.feature_pkl)
    config["resume"] = True
    config["resume_path"] = resolve_path(args.resume_path)
    config["out_dir"] = resolve_path(args.out_dir)

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate

    solver = Solver(config)
    solver.train()


if __name__ == "__main__":
    main()
