import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare clean victim recordings for adaptive AdaIn-VC fine-tuning.")
    parser.add_argument("--clean-dir", required=True, help="directory containing clean victim wavs")
    parser.add_argument("--output-dir", required=True, help="output feature directory")
    parser.add_argument("--segment-size", type=int, default=128)
    parser.add_argument("--training-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    from preprocess.tacotron.utils import get_spectrograms

    random.seed(args.seed)

    clean_dir = Path(args.clean_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_paths = sorted(clean_dir.glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No .wav files found under {clean_dir}")

    raw_data = {}
    all_mels = []
    for index, wav_path in enumerate(wav_paths):
        if index % 50 == 0:
            print(f"processing {index}/{len(wav_paths)} files")
        mel, _ = get_spectrograms(str(wav_path))
        raw_data[wav_path.name] = mel
        all_mels.append(mel)

    stacked = np.concatenate(all_mels)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)
    std = np.maximum(std, 1e-8)

    attr = {"mean": mean, "std": std}
    with open(output_dir / "attr.pkl", "wb") as file:
        pickle.dump(attr, file)

    train_data = {}
    for key, value in raw_data.items():
        train_data[key] = (value - mean) / std

    train_path = output_dir / "train.pkl"
    with open(train_path, "wb") as file:
        pickle.dump(train_data, file)

    reduced = {key: value for key, value in train_data.items() if value.shape[0] > args.segment_size}
    reduced_path = output_dir / f"train_{args.segment_size}.pkl"
    with open(reduced_path, "wb") as file:
        pickle.dump(reduced, file)

    utt_list = sorted(reduced.keys())
    if not utt_list:
        raise ValueError(f"No utterances are longer than segment size {args.segment_size}")

    samples = []
    for utt_id in random.choices(utt_list, k=args.training_samples):
        start = random.randint(0, len(reduced[utt_id]) - args.segment_size)
        samples.append((utt_id, start))

    samples_path = output_dir / f"train_samples_{args.segment_size}.json"
    with open(samples_path, "w", encoding="utf-8") as file:
        json.dump(samples, file)

    print(f"wrote {train_path}")
    print(f"wrote {reduced_path}")
    print(f"wrote {samples_path}")
    print(f"wrote {output_dir / 'attr.pkl'}")


if __name__ == "__main__":
    main()
