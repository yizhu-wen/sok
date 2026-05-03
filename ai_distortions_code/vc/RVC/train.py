import argparse
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune/train RVC on clean victim recordings.")
    parser.add_argument("--rvc-repo", default=os.environ.get("RVC_REPO"), help="path to RVC-Project/Retrieval-based-Voice-Conversion-WebUI")
    parser.add_argument("--clean-dir", required=True, help="directory containing clean victim wavs")
    parser.add_argument("--exp-name", required=True, help="RVC experiment name under <rvc-repo>/logs")
    parser.add_argument("--sample-rate", default="40k", choices=["32k", "40k", "48k"])
    parser.add_argument("--version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--if-f0", type=int, default=1, choices=[0, 1])
    parser.add_argument("--f0-method", default="rmvpe", choices=["pm", "harvest", "dio", "rmvpe"])
    parser.add_argument("--gpus", default="", help="RVC GPU list such as 0 or 0-1; empty uses CPU/MPS fallback")
    parser.add_argument("--n-procs", type=int, default=4)
    parser.add_argument("--save-epoch", type=int, default=5)
    parser.add_argument("--total-epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--pretrained-g", help="optional pretrained RVC generator path")
    parser.add_argument("--pretrained-d", help="optional pretrained RVC discriminator path")
    parser.add_argument("--cache-gpu", action="store_true")
    parser.add_argument("--save-latest", action="store_true")
    parser.add_argument("--save-every-weights", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    return parser.parse_args()


def require_rvc_repo(path):
    if not path:
        raise SystemExit("Pass --rvc-repo or set RVC_REPO to the RVC WebUI repository path.")
    repo = Path(path).resolve()
    expected = repo / "infer" / "modules" / "train" / "train.py"
    if not expected.exists():
        raise FileNotFoundError(f"Not an RVC WebUI checkout: missing {expected}")
    return repo


def run(command, cwd):
    print("\n$ " + " ".join(str(part) for part in command) + "\n")
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def pretrained_path(repo, version, if_f0, sample_rate, kind):
    folder = "pretrained_v2" if version == "v2" else "pretrained"
    f0_tag = "f0" if if_f0 else ""
    path = repo / "assets" / folder / f"{f0_tag}{kind}{sample_rate}.pth"
    return str(path) if path.exists() else ""


def write_config(repo, exp_dir, version, sample_rate):
    source = repo / "configs" / version / f"{sample_rate}.json"
    target = exp_dir / "config.json"
    if not source.exists():
        raise FileNotFoundError(f"Missing RVC config template: {source}")
    if not target.exists():
        shutil.copy2(source, target)


def build_filelist(repo, exp_dir, exp_name, sample_rate, version, if_f0, speaker_id):
    gt_wavs_dir = exp_dir / "0_gt_wavs"
    feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    names = {item.stem for item in gt_wavs_dir.glob("*.wav")} & {item.stem for item in feature_dir.glob("*.npy")}

    if if_f0:
        f0_dir = exp_dir / "2a_f0"
        f0nsf_dir = exp_dir / "2b-f0nsf"
        names = names & {item.name.replace(".wav.npy", "") for item in f0_dir.glob("*.wav.npy")}
        names = names & {item.name.replace(".wav.npy", "") for item in f0nsf_dir.glob("*.wav.npy")}

    rows = []
    for name in sorted(names):
        if if_f0:
            rows.append(
                f"{gt_wavs_dir / (name + '.wav')}|{feature_dir / (name + '.npy')}|"
                f"{f0_dir / (name + '.wav.npy')}|{f0nsf_dir / (name + '.wav.npy')}|{speaker_id}"
            )
        else:
            rows.append(f"{gt_wavs_dir / (name + '.wav')}|{feature_dir / (name + '.npy')}|{speaker_id}")

    if not rows:
        raise RuntimeError(f"No RVC training rows were generated under {exp_dir}")

    fea_dim = 256 if version == "v1" else 768
    mute_root = repo / "logs" / "mute"
    if if_f0:
        mute = (
            f"{mute_root / '0_gt_wavs' / ('mute' + sample_rate + '.wav')}|"
            f"{mute_root / ('3_feature' + str(fea_dim)) / 'mute.npy'}|"
            f"{mute_root / '2a_f0' / 'mute.wav.npy'}|"
            f"{mute_root / '2b-f0nsf' / 'mute.wav.npy'}|{speaker_id}"
        )
    else:
        mute = (
            f"{mute_root / '0_gt_wavs' / ('mute' + sample_rate + '.wav')}|"
            f"{mute_root / ('3_feature' + str(fea_dim)) / 'mute.npy'}|{speaker_id}"
        )
    rows.extend([mute, mute])
    random.shuffle(rows)

    filelist_path = exp_dir / "filelist.txt"
    filelist_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"wrote {filelist_path}")


def train_index(exp_dir, exp_name, version):
    import faiss

    feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    features = [np.load(path) for path in sorted(feature_dir.glob("*.npy"))]
    if not features:
        raise RuntimeError(f"No RVC features found under {feature_dir}")

    big_npy = np.concatenate(features, 0)
    order = np.arange(big_npy.shape[0])
    np.random.shuffle(order)
    big_npy = big_npy[order]

    if big_npy.shape[0] > 2e5:
        from sklearn.cluster import MiniBatchKMeans

        big_npy = MiniBatchKMeans(
            n_clusters=10000,
            verbose=True,
            batch_size=256,
            compute_labels=False,
            init="random",
        ).fit(big_npy).cluster_centers_

    np.save(exp_dir / "total_fea.npy", big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), max(big_npy.shape[0] // 39, 1))
    dim = 256 if version == "v1" else 768
    index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")
    faiss.extract_index_ivf(index).nprobe = 1
    index.train(big_npy)
    trained = exp_dir / f"trained_IVF{n_ivf}_Flat_nprobe_1_{exp_name}_{version}.index"
    added = exp_dir / f"added_IVF{n_ivf}_Flat_nprobe_1_{exp_name}_{version}.index"
    faiss.write_index(index, str(trained))
    for start in range(0, big_npy.shape[0], 8192):
        index.add(big_npy[start : start + 8192])
    faiss.write_index(index, str(added))
    print(f"wrote {added}")


def main():
    args = parse_args()
    repo = require_rvc_repo(args.rvc_repo)
    clean_dir = Path(args.clean_dir).resolve()
    if not clean_dir.exists():
        raise FileNotFoundError(f"Clean directory does not exist: {clean_dir}")

    exp_dir = repo / "logs" / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "infer/modules/train/preprocess.py",
            clean_dir,
            args.sample_rate.replace("k", "000"),
            args.n_procs,
            exp_dir,
            "False",
            "3.7",
        ],
        cwd=repo,
    )

    if args.if_f0:
        run(
            [
                sys.executable,
                "infer/modules/train/extract/extract_f0_print.py",
                exp_dir,
                args.n_procs,
                args.f0_method,
            ],
            cwd=repo,
        )

    gpus = [item for item in args.gpus.split("-") if item != ""]
    n_part = max(len(gpus), 1)
    for idx in range(n_part):
        command = [
            sys.executable,
            "infer/modules/train/extract_feature_print.py",
            "cuda" if gpus else "cpu",
            n_part,
            idx,
        ]
        if gpus:
            command.extend([gpus[idx], exp_dir, args.version, "False"])
        else:
            command.extend([exp_dir, args.version, "False"])
        run(command, cwd=repo)

    write_config(repo, exp_dir, args.version, args.sample_rate)
    build_filelist(repo, exp_dir, args.exp_name, args.sample_rate, args.version, args.if_f0, args.speaker_id)

    pretrained_g = args.pretrained_g or pretrained_path(repo, args.version, args.if_f0, args.sample_rate, "G")
    pretrained_d = args.pretrained_d or pretrained_path(repo, args.version, args.if_f0, args.sample_rate, "D")

    train_command = [
        sys.executable,
        "infer/modules/train/train.py",
        "-e",
        args.exp_name,
        "-sr",
        args.sample_rate,
        "-f0",
        args.if_f0,
        "-bs",
        args.batch_size,
        "-te",
        args.total_epoch,
        "-se",
        args.save_epoch,
        "-l",
        int(args.save_latest),
        "-c",
        int(args.cache_gpu),
        "-sw",
        int(args.save_every_weights),
        "-v",
        args.version,
    ]
    if args.gpus:
        train_command.extend(["-g", args.gpus])
    if pretrained_g:
        train_command.extend(["-pg", pretrained_g])
    if pretrained_d:
        train_command.extend(["-pd", pretrained_d])
    run(train_command, cwd=repo)

    if not args.skip_index:
        train_index(exp_dir, args.exp_name, args.version)


if __name__ == "__main__":
    main()
