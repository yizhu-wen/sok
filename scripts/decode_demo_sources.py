#!/usr/bin/env python3
"""Decode bit recovery accuracy for algorithm-specific reviewer demo WAVs.

Run this script once per algorithm with that algorithm's Python environment.
It updates demo_audio_clips/bit_accuracy.json using source paths from the
reviewer manifest. Digital-level rows are intentionally skipped because those
preview files are distortion examples and are not tied to one watermarking
algorithm.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENT_ROOT = Path("/home/yizhu/Desktop/Production/PycharmProjects/sok_experiment")
OUT_PATH = PROJECT_DIR / "demo_audio_clips" / "bit_accuracy.json"

LABEL_TO_ALGO = {
    "AudioSeal": "audioseal",
    "WavMark": "wavmark",
    "SilentCipher": "silentcipher",
    "Timbre": "timbre",
    "audiowmark": "audiowmark",
    "audiowavmark": "audiowmark",
    "FSVC": "fsvc",
    "Patchwork": "patchwork",
    "Norm-space": "normspace",
    "RobustDNN": "dnn_watermark",
}

ALGO_TO_LABELS = {}
for label, algo in LABEL_TO_ALGO.items():
    ALGO_TO_LABELS.setdefault(algo, set()).add(label)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32), int(sr)


def load_external_config(experiment_root: Path):
    import importlib.util

    cfg_path = experiment_root / "scripts" / "config.py"
    if not cfg_path.exists():
        cfg_path = PROJECT_DIR / "scripts" / "config.py"
    spec = importlib.util.spec_from_file_location("sok_external_config", cfg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config from {cfg_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_cuda_runtime(config, reexec: bool = False) -> None:
    base = getattr(config, "NVIDIA_BASE", None)
    if base is None or not Path(base).exists():
        return
    base = Path(base)
    lib_dirs = [str(p) for p in base.glob("*/lib") if p.is_dir()]
    nvrtc_lib = base / "cuda_nvrtc" / "lib" / "libnvrtc.so.12"
    compat_dir = Path("/tmp/silentcipher_cuda_compat")
    compat_dir.mkdir(parents=True, exist_ok=True)
    if nvrtc_lib.exists():
        link_path = compat_dir / "libnvrtc.so"
        if link_path.is_symlink() or link_path.exists():
            if link_path.resolve() != nvrtc_lib.resolve():
                link_path.unlink()
        if not link_path.exists():
            link_path.symlink_to(nvrtc_lib)
        ctypes.CDLL(str(link_path), mode=ctypes.RTLD_GLOBAL)
        lib_dirs.insert(0, str(compat_dir))
    current = os.environ.get("LD_LIBRARY_PATH", "")
    ld_path = ":".join(lib_dirs + ([current] if current else []))
    if reexec and os.environ.get("SOK_DEMO_CUDA_LIBS_SET") != "1":
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ld_path
        env["SOK_DEMO_CUDA_LIBS_SET"] = "1"
        os.execve(sys.executable, [sys.executable] + sys.argv, env)
    os.environ["LD_LIBRARY_PATH"] = ld_path


def resample(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return y.astype(np.float32)
    import librosa

    return librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr).astype(np.float32)


def build_audioseal_decoder():
    import librosa
    import torch
    from audioseal import AudioSeal

    msg = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0])
    sr_model = 16000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
    detector.eval()

    def decode(paths: list[Path]) -> list[float]:
        results = []
        batch_size = 16
        for offset in range(0, len(paths), batch_size):
            chunk = paths[offset:offset + batch_size]
            waves = []
            max_len = 0
            for path in chunk:
                y, sr = load_audio(path)
                y16 = librosa.resample(y, orig_sr=sr, target_sr=sr_model) if sr != sr_model else y
                waves.append(y16.astype(np.float32))
                max_len = max(max_len, len(y16))
            padded = [np.pad(y, (0, max_len - len(y))) if len(y) < max_len else y for y in waves]
            t = torch.from_numpy(np.stack(padded)).unsqueeze(1).to(device)
            with torch.no_grad():
                _, msg_out = detector.detect_watermark(t)
            bits = (msg_out.detach().cpu().numpy() > 0.5).astype(int)
            results.extend(float(np.mean(row == msg)) for row in bits)
        return results

    return decode


def build_wavmark_decoder():
    import torch
    import wavmark

    payload = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0], dtype=np.int32)
    sr_model = 16000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = wavmark.load_model().to(device)
    model.eval()

    def decode(paths: list[Path]) -> list[float]:
        results = []
        for path in paths:
            y, sr = load_audio(path)
            y16 = resample(y, sr, sr_model)
            decoded, _ = wavmark.decode_watermark(model, y16, show_progress=False)
            results.append(float("nan") if decoded is None else float(np.mean(decoded == payload)))
        return results

    return decode


def build_silentcipher_decoder(config):
    ensure_cuda_runtime(config, reexec=True)
    import silentcipher
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    message = np.array([123, 45, 67, 89, 12])
    model = silentcipher.get_model(model_type="44.1k", device=device)

    def decode(paths: list[Path]) -> list[float]:
        results = []
        for path in paths:
            y, sr = load_audio(path)
            try:
                result = model.decode_wav(y, sr, phase_shift_decoding=False)
                if result.get("status") and result.get("messages"):
                    results.append(float(np.mean(np.array(result["messages"][0]) == message)))
                else:
                    results.append(float("nan"))
            except Exception:
                results.append(float("nan"))
        return results

    return decode


def build_audiowmark_decoder(config):
    binary = Path(getattr(config, "AUDIOWMARK_BIN"))
    lib_dir = Path(getattr(config, "AUDIOWMARK_LIB"))
    payload_hex = "0102030405060708090a0b0c0d0e0f10"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join([str(lib_dir), env.get("LD_LIBRARY_PATH", "")]).strip(":")

    def decode(paths: list[Path]) -> list[float]:
        results = []
        for path in paths:
            result = subprocess.run(
                [str(binary), "get", str(path)],
                capture_output=True,
                text=True,
                env=env,
            )
            results.append(1.0 if payload_hex.lower() in result.stdout.lower() else 0.0)
        return results

    return decode


def build_kosta_decoder(algo: str, experiment_root: Path):
    import librosa
    import torch

    repo_dir = experiment_root / "repos" / "audio-watermarking"
    sys.path.insert(0, str(repo_dir))
    from fsvc_watermarking import fsvc_watermark_detection
    from norm_space_watermarking import norm_space_watermark_detection
    from patchwork_multylayer_watermarking import patchwork_multilayer_watermark_detection

    watermark = np.array(
        [
            1, 0, 1, 0, 1, 1, 0, 0, 1, 0,
            1, 1, 0, 1, 0, 0, 1, 0, 1, 1,
            0, 0, 1, 1, 0, 1, 0, 1, 1, 0,
            0, 1, 0, 1, 0, 1, 1, 0, 0, 1,
        ],
        dtype=int,
    )
    wm_length = 40
    sr_model = 16000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def decode(paths: list[Path]) -> list[float]:
        results = []
        for path in paths:
            y, sr = load_audio(path)
            y16 = librosa.resample(y, orig_sr=sr, target_sr=sr_model) if sr != sr_model else y
            if algo == "fsvc":
                bits = fsvc_watermark_detection(y16, wm_length, sr_model, device=device)
            elif algo == "patchwork":
                bits = patchwork_multilayer_watermark_detection(y16, wm_length, sr_model, device=device)
            elif algo == "normspace":
                bits = norm_space_watermark_detection(y16, wm_length, device=device)
            else:
                raise ValueError(algo)
            n = min(len(bits), wm_length)
            results.append(float(np.mean(np.asarray(bits[:n]) == watermark[:n])))
        return results

    return decode


def build_dnn_decoder(config, experiment_root: Path):
    ensure_cuda_runtime(config, reexec=True)
    ptxas_dir = getattr(config, "PTXAS_DIR", None)
    if ptxas_dir is not None:
        os.environ["PATH"] = f"{ptxas_dir}:{os.environ.get('PATH', '')}"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")

    import librosa
    import tensorflow as tf

    repo_dir = experiment_root / "repos" / "dnn-audio-watermarking"
    sr_model = 16000
    step_size = 33216
    hop_length = 511
    window_len = 1023
    decode_batch = 32

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    detector = tf.keras.models.load_model(str(repo_dir / "detector_model"))
    pool_path = repo_dir / "samples" / "message_pool.npy"
    if not pool_path.exists():
        pool_path = repo_dir / "dataset" / "message_pool.npy"
    message = (np.load(str(pool_path))[0].astype(np.float32) >= 0.5).astype(int)

    def stft_input(chunk: np.ndarray):
        chunk_tf = tf.constant(chunk.astype(np.float32))[tf.newaxis]
        stft = tf.transpose(
            tf.signal.stft(chunk_tf, window_len, hop_length, window_len),
            perm=[0, 2, 1],
        )
        return tf.stack([tf.math.real(stft), tf.math.imag(stft)], axis=-1)

    def decode(paths: list[Path]) -> list[float]:
        all_chunks = []
        chunk_counts = []
        for path in paths:
            y, sr = load_audio(path)
            y16 = librosa.resample(y, orig_sr=sr, target_sr=sr_model) if sr != sr_model else y
            pad_len = step_size - (len(y16) % step_size) if len(y16) % step_size else 0
            padded = np.pad(y16, (0, pad_len))
            for start in range(0, len(padded), step_size):
                all_chunks.append(padded[start:start + step_size].astype(np.float32))
            chunk_counts.append(len(padded) // step_size)

        out_parts = []
        for idx in range(0, len(all_chunks), decode_batch):
            batch_in = tf.concat([stft_input(c) for c in all_chunks[idx:idx + decode_batch]], axis=0)
            out_parts.append(detector(batch_in).numpy())
        out_all = np.concatenate(out_parts, axis=0)

        results = []
        offset = 0
        for count in chunk_counts:
            bits_for_audio = (out_all[offset:offset + count] >= 0.5).astype(int)
            bits = (np.mean(bits_for_audio, axis=0) >= 0.5).astype(int)
            results.append(float(np.mean(bits == message[:len(bits)])))
            offset += count
        return results

    return decode


def build_timbre_decoder(experiment_root: Path):
    import torch

    repo_dir = experiment_root / "repos" / "TimbreWatermarking" / "watermarking_model"
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))
    sys.path.insert(0, str(repo_dir))
    sys.path.insert(0, str(repo_dir / "distortions"))
    os.chdir(str(repo_dir))
    from timbre_eval_utils import build_timbre_embed_decode, load_timbre_model

    device = torch.device("cpu")
    _encoder, decoder, wm_bits, model_sr = load_timbre_model(experiment_root, device)
    _embed, decode_arrays = build_timbre_embed_decode(
        _encoder,
        decoder,
        wm_bits,
        model_sr,
        device,
        decode_batch=1,
        min_free_mb=0,
        poll_s=1,
        max_retry=1,
        force_cpu_decode=True,
    )

    def decode(paths: list[Path]) -> list[float]:
        waves = []
        sr_ref = None
        for path in paths:
            y, sr = load_audio(path)
            if sr_ref is None:
                sr_ref = sr
            if sr != sr_ref:
                y = resample(y, sr, sr_ref)
            waves.append(y)
        return decode_arrays(waves, sr_ref or model_sr)

    return decode


def build_decoder(algo: str, experiment_root: Path, config):
    if algo == "audioseal":
        return build_audioseal_decoder()
    if algo == "wavmark":
        return build_wavmark_decoder()
    if algo == "silentcipher":
        return build_silentcipher_decoder(config)
    if algo == "audiowmark":
        return build_audiowmark_decoder(config)
    if algo in {"fsvc", "patchwork", "normspace"}:
        return build_kosta_decoder(algo, experiment_root)
    if algo == "dnn_watermark":
        return build_dnn_decoder(config, experiment_root)
    if algo == "timbre":
        return build_timbre_decoder(experiment_root)
    raise ValueError(f"Unsupported algorithm: {algo}")


def read_existing() -> dict:
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            return json.load(f)
    return {"sources": {}}


def write_existing(data: dict) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(OUT_PATH)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", choices=sorted(ALGO_TO_LABELS))
    parser.add_argument("--manifest", default=PROJECT_DIR / "demo_audio_clips" / "manifest.json", type=Path)
    parser.add_argument("--experiment-root", default=DEFAULT_EXPERIMENT_ROOT, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    labels = ALGO_TO_LABELS[args.algorithm]
    paths = []
    for row in manifest:
        if row.get("category") == "Digital-level":
            continue
        if row.get("source_group") not in labels:
            continue
        path = PROJECT_DIR / row["source"]
        if path.exists():
            paths.append(path)

    data = read_existing()
    source_map = data.setdefault("sources", {})
    todo = [path for path in paths if args.force or path.relative_to(PROJECT_DIR).as_posix() not in source_map]
    if not todo:
        print(f"{args.algorithm}: all {len(paths)} source files already decoded")
        return 0

    config = load_external_config(args.experiment_root)
    decoder = build_decoder(args.algorithm, args.experiment_root, config)
    print(f"{args.algorithm}: decoding {len(todo)} source files", flush=True)
    scores = decoder(todo)

    for path, score in zip(todo, scores):
        rel = path.relative_to(PROJECT_DIR).as_posix()
        source_map[rel] = {
            "bit_accuracy": None if score != score else round(float(score), 6),
            "algorithm": args.algorithm,
        }
    write_existing(data)
    print(f"{args.algorithm}: wrote {OUT_PATH.relative_to(PROJECT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
