#!/usr/bin/env python3
"""Build short reviewer audio previews for the GitHub Pages demo.

The page uses one representative clip for each setting-level condition, not
every source WAV. The manifest is grouped into digital-level, physical-level,
and AI-induced distortions.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
DIGITAL_ROOT = PROJECT_DIR / "distortion_demo_audio"
PHYSICAL_ROOT = PROJECT_DIR / "physical"
AI_ROOT = PROJECT_DIR / "ai_distorted"
OUT_DIR = PROJECT_DIR / "demo_audio_clips"
CLIP_SECONDS = 10.0
SEED = 2026
MEDIA_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a"}

CATEGORY_ORDER = {
    "Digital-level": 0,
    "Physical-level": 1,
    "AI-induced": 2,
}

VARIANT_LABELS = {
    "gl": "Griffin-Lim",
    "u_hifi": "Universal HiFi-GAN",
    "fine_hifi": "Fine-tuned HiFi-GAN",
    "add_wm": "Watermarked input",
    "rm_wm": "Watermark-removal output",
    "default": "Default",
}

MODEL_LABELS = {
    "Fastspeech2": "FastSpeech2",
    "Tacotron2": "Tacotron2",
    "YourTTS": "YourTTS",
    "YourTTS_VC": "YourTTS VC",
}


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def duration_seconds(path: Path) -> float:
    raw = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return float(raw)


def media_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES
    )


def slugify(value: str) -> str:
    value = value.replace("LJSpeech-1.1", "ljspeech")
    value = re.sub(r"[^A-Za-z0-9]+", "-", value.lower()).strip("-")
    return value or "item"


def setting_label(raw: str) -> str:
    if raw.startswith("plus"):
        return "+" + raw[4:].replace("p", ".")
    if raw.startswith("neg"):
        return "-" + raw[3:].replace("p", ".")
    return raw.replace("_", " ")


def title_label(raw: str) -> str:
    return raw.replace("_", " ").replace("-", " ").title()


def model_label(raw: str) -> str:
    return MODEL_LABELS.get(raw, raw.replace("_", " "))


def variant_label(raw: str) -> str:
    return VARIANT_LABELS.get(raw, raw.replace("_", " "))


def method_label(raw: str) -> str:
    normalized = {
        "fvsc": "FSVC",
        "fsvc": "FSVC",
        "norm_space": "Norm-space",
        "normspace": "Norm-space",
        "dnn_watermark": "RobustDNN",
        "robustdnn": "RobustDNN",
        "audioseal": "AudioSeal",
        "wavmark": "WavMark",
        "silentcipher": "SilentCipher",
        "timbre": "Timbre",
        "audiowmark": "audiowmark",
        "patchwork": "Patchwork",
    }
    return normalized.get(raw, raw)


def collect_digital() -> dict[tuple[str, str, str], list[Path]]:
    groups: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    for path in media_files(DIGITAL_ROOT):
        rel = path.relative_to(DIGITAL_ROOT)
        if len(rel.parts) < 5:
            continue
        domain_attack, attack, setting, dataset = rel.parts[:4]
        domain = domain_attack.split("_", 1)[0]
        detail = f"{domain}, {dataset}, setting {setting_label(setting)}"
        groups[("Digital-level", title_label(attack), detail)].append(path)
    return groups


def collect_physical() -> dict[tuple[str, str, str], list[Path]]:
    groups: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    for path in media_files(PHYSICAL_ROOT):
        rel = path.relative_to(PHYSICAL_ROOT)
        if len(rel.parts) < 4:
            continue
        if rel.parts[0] == "clean_watermarked":
            continue
        if rel.parts[0] == "distance":
            setting = setting_label(rel.parts[1])
            method = method_label(rel.parts[2])
            key = ("Physical-level", "Re-recording distance", setting)
            groups[(key[0], key[1], f"{key[2]}, {method}")].append(path)
        elif rel.parts[0] == "device":
            setting = title_label(rel.parts[1].replace("_tests", ""))
            method = method_label(rel.parts[2])
            key = ("Physical-level", "Device variation", setting)
            groups[(key[0], key[1], f"{key[2]}, {method}")].append(path)
        else:
            continue
    return groups


def collect_ai() -> dict[tuple[str, str, str], list[Path]]:
    groups: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    for path in media_files(AI_ROOT):
        rel = path.relative_to(AI_ROOT)
        if len(rel.parts) < 4:
            continue
        if rel.parts[0] == "tts":
            model = model_label(rel.parts[1])
            method = method_label(rel.parts[2])
            variant = variant_label(rel.parts[3] if len(rel.parts) >= 5 else "default")
            groups[("AI-induced", "TTS regeneration", f"{model}, {method}, {variant}")].append(path)
        elif rel.parts[0] == "vc":
            if len(rel.parts) < 5:
                continue
            model = model_label(rel.parts[1])
            method = method_label(rel.parts[2])
            variant = variant_label(rel.parts[3])
            groups[("AI-induced", "Voice conversion", f"{model}, {method}, {variant}")].append(path)
    return groups


def selected_metadata(category: str, path: Path) -> tuple[str, str]:
    if category == "Digital-level":
        rel = path.relative_to(DIGITAL_ROOT)
        _domain_attack, _attack, _setting, dataset = rel.parts[:4]
        return dataset, rel.name

    if category == "Physical-level":
        rel = path.relative_to(PHYSICAL_ROOT)
        if rel.parts[0] in {"distance", "device"} and len(rel.parts) >= 4:
            return method_label(rel.parts[2]), rel.name
        return "physical", rel.name

    rel = path.relative_to(AI_ROOT)
    if rel.parts[0] in {"tts", "vc"} and len(rel.parts) >= 4:
        return method_label(rel.parts[2]), rel.name
    return "AI", rel.name


def build_clip(src: Path, dst: Path, start: float, length: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{length:.3f}",
        "-i",
        str(src),
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "160k",
        str(dst),
    ]
    subprocess.check_call(cmd)


def clean_output() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUT_DIR.glob("*.mp3"):
        path.unlink()
    manifest_path = OUT_DIR / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()


def sorted_groups() -> list[tuple[tuple[str, str, str], list[Path]]]:
    groups = {}
    for collector in (collect_digital, collect_physical, collect_ai):
        groups.update(collector())
    return sorted(
        groups.items(),
        key=lambda item: (
            CATEGORY_ORDER.get(item[0][0], 99),
            item[0][1],
            item[0][2],
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="do not remove existing MP3 previews before rebuilding",
    )
    args = parser.parse_args()

    if not args.keep_existing:
        clean_output()
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    manifest = []

    for idx, (key, files) in enumerate(sorted_groups(), start=1):
        category, distortion, setting = key
        src = rng.choice(files)
        source_group, selected_file = selected_metadata(category, src)
        duration = duration_seconds(src)
        clip_len = min(CLIP_SECONDS, max(duration, 0.001))
        start = 0.0
        if duration > CLIP_SECONDS:
            start = rng.uniform(0, max(0.0, duration - CLIP_SECONDS))

        filename = (
            f"{idx:03d}-{slugify(category)}-{slugify(distortion)}-"
            f"{slugify(setting)}.mp3"
        )
        dst = OUT_DIR / filename
        build_clip(src, dst, start, clip_len)

        manifest.append(
            {
                "category": category,
                "distortion": distortion,
                "setting": setting,
                "source_group": source_group,
                "selected_file": selected_file,
                "clip": dst.relative_to(PROJECT_DIR).as_posix(),
                "source": src.relative_to(PROJECT_DIR).as_posix(),
                "start_seconds": round(start, 2),
                "duration_seconds": round(clip_len, 2),
            }
        )

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(manifest)} clips and {manifest_path.relative_to(PROJECT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
