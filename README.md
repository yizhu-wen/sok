# Is Audio Watermarking Robust to Removal Attacks? A Comprehensive Measurement Study

This repository contains the experiment code and reviewer audio demo for the
measurement study:

> **Is Audio Watermarking Robust to Removal Attacks? A Comprehensive Measurement
> Study**

The project evaluates whether current audio watermarking systems survive
quality-preserving removal attacks. It combines a component-wise survey of 26
watermarking schemes with a large-scale benchmark of 10 reproducible,
open-source methods across speech and music.

For a reviewer-friendly audio demo overview, open the project page:

- [Audio demo project overview](https://anonymous.4open.science/w/sok-6DB0/)

The page is backed by `index.html` and the clipped audio files under
`demo_audio_clips/`.

## What This Repo Contains

- `scripts/`: benchmark runners, timing scripts, reporting utilities, and
  preflight checks
- `ai_distortions_code/`: TTS and voice-conversion distortion pipelines used for
  AI-induced removal attacks
- `demo_audio_clips/`: 10-second-or-shorter reviewer previews generated from
  the demo audio
- `requirements/`: per-method dependency manifests
- `repos/README.md`: expected layout for third-party method repositories
- top-level `*_watermarking_gpu.py` files: classic watermarking baselines used
  by the Kosta-method wrappers
- `index.html`: static GitHub Pages overview for reviewers

The public package intentionally excludes generated benchmark outputs, local
virtual environments, private logs, and machine-specific paths.

## Paper Snapshot

The benchmark reproduces 10 audio watermarking methods:

| Method | Type | Runner |
| --- | --- | --- |
| AudioSeal | AI-based | `scripts/13_large_audioseal.py` |
| WavMark | AI-based | `scripts/13_large_wavmark.py` |
| SilentCipher | AI-based | `scripts/13_large_silentcipher.py` |
| Timbre | AI-based | `scripts/13_large_timbre.py` |
| RobustDNN / DNN-WM | AI-based | `scripts/13_large_dnn.py` |
| AWARE | AI-based | `scripts/13_large_aware.py` |
| audiowmark | Traditional / CLI | `scripts/13_large_audiowmark.py` |
| FSVC | Traditional | `scripts/13_large_kosta.py` |
| Patchwork | Traditional | `scripts/13_large_kosta.py` |
| Norm-space | Traditional | `scripts/13_large_kosta.py` |

The paper evaluation covers:

- 5 speech/music datasets: LJSpeech, LibriSpeech, DAPS, M4Singer, and MoisesDB
- 3 attack families: digital-level, physical-level, and AI-induced distortions
- 127 attack settings in the current evaluation section
- metrics for watermark recovery and perceptual quality, including bit accuracy,
  SI-SNR, PESQ, ESTOI, ViSQOL, SECS, and subjective MUSHRA scores

The main takeaway is that no evaluated method is robust to every tested
quality-preserving removal attack. Pitch shift, physical re-recording, and
AI-induced voice conversion or TTS are the major failure modes.

## Setup

This project uses separate Python environments because the evaluated methods
depend on incompatible Python, PyTorch, TensorFlow, and CUDA versions. Treat the
files under `requirements/` as reference manifests, then adjust CUDA wheels and
Python minor versions for your host as needed.

Example:

```bash
python -m venv envs/viz
envs/viz/bin/pip install -r requirements/viz.txt

python -m venv envs/audioseal
envs/audioseal/bin/pip install -r requirements/audioseal.txt
```

Install system tools separately:

- `ffmpeg`
- `audiowmark`, exposed through `SOK_AUDIOWMARK_BIN`
- CUDA libraries required by the specific GPU environments

Several methods also require external upstream repositories or checkpoints. See
[`repos/README.md`](repos/README.md) for the expected repository layout.

## Configuration

Set machine-local paths with environment variables or edit `scripts/config.py`.
The most important variables are:

- `SOK_STORAGE_DIR`
- `SOK_DATASET_DIR`
- `SOK_NOISE_DIR`
- `SOK_RIR_DIR`
- `SOK_AUDIOWMARK_BIN`
- `SOK_AUDIOWMARK_LIB`
- `SOK_NVIDIA_BASE`
- `SOK_PTXAS_DIR`

Run the preflight check before launching full benchmarks:

```bash
envs/viz/bin/python scripts/preflight.py
```

## Running Benchmarks

Each `scripts/13_large_*.py` file embeds a method-specific watermark, applies
the configured distortion suite, decodes the watermark, and writes aggregate
JSON results under `results/benchmark/{dataset_key}/{algorithm}.json`.

Examples:

```bash
envs/audioseal/bin/python scripts/13_large_audioseal.py
envs/timbre/bin/python scripts/13_large_timbre.py
envs/dnn_wm/bin/python scripts/13_large_dnn.py
envs/kosta/bin/python scripts/13_large_kosta.py
```

The shared benchmark utility supports checkpoint resume. If a run is interrupted,
rerun the same command and completed files will be skipped.

## Timing Experiments

Timing scripts are under `scripts/14_timing_*.py`. They measure method-level
embed/decode cost and write results under `results/timing/`.

```bash
envs/audioseal/bin/python scripts/14_timing_audioseal.py
envs/wavmark/bin/python scripts/14_timing_wavmark.py
python scripts/14_timing_report.py
```

## Reporting

Build a dataset workbook from completed benchmark JSON files:

```bash
python3 scripts/18_dataset_full_excel.py \
  --dataset speech_ljspeech \
  --out results/speech_ljspeech_full.xlsx \
  --suffixes __plain__
```

Generated outputs are intentionally not committed. See
[`results/README.md`](results/README.md) for the expected output structure.

## Demo Media

The reviewer audio demo page is available here:

- [Audio demo project overview](https://anonymous.4open.science/w/sok-6DB0/)

It includes:

- an audio-first table grouped into digital-level, physical-level, and
  AI-induced distortions
- 702 MP3 previews under `demo_audio_clips/`, each capped at 10 seconds
- one representative clip per setting-level distortion condition
- a manifest under `demo_audio_clips/manifest.json` used by `index.html`

Rebuild the clipped audio previews and manifest with:

```bash
python3 scripts/build_reviewer_audio_demo.py
```

## Reproducibility Notes

- Keep full benchmark results separate from reduced-sample validation runs.
- Use reduced sample counts only for bring-up or debugging, and label those
  outputs separately from full benchmark results.
- Do not commit local datasets, model checkpoints, logs, or generated outputs.
- If upstream method APIs change, prefer updating the corresponding wrapper
  script instead of changing shared benchmark semantics.
