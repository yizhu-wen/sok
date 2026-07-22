# Is Audio Watermarking Robust to Removal Attacks? A Comprehensive Measurement Study

This repository contains the experiment code and reviewer audio demo for the
measurement study:

> **Is Audio Watermarking Robust to Removal Attacks? A Comprehensive Measurement
> Study**

For a audio demo overview, open the
[project page](https://anonymous.4open.science/w/sok-6DB0/) (see
[Demo Media](#demo-media) below).

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


## Interactive Demo (Docker, one command)

For an evaluation-friendly, browser-based demo — upload an audio file, embed
watermarks with any subset of all 10 benchmark methods (editable payload bits
with per-method length constraints), apply digital-level distortions, and
inspect bit recovery rate, SI-SNR, ViSQOL, and the decoded bits per distorted
clip:

```bash
./install.sh
```

The script installs Docker if needed, builds a self-contained image, and
serves the demo. See
[`demo/README.md`](demo/README.md) for details, options, and scope.


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

The upstream code and pretrained checkpoints for Timbre, AWARE, and RobustDNN
are vendored under [`repos/`](repos/README.md); no external cloning is needed.

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

