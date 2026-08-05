# Is Audio Watermarking Robust to Removal Attacks? A Comprehensive Measurement Study

This repository contains the experiment code and audio demo for the measurement
study:

> **Is Audio Watermarking Robust to Removal Attacks? A Comprehensive Measurement
> Study**

For an audio demo overview, open the
[project page](https://anonymous.4open.science/w/sok-6DB0/).

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

The evaluation covers 5 speech/music datasets, 3 attack families
(digital-level, physical-level, AI-induced), and 127 attack settings. No
evaluated method is robust to every tested quality-preserving removal attack;
pitch shift, physical re-recording, and AI-induced voice conversion or TTS are
the major failure modes.

## Interactive Watermarking Demo (Docker + Gradio)

An evaluation-friendly, one-command package of the benchmark: upload a single
audio file, embed watermarks with one or more methods, apply digital-level
distortions, and inspect robustness metrics — all from a browser UI.

### Quick start

```bash
./install.sh
```

The script:

1. installs Docker if it is missing (Linux; on macOS it points you to Docker
   Desktop),
2. builds the demo image (all Python and system dependencies across three
   isolated environments, the `audiowmark` CLI compiled from source, and
   pre-downloaded model checkpoints for AudioSeal / WavMark / SilentCipher;
   Timbre / RobustDNN checkpoints are vendored under `repos/`),
3. starts the demo at <http://localhost:7860>.

## Datasets

Five public speech/music datasets, randomly sampled with a fixed seed (42):

| Dataset | Domain | Clips used | Source |
| --- | --- | --- | --- |
| LJSpeech | speech | 2,000 of 13,100 | <https://keithito.com/LJ-Speech-Dataset/> |
| LibriSpeech | speech | 2,000 sampled | <https://www.openslr.org/12> |
| DAPS | speech | full dataset (1,500) | <https://zenodo.org/records/4660670> |
| M4Singer | music (singing) | 2,000 sampled | <https://github.com/M4Singer/M4Singer> |
| MoisesDB | music (multitrack) | 2,000 sampled | <https://github.com/moises-ai/moises-db> |

Download the originals, then reproduce the sampling into the layout the
benchmark expects:

```bash
python3 scripts/16_sample_dataset.py \
  --raw ljspeech=/data/LJSpeech-1.1 \
  --raw librispeech=/data/LibriSpeech/test-clean \
  --raw daps=/data/daps \
  --raw m4singer=/data/m4singer \
  --raw moisesdb=/data/moisesdb \
  --out /data/sok_dataset \
  --zip sok_dataset_sample.zip          # optional shareable bundle

export SOK_DATASET_DIR=/data/sok_dataset
```

Any subset of `--raw` entries works — missing datasets are skipped by the
benchmark. The script writes `sample_manifest.json` recording the exact files
selected.

Pre-sampled bundle: **[Google Drive link — TODO: upload
`sok_dataset_sample.zip` and paste the share link here]**. LJSpeech, LibriSpeech,
and DAPS are redistributable; M4Singer and MoisesDB have research-use licenses
— check them before sharing publicly.

Background-noise and reverberation distortions additionally need the
[DEMAND](https://zenodo.org/records/1227121) noise corpus (`SOK_NOISE_DIR`) and
the [Aachen Impulse Response](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/)
database (`SOK_RIR_DIR`).

## Running Benchmark

One command runs all 10 methods × all automated attack settings × every dataset
present under `SOK_DATASET_DIR`:

```bash
./run_benchmarks.sh                              # everything
./run_benchmarks.sh --methods audioseal,kosta    # a subset
```

For each method and dataset, the runner embeds the watermark, applies 116
automated attack settings across 12 distortion types (pitch shift, time stretch,
Gaussian noise, bitcrush, MP3, background noise, cutting, high/low-pass, sample
suppression, resampling, reverberation), decodes the watermark from each
distorted clip, and measures bit accuracy plus SI-SNR, PESQ, ESTOI, ViSQOL, and
SECS against the clean original. The physical re-recording and AI-induced
attacks (under `ai_distortions_code/`) that complete the 127 settings need
hardware or separate model pipelines and are not part of this script.

Prerequisites: per-method virtualenvs under `envs/` (one per
`requirements/*.txt`), a CUDA GPU for the AI-based methods, `ffmpeg`, and
`SOK_AUDIOWMARK_BIN`. The script runs a preflight check and skips methods whose
environment is missing.

Results go to `results/benchmark/{dataset_key}/{algorithm}.json` with logs under
`results/logs/`; re-running skips completed work. Build per-dataset workbooks
with:

```bash
python3 scripts/18_dataset_full_excel.py \
  --dataset speech_ljspeech \
  --out results/speech_ljspeech_full.xlsx \
  --suffixes __plain__
```
