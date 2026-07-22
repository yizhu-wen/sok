# Is Audio Watermarking Robust to Removal Attacks? A Comprehensive Measurement Study

This repository contains the experiment code and reviewer audio demo for the
measurement study:

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

The paper evaluation covers:

- 5 speech/music datasets: LJSpeech, LibriSpeech, DAPS, M4Singer, and MoisesDB
- 3 attack families: digital-level, physical-level, and AI-induced distortions
- 127 attack settings in the current evaluation section

The main takeaway is that no evaluated method is robust to every tested
quality-preserving removal attack. Pitch shift, physical re-recording, and
AI-induced voice conversion or TTS are the major failure modes.


## Interactive Demo Setup

Browser-based audio watermark demo for evaluation: upload an audio file, 
embed watermarks using the 10 benchmark methods, apply digital-level 
distortions, and review bit recovery rate and ViSQOL for each distorted clip.

### Running

Run the following script to install Docker if needed and builds a self-contained image, and
serves the demo with Gradio. See [`demo/README.md`](demo/README.md) for details, options, and scope.

```bash
./install.sh
```

## Datasets

The benchmark uses five public speech/music datasets, sampled as in the paper
(fixed seed 42):

| Dataset | Domain | Clips used | Source |
| --- | --- | --- | --- |
| LJSpeech | speech | 2,000 sampled of 13,100 | <https://keithito.com/LJ-Speech-Dataset/> |
| LibriSpeech | speech | 200 sampled | <https://www.openslr.org/12> |
| DAPS | speech | full dataset | <https://zenodo.org/records/4660670> |
| M4Singer | music (singing) | 2,000 sampled | <https://github.com/M4Singer/M4Singer> |
| MoisesDB | music (multitrack) | 2,000 sampled | <https://github.com/moises-ai/moises-db> |

Download the originals from the sources above, then reproduce the paper's
sampling into the directory layout the benchmark expects:

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

Any subset of `--raw` entries works — missing datasets are simply skipped by
the benchmark. The script writes `sample_manifest.json` recording the exact
files selected, and `--zip` produces a single archive of the sampled tree.

Pre-sampled bundle: **[Google Drive link — TODO: upload
`sok_dataset_sample.zip` and paste the share link here]**. Note that LJSpeech
(public domain), LibriSpeech (CC BY 4.0), and DAPS are redistributable;
M4Singer and MoisesDB have research-use licenses — check them before sharing
those two publicly, or share the manifest instead.

The background-noise and reverberation distortions additionally need the
[DEMAND](https://zenodo.org/records/1227121) noise corpus (`SOK_NOISE_DIR`)
and the [Aachen Impulse Response](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/)
database (`SOK_RIR_DIR`).

## Running Benchmark

One command runs the complete robustness benchmark — all 10 watermarking
methods × all automated attack settings × every dataset present under
`SOK_DATASET_DIR`:

```bash
./run_benchmarks.sh                        # everything
./run_benchmarks.sh --methods audioseal,kosta   # a subset
```

For each method and dataset, the runner embeds the watermark into every clip,
applies the full distortion suite to the watermarked audio — 116 automated
attack settings across 12 distortion types (pitch shift, time stretch,
Gaussian noise, bitcrush, MP3, background noise, cutting, high/low-pass,
sample suppression, resampling, reverberation) — decodes the watermark from
each distorted clip, and measures bit accuracy plus SI-SNR, PESQ, ESTOI,
ViSQOL, and SECS against the clean original. The physical re-recording and
AI-induced attacks (TTS / voice conversion, under `ai_distortions_code/`)
that complete the paper's 127 settings require hardware or separate model
pipelines and are not part of this script.

Prerequisites: the per-method virtualenvs under `envs/` (one per
`requirements/*.txt`; the methods' dependency stacks are mutually
incompatible), a CUDA GPU for the AI-based methods, `ffmpeg`, and
`SOK_AUDIOWMARK_BIN` pointing at an `audiowmark` binary. The script runs a
preflight check first and skips methods whose environment is missing.

Results are written to `results/benchmark/{dataset_key}/{algorithm}.json`
with logs under `results/logs/`. Runs are resume-safe: re-running the script
skips completed work. Build per-dataset Excel workbooks afterwards with:

```bash
python3 scripts/18_dataset_full_excel.py \
  --dataset speech_ljspeech \
  --out results/speech_ljspeech_full.xlsx \
  --suffixes __plain__
```