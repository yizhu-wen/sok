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

## Running Benchmark