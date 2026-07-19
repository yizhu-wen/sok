# Interactive Watermarking Demo (Docker + Gradio)

An evaluation-friendly, one-command package of the benchmark: upload a single
audio file, embed watermarks with one or more methods, apply digital-level
distortions, and inspect robustness metrics — all from a browser UI.

## Quick start

```bash
./install.sh
```

The script:

1. installs Docker if it is missing (Linux; on macOS it points you to Docker
   Desktop),
2. builds the demo image (all Python and system dependencies, the
   `audiowmark` CLI compiled from source, and pre-downloaded model
   checkpoints for AudioSeal / WavMark / SilentCipher),
3. starts the demo at **http://localhost:7860**.

Options: `--port N`, `--rebuild` (clean rebuild), `--build-only`.

Manual equivalent:

```bash
docker build -t sok-audio-watermark-demo .
docker run -d -p 7860:7860 --name sok-audio-watermark-demo sok-audio-watermark-demo
```

## What the demo does

1. **Input** — upload (or record) a single audio file. Audio longer than 20 s
   is clipped, matching the benchmark's clip rule.
2. **Watermark embedding** — select any subset of methods (or all). Each
   method has a default payload identical to the paper benchmark; you can
   edit the bits, and the bit-length constraint of each algorithm is
   enforced:

   | Method | Type | Payload bits |
   | --- | --- | --- |
   | AudioSeal | AI-based | exactly 16 |
   | WavMark | AI-based | exactly 16 |
   | SilentCipher | AI-based | exactly 40 (5 bytes) |
   | audiowmark | Traditional (CLI) | exactly 128 (16 bytes) |
   | FSVC | Traditional | 8–64 (default 40) |
   | Patchwork | Traditional | 8–64 (default 40) |
   | Norm-space | Traditional | 8–64 (default 40) |

   Each watermarked output is playable in the browser next to its
   spectrogram, with the SI-SNR against the original and a sanity decode of
   the clean watermarked audio.
3. **Digital distortions** — select any subset of the digital-level
   distortion family from the benchmark (pitch shift, time stretch, Gaussian
   noise, bitcrush, MP3 compression, cutting, high/low-pass filtering,
   sample suppression, resampling). Every selected distortion applies **all**
   of its benchmark settings by default; individual settings can be narrowed
   in the *Distortion settings* accordion. The distortion implementations and
   setting grids are imported directly from `scripts/benchmark_utils.py`.
4. **Evaluation** — for every distorted watermarked clip the app reports:
   - **Bit recovery rate** — fraction of payload bits recovered,
   - **SI-SNR (dB)** — vs. the clean original,
   - **ViSQOL** — MOS-LQO vs. the clean original (speech 16 kHz or
     audio 48 kHz mode),
   - **Decoded bits** — the raw bit string the decoder returned. Methods
     that refuse to decode (WavMark, SilentCipher, audiowmark) show `None`,
     and the refusal counts as bit recovery 0.0, exactly as in the benchmark.

   Any distorted clip can be selected for playback with its spectrogram.

## Scope and limitations

- **Methods included:** 7 of the 10 benchmark methods. **Timbre, AWARE, and
  DNN-WM are excluded** — they require external upstream repositories and
  checkpoints (`repos/TimbreWatermarking`, `repos/aware`,
  `repos/dnn-audio-watermarking`) plus mutually incompatible framework
  stacks (TensorFlow 2.12, PyTorch 1.13), which cannot be combined into one
  portable image. The full benchmark still uses the per-method environments
  described in the top-level README.
- **Distortions included:** digital-level only. Background noise and
  reverberation need the external DEMAND / AIR corpora, and physical-level /
  AI-induced attacks need hardware or separate model pipelines.
- **CPU-only:** the image runs every model on CPU for portability. With the
  default 20 s clip this is interactive; evaluating *all* methods ×
  *all* distortions × *all* settings (≈ 90 settings per method) takes a
  while — the UI shows progress and caps a single run at
  `SOK_DEMO_MAX_JOBS` (default 1200) evaluations.
- `torch` is pinned to 2.0.0 inside the image because `silentcipher`
  requires `torch<=2.0.0`.

## Layout

```
Dockerfile               multi-stage build (audiowmark from source + demo env)
install.sh               one-command installer/launcher
demo/app.py              Gradio UI
demo/methods.py          watermark method registry (embed/decode wrappers)
demo/eval_utils.py       distortions + metrics (reuses scripts/benchmark_utils.py)
demo/prefetch_models.py  bakes model checkpoints into the image at build time
demo/requirements.txt    demo Python dependencies
```

## Troubleshooting

- **Port already in use** — `./install.sh --port 8080`.
- **audiowmark shown as unavailable** — the CLI failed to build or is not on
  `PATH`; rebuild with `./install.sh --rebuild` and check the build log of
  the `audiowmark-build` stage.
- **Model download failures during build** — the prefetch step is
  best-effort; the demo re-downloads checkpoints on first use at runtime
  (requires network in the container).
- **ViSQOL shows `n/a`** — the `visqol-python` package could not compute a
  score for that clip (very short/silent clips can fail); other metrics are
  unaffected.
- Logs: `docker logs -f sok-audio-watermark-demo`.
