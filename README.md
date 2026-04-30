# SOK Open-Science Evaluation Framework

This package contains the benchmark and reporting code used to evaluate audio
watermarking methods across multiple speech and music datasets.

The release is intentionally curated:
- Local machine paths and user-specific configuration have been removed.
- Private logs and ad hoc trial outputs are not included.
- Generated benchmark results should be reproduced from the code in this
  package rather than inferred from local experiment leftovers.

## Included

- `scripts/`: benchmark runners, timing scripts, reporting utilities, and
  preflight checks
- `ai_distortions/`: AI-based TTS and voice-conversion distortion pipelines
  packaged with only inference-time code and no bundled checkpoints
- `requirements/`: per-environment dependency snapshots
- `repos/README.md`: expected layout for third-party repositories
- top-level classic watermarking implementations used by the Kosta baselines

## Not Included

- local virtual environments
- generated logs
- generated benchmark outputs
- reduced-sample or debugging artifacts from the private worktree

## Expected Layout

Populate the third-party repositories under `repos/` as described in
`repos/README.md`. Generated outputs will be written under:

- `results/benchmark/`
- `results/timing/`
- `results/visqol_unwm/`
- `logs/`

## Environment Setup

This project uses separate environments for different algorithms. The
requirements files are kept per environment rather than as one monolithic
`requirements.txt`.

Examples:

```bash
python -m venv envs/viz
envs/viz/bin/pip install -r requirements/viz.txt

python -m venv envs/audioseal
envs/audioseal/bin/pip install -r requirements/audioseal.txt
```

Some environments require CUDA-enabled PyTorch or TensorFlow builds. Treat the
provided requirement files as reference manifests and adjust CUDA wheel indexes
or platform-specific packages for your machine as needed.

## Configuration

Set the required paths with environment variables or edit `scripts/config.py`.
The most important variables are:

- `SOK_DATASET_DIR`
- `SOK_NOISE_DIR`
- `SOK_RIR_DIR`
- `SOK_AUDIOWMARK_BIN`
- `SOK_NVIDIA_BASE`
- `SOK_PTXAS_DIR`

Run the preflight check before launching benchmarks:

```bash
envs/viz/bin/python scripts/preflight.py
```

## Running Full Benchmarks

Examples:

```bash
envs/audioseal/bin/python scripts/13_large_audioseal.py
envs/timbre/bin/python scripts/13_large_timbre.py
envs/dnn_wm/bin/python scripts/13_large_dnn.py
```

These scripts default to the dataset roots defined in `scripts/config.py`.

## Reporting

Build a dataset workbook from completed benchmark JSON files:

```bash
python3 scripts/18_dataset_full_excel.py \
  --dataset speech_ljspeech \
  --out results/speech_ljspeech_full.xlsx \
  --suffixes __plain__
```

## Reproducibility Notes

If you need a smaller validation subset while bringing up a new machine, use
explicit environment variables such as `SOK_N_SAMPLES`. Keep those subset runs
separate from the full benchmark outputs.
