# Third-Party Repositories (vendored)

This directory vendors the upstream repositories needed by the Timbre, AWARE,
and RobustDNN (DNN-WM) wrappers, so the benchmark scripts and the Docker demo
work without any external cloning:

```text
repos/
  TimbreWatermarking/       code + trained checkpoint + hifigan vocoder
                            (pruned: only watermarking_model/, the paper's
                            voice-cloning pipeline is not included)
  aware/                    code (AWARE has no pretrained weights — it
                            optimizes a perturbation per clip)
  dnn-audio-watermarking/   code + pretrained embedder/detector SavedModels
                            + samples/message_pool.npy
```

Upstream sources (each keeps its own LICENSE file in its directory):

- Timbre: <https://github.com/TimbreWatermarking/TimbreWatermarking>
- AWARE: <https://github.com/deepmark/aware>
- RobustDNN: <https://github.com/kosta-pmf/dnn-audio-watermarking>

Used by:

- `scripts/13_large_timbre.py`, `scripts/13_large_aware.py`,
  `scripts/13_large_dnn.py` (and matching `scripts/14_*` timing scripts)
- `demo/workers/{timbre,aware,dnn}_worker.py` (Docker demo)

Notes:

- `audioseal` and `wavmark` are loaded from installed Python packages rather
  than from vendored repositories.
- `scripts/13_large_kosta.py` references `repos/audio-watermarking/`, which is
  not vendored — the equivalent FSVC / Patchwork / Norm-space implementations
  live at the repository top level (`fsvc_watermarking_gpu.py`,
  `patchwork_multylayer_watermarking_gpu.py`,
  `norm_space_watermarking_gpu.py`) and are what the demo uses.
- Keep repository names unchanged unless you also update the wrapper scripts.
