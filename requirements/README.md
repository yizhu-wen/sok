# Dependency Notes

The project uses multiple environments because the watermarking methods depend
on incompatible combinations of Python, PyTorch, TensorFlow, and CUDA wheels.

Files:

- `audioseal.txt`
- `aware.txt`
- `dnn_wm.txt`
- `kosta.txt`
- `silentcipher.txt`
- `timbre.txt`
- `viz.txt`
- `wavmark.txt`

Practical guidance:

- Use each file as a reference environment manifest, not as a promise of
  one-command portability across all machines.
- GPU environments may require changing the wheel index, CUDA build, or Python
  minor version to match your host.
- Install `ffmpeg` separately at the system level.
- Install the `audiowmark` binary separately and expose it through
  `SOK_AUDIOWMARK_BIN`.
