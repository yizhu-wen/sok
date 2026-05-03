# Third-Party Repository Layout

Several benchmark scripts expect external repositories to be available under
fixed relative paths inside `repos/`. This reviewer package includes this
README only; clone or copy the upstream repositories here before running the
corresponding wrappers.

Expected layout:

```text
repos/
  TimbreWatermarking/
  aware/
  audio-watermarking/
  dnn-audio-watermarking/
```

Used by:

- `scripts/13_large_timbre.py`
- `scripts/13_large_aware.py`
- `scripts/13_large_kosta.py`
- `scripts/13_large_dnn.py`
- matching timing scripts under `scripts/14_*`

Notes:

- `audioseal` and `wavmark` are loaded from installed Python packages rather
  than from vendored repositories.
- Place any required upstream checkpoints or model assets in the locations
  expected by the original repositories.
- Keep repository names unchanged unless you also update the wrapper scripts.
