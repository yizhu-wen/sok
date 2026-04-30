# adaptive_voice_conversion Notes

This package keeps only the inference-time files used by `generate.py`.

External assets are not bundled. According to the official repository, inference
requires:

- `vctk_model.ckpt`
- `attr.pkl`

Place both files in this directory with the original filenames.

Official repo: https://github.com/jjery2243542/adaptive_voice_conversion
