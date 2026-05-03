# RVC Backend

This wrapper targets:

```text
https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
```

The RVC WebUI code is treated as an external dependency. Clone and set it up separately, including its assets such as HuBERT, RMVPE, pretrained weights, ffmpeg, and Python dependencies.

```bash
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI /path/to/RVC
export RVC_REPO=/path/to/RVC
```

## Adaptive Training

RVC is not a reference-wav zero-shot converter. For this experiment, train or fine-tune an RVC model on the clean victim recordings, then use that trained model to convert watermarked source utterances.

```bash
cd vc/RVC
python train.py \
  --rvc-repo /path/to/RVC \
  --clean-dir /path/to/clean_victim_wavs \
  --exp-name victim_rvc \
  --sample-rate 40k \
  --version v2 \
  --if-f0 1 \
  --f0-method rmvpe \
  --gpus 0
```

The trained model and index are written under:

```text
/path/to/RVC/logs/victim_rvc/
```

## Conversion

```bash
cd vc/RVC
python generate.py \
  --rvc-repo /path/to/RVC \
  --source /path/to/watermarked_wavs \
  --target /path/to/clean_victim_wavs \
  --model-path /path/to/RVC/logs/victim_rvc/<model>.pth \
  --index-path /path/to/RVC/logs/victim_rvc/<added_index>.index \
  --output rvc_adaptive
```

`--target` is kept for a consistent experiment interface and sanity-checks that the clean victim reference set exists. At inference time, RVC uses the trained `.pth` model and optional `.index`, not individual target wav files.
