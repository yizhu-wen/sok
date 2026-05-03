# VC Attack Experiments

The VC runners now use a consistent convention:

- `--source`: watermarked victim recordings used for linguistic content.
- `--target`: clean victim recordings used as target-speaker references.
- `--filelist`: rows formatted as `filename|text`; filenames are resolved under `--source`, and for AdaIn-VC, MediumVC, and YourTTS-VC also under `--target`.
- `--output`: writes converted wavs under `converted/<output>`.

## Reviewer Release Notes

The example commands use placeholder paths such as `/path/watermarked` and `/path/clean`. Do not commit local datasets, generated conversions, checkpoints, indexes, or experiment outputs to the anonymous reviewer repository. The root `.gitignore` excludes these artifacts; provide any required model downloads, checksums, or setup notes separately.

## Zero/Few-Shot

AdaIn-VC:

```bash
cd vc/adaptive_voice_conversion
python generate.py --source /path/watermarked --target /path/clean --output adain_zero
```

FragmentVC:

```bash
cd vc/FragmentVC
python generate.py --source /path/watermarked --target /path/clean --output fragment_zero
```

MediumVC:

```bash
cd vc/MediumVC
python generate.py --source /path/watermarked --target /path/clean --output medium_zero
```

YourTTS-VC:

```bash
cd vc/YourTTS
python generate.py --source /path/watermarked --target /path/clean --output yourtts_vc_zero
```

## Adaptive

Adaptive runs fine-tune on clean victim recordings, then pass the fine-tuned checkpoint to the generator.

AdaIn-VC:

```bash
cd vc/adaptive_voice_conversion
python prepare_adaptive.py --clean-dir /path/clean --output-dir experiments/victim_features
python main.py \
  -d experiments/victim_features \
  -train_set train_128 \
  -train_index_file train_samples_128.json \
  -store_model_path experiments/adain_victim \
  --restore_model_path vctk_model.ckpt \
  -iters 50000
python generate.py \
  --source /path/watermarked \
  --target /path/clean \
  --model experiments/adain_victim.ckpt \
  --attr experiments/victim_features/attr.pkl \
  --output adain_adaptive
```

FragmentVC:

```bash
cd vc/FragmentVC
python prepare_adaptive.py --clean-dir /path/clean --output-dir experiments/victim_features
python train.py experiments/victim_features --save_dir experiments/adaptive_ckpts --restore_path checkpoints/fragmentvc.pt
python generate.py \
  --source /path/watermarked \
  --target /path/clean \
  --ckpt-path experiments/adaptive_ckpts/<checkpoint>.pt \
  --output fragment_adaptive
```

MediumVC:

```bash
cd vc/MediumVC
python prepare_adaptive.py --clean-dir /path/clean --output Any2Any/pre_feature/spk_emb_mel_label_adaptive.pkl
python train.py --feature-pkl Any2Any/pre_feature/spk_emb_mel_label_adaptive.pkl --resume-path Any2Any/model/checkpoint-3900.pt
python generate.py \
  --source /path/watermarked \
  --target /path/clean \
  --checkpoint Any2Any/output_adaptive/<run>/model/<checkpoint>.pt \
  --output medium_adaptive
```

## Remaining Gaps

- RVC is wired as an external backend for `RVC-Project/Retrieval-based-Voice-Conversion-WebUI`; see `vc/RVC/README.md`.
- YourTTS-VC still only has the zero/few-shot conversion wrapper here. Adaptive YourTTS/XTTS fine-tuning would need a dedicated Coqui fine-tuning path and checkpoint handoff.
