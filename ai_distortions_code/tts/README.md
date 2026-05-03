# TTS Attack Experiments

This folder contains TTS attack runners for the three backbones used in the paper setup:

- `YourTTS/`: existing zero/few-shot speaker-reference synthesis.
- `Tacotron2/`: Tacotron-2 adaptive and adaptive* experiments.
- `FastSpeech2/`: FastSpeech-2 adaptive and adaptive* experiments.

The Tacotron-2 and FastSpeech-2 code assumes the attacker has only victim watermarked recordings plus matching transcripts. It prepares those files as a single-speaker LJSpeech-style dataset and then fine-tunes the selected acoustic model. The adaptive* setting additionally fine-tunes a HiFi-GAN vocoder on the same watermarked recordings.

## Reviewer Release Notes

The example commands use placeholder paths such as `/path/to/watermarked_wavs`. Do not commit local datasets, generated audio, checkpoints, or experiment outputs to the anonymous reviewer repository. The root `.gitignore` excludes these artifacts; provide any required downloads, checksums, or setup notes separately.

## Inputs

The scripts expect filelists in the repo format:

```text
LJ045-0096.wav|Mrs. De Mohrenschildt thought that Oswald,
```

The `--target` directory must contain the matching watermarked wav files by filename.

## Tacotron-2

Run from `tts/Tacotron2`.

Adaptive acoustic fine-tuning:

```bash
python train.py \
  --target /path/to/watermarked_wavs \
  --train-filelist ../YourTTS/ljs_audio_text_test_filelist.txt \
  --output victim_tacotron2 \
  --restore-path /path/to/pretrained_tacotron2_checkpoint.pth
```

Synthesize with the fine-tuned acoustic model:

```bash
python generate.py \
  --model-path experiments/victim_tacotron2/coqui \
  --output victim_tacotron2
```

Adaptive* vocoder fine-tuning:

```bash
python train_vocoder.py \
  --target /path/to/watermarked_wavs \
  --train-filelist ../YourTTS/ljs_audio_text_test_filelist.txt \
  --output victim_tacotron2_star \
  --restore-path /path/to/pretrained_hifigan_checkpoint.pth
```

Synthesize with both fine-tuned acoustic model and fine-tuned vocoder:

```bash
python generate.py \
  --model-path experiments/victim_tacotron2/coqui \
  --vocoder-path experiments/victim_tacotron2_star/hifigan \
  --output victim_tacotron2_star
```

## FastSpeech-2

Run from `tts/FastSpeech2`.

Adaptive acoustic fine-tuning:

```bash
python train.py \
  --target /path/to/watermarked_wavs \
  --train-filelist ../YourTTS/ljs_audio_text_test_filelist.txt \
  --output victim_fastspeech2 \
  --restore-path /path/to/pretrained_fastspeech2_checkpoint.pth
```

If the FastSpeech-2 run needs external attention masks for duration training, add:

```bash
--compute-attention-masks
```

or provide the mask extractor explicitly:

```bash
--compute-attention-masks \
--attention-model-path /path/to/tacotron2_dca.pth \
--attention-config-path /path/to/tacotron2_dca_config.json
```

Synthesize with the fine-tuned acoustic model:

```bash
python generate.py \
  --model-path experiments/victim_fastspeech2/coqui \
  --output victim_fastspeech2
```

Adaptive* uses the same vocoder step:

```bash
python train_vocoder.py \
  --target /path/to/watermarked_wavs \
  --train-filelist ../YourTTS/ljs_audio_text_test_filelist.txt \
  --output victim_fastspeech2_star \
  --restore-path /path/to/pretrained_hifigan_checkpoint.pth

python generate.py \
  --model-path experiments/victim_fastspeech2/coqui \
  --vocoder-path experiments/victim_fastspeech2_star/hifigan \
  --output victim_fastspeech2_star
```

## Zero/Few-Shot Reference Mode

Tacotron-2 and FastSpeech-2 are not speaker-reference models in the same way as YourTTS. For a reference-audio baseline with these acoustic backbones, the generation scripts expose Coqui's `tts_with_vc_to_file` path:

```bash
python generate.py \
  --target /path/to/watermarked_wavs \
  --output tacotron2_reference_baseline \
  --voice-convert
```

For FastSpeech-2, also provide `--model-path` or `--model-name` because no default FastSpeech-2 checkpoint is bundled here.
