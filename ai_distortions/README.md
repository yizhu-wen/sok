# AI Distortions

This folder contains the AI-based distortion pipelines that were used outside
the main watermark benchmark runners.

Included methods:

- `tts/YourTTS`: zero-shot voice-cloned text-to-speech
- `vc/YourTTS`: zero-shot voice conversion with the released YourTTS model
- `vc/FragmentVC`: any-to-any voice conversion
- `vc/MediumVC`: any-to-any voice conversion
- `vc/adaptive_voice_conversion`: one-shot voice conversion with adaptive
  instance normalization

## What Is Included

- inference entrypoints
- only the imported inference-time Python modules
- text filelists listing the evaluation filenames
- minimal notes describing where upstream weights should be placed

## What Is Not Included

- generated audio
- local demo audio waveforms
- downloaded pretrained checkpoints
- training scripts and preprocessing code that is not needed for inference

## Checkpoints

No model checkpoints are bundled here. Based on the local filenames and the
corresponding official repositories, the checkpoints found in the private tree
appear to be upstream pretrained/open-weight assets rather than private
fine-tuned releases.

Use the following upstream sources:

- YourTTS
  - Coqui TTS docs: https://coqui-tts.readthedocs.io/en/latest/models/vits.html
  - Official repo: https://github.com/Edresson/YourTTS
  - The packaged scripts use the released model name
    `tts_models/multilingual/multi-dataset/your_tts`, so weights are fetched by
    Coqui TTS at runtime.
- FragmentVC
  - Official repo: https://github.com/yistLin/FragmentVC
  - The official README states that the pretrained FragmentVC model and vocoder
    are provided via the repo's Releases section.
  - The same README links the upstream wav2vec checkpoint:
    `https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt`
- MediumVC
  - Official repo: https://github.com/BrightGu/MediumVC
  - The official README instructs users to download the pretrained model and
    then edit `Any2Any/infer/infer_config.yaml`.
- adaptive_voice_conversion
  - Official repo: https://github.com/jjery2243542/adaptive_voice_conversion
  - The official README says to download the pretrained model and the matching
    normalization parameters from the links referenced there.

## Audio Filename Lists

- `tts/YourTTS/ljs_audio_text_test_filelist.txt`
- `vc/ljs_audio_text_test_filelist.txt`
- `audio_file_names/mediumvc_demo_dataset_filenames.txt`

The actual waveform files are intentionally not bundled.
