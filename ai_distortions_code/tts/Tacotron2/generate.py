import argparse
import sys
from pathlib import Path


TTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TTS_DIR))

from common.experiment import add_generation_args, synthesize_filelist  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate Tacotron-2 speech for the TTS attack experiments.")
    add_generation_args(parser, default_model_name="tts_models/en/ljspeech/tacotron2-DDC")
    args = parser.parse_args()

    if not args.model_path and not args.model_name:
        parser.error("Provide --model-name for pretrained inference or --model-path/--config-path for a fine-tuned model.")

    print("\nGenerating synthesized audios...")
    synthesize_filelist(args)


if __name__ == "__main__":
    main()
