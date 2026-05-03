import argparse
import sys
from pathlib import Path


TTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TTS_DIR))

from common.experiment import add_generation_args, synthesize_filelist  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate FastSpeech-2 speech for the TTS attack experiments.")
    add_generation_args(parser, default_model_name=None)
    args = parser.parse_args()

    if not args.model_path and not args.model_name:
        parser.error(
            "FastSpeech-2 has no default released Coqui checkpoint in this repo. "
            "Provide --model-path/--config-path after fine-tuning, or pass --model-name for your installed checkpoint."
        )

    print("\nGenerating synthesized audios...")
    synthesize_filelist(args)


if __name__ == "__main__":
    main()
