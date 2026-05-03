import sys
from pathlib import Path


TTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TTS_DIR))

from common.vocoder import train_hifigan  # noqa: E402


if __name__ == "__main__":
    train_hifigan("fastspeech2")
