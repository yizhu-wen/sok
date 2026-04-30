"""Utilities for FragmentVC inference."""

from pathlib import Path
from typing import Union

import librosa
import numpy as np
from scipy.signal import lfilter


def load_wav(
    audio_path: Union[str, Path], sample_rate: int, trim: bool = False
) -> np.ndarray:
    """Load and preprocess waveform."""
    wav = librosa.load(audio_path, sr=sample_rate)[0]
    wav = wav / (np.abs(wav).max() + 1e-6)
    if trim:
        _, (start_frame, end_frame) = librosa.effects.trim(
            wav, top_db=25, frame_length=512, hop_length=128
        )
        start_frame = max(0, start_frame - 0.1 * sample_rate)
        end_frame = min(len(wav), end_frame + 0.1 * sample_rate)

        start = int(start_frame)
        end = int(end_frame)
        if end - start > 1000:  # prevent empty slice
            wav = wav[start:end]

    return wav


def log_mel_spectrogram(
    x: np.ndarray,
    preemph: float,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    f_min: int,
) -> np.ndarray:
    """Create a log Mel spectrogram from a raw audio signal."""
    x = lfilter([1, -preemph], [1], x)
    magnitude = np.abs(
        librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    )
    mel_fb = librosa.filters.mel(sample_rate, n_fft, n_mels=n_mels, fmin=f_min)
    mel_spec = np.dot(mel_fb, magnitude)
    log_mel_spec = np.log(mel_spec + 1e-9)
    return log_mel_spec.T
