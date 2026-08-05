"""
Distortion + metric helpers for the Gradio demo.

Reuses scripts/benchmark_utils.py so the demo applies exactly the same
digital-level distortions and metric implementations as the paper benchmark.
Only the digital-level distortion family is exposed; background noise and
reverberation need the external DEMAND / AIR corpora and are physical-level
conditions in the paper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
for _p in (str(PROJECT_DIR), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmark_utils as bu  # noqa: E402

# ─── Digital-level distortions (same settings as the benchmark) ──────────────

DIGITAL_DISTORTIONS = {
    "pitch_shift":        list(bu.DISTORTIONS["pitch_shift"]),
    "time_stretch":       list(bu.DISTORTIONS["time_stretch"]),
    "gaussian_noise":     list(bu.DISTORTIONS["gaussian_noise"]),
    "bitcrush":           list(bu.DISTORTIONS["bitcrush"]),
    "mp3_compression":    list(bu.DISTORTIONS["mp3_compression"]),
    "cutting_audio":      list(bu.DISTORTIONS["cutting_audio"]),
    "high_pass_filter":   list(bu.DISTORTIONS["high_pass_filter"]),
    "low_pass_filter":    list(bu.DISTORTIONS["low_pass_filter"]),
    "sample_suppression": list(bu.DISTORTIONS["sample_suppression"]),
    "resampling":         list(bu.DISTORTIONS["resampling"]),
}

DISTORTION_LABELS = {
    "pitch_shift":        "Pitch shift (cents)",
    "time_stretch":       "Time stretch (rate)",
    "gaussian_noise":     "Gaussian noise (SNR dB)",
    "bitcrush":           "Bitcrush (bits)",
    "mp3_compression":    "MP3 compression (kbps)",
    "cutting_audio":      "Cutting audio (% zeroed)",
    "high_pass_filter":   "High-pass filter (Hz)",
    "low_pass_filter":    "Low-pass filter (Hz)",
    "sample_suppression": "Sample suppression (%)",
    "resampling":         "Resampling (kHz)",
}
LABELS_TO_KEY = {v: k for k, v in DISTORTION_LABELS.items()}

MAX_AUDIO_DURATION = bu.MAX_AUDIO_DURATION  # 20 s, same clip rule as benchmark


def apply_distortion(y, sr, name, setting, stem=""):
    return bu.apply_distortion(y, sr, name, setting, stem=stem)


def compute_si_snr(ref, deg):
    return bu.compute_si_snr(ref, deg)


def compute_visqol(ref, deg, sr, mode="speech"):
    return bu.compute_visqol(ref, deg, sr, mode=mode)


def load_audio_clipped(path, stem=""):
    return bu.load_audio_clipped(path, stem=stem)


def align_lengths(ref: np.ndarray, deg: np.ndarray):
    n = min(len(ref), len(deg))
    return ref[:n], deg[:n]


def bit_recovery(embedded_bits: np.ndarray, decoded) -> float:
    """Fraction of embedded bits recovered.  Decode refusal (None) counts as
    0.0, matching the benchmark's treatment of decode refusals."""
    if decoded is None:
        return 0.0
    decoded = np.asarray(decoded).astype(np.int32).ravel()
    n = min(len(embedded_bits), len(decoded))
    if n == 0:
        return 0.0
    return float(np.mean(decoded[:n] == embedded_bits[:n]))


# ─── Spectrogram rendering ────────────────────────────────────────────────────

def spectrogram_png(y: np.ndarray, sr: int, title: str, out_path: str) -> str:
    """Save a log-frequency STFT magnitude spectrogram to out_path (PNG)."""
    import matplotlib
    matplotlib.use("Agg")
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt

    stft_db = librosa.amplitude_to_db(
        np.abs(librosa.stft(y.astype(np.float32), n_fft=1024, hop_length=256)),
        ref=np.max)
    fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=110)
    img = librosa.display.specshow(stft_db, sr=sr, hop_length=256,
                                   x_axis="time", y_axis="log", ax=ax,
                                   cmap="magma")
    ax.set_title(title, fontsize=10)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
