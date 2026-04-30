"""
Shared utilities for the benchmarking pipeline (script 12).
Distortion functions, quality metrics (SI-SNR, PESQ, ESTOI).

Usage in a benchmark script:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import benchmark_utils as bu
"""

import hashlib
import subprocess
import tempfile
import threading
from pathlib import Path

import librosa
import numpy as np
import scipy.io
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR     = Path(__file__).parent.parent
DATA_DIR        = PROJECT_DIR / "data"
OUT_DIR         = PROJECT_DIR / "outputs"
RESULTS_DIR     = PROJECT_DIR / "results" / "benchmark"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from config import NOISE_DIR, RIR_DIR
BG_NOISE_SNR_DB = 0.0

# ─── Distortion settings ──────────────────────────────────────────────────────

DISTORTIONS = {
    "pitch_shift":        [-100, -50, -25, -12.5, -6.25, 6.25, 12.5, 25, 50, 100],
    "time_stretch":       [0.25, 0.5, 0.75, 0.9, 0.95, 1.05, 1.1, 1.25, 1.5, 1.75, 2.0],
    "gaussian_noise":     [-20, -15, -10, -5, 0, 5, 10, 20, 30, 40],
    "bitcrush":           [2, 4, 6, 8, 10, 12],
    "mp3_compression":    [8, 16, 24, 32, 40, 48, 56, 64, 128, 192, 256, 320],
    "background_noise":   ["DKITCHEN", "DLIVING", "DWASHING",
                           "NFIELD", "NPARK", "NRIVER",
                           "OHALLWAY", "OMEETING", "OOFFICE",
                           "PCAFETER", "PRESTO", "PSTATION",
                           "SCAFE", "SPSQUARE", "STRAFFIC",
                           "TBUS", "TCAR", "TMETRO"],
    "cutting_audio":      [25, 50, 75, 90, 95, 97],
    "high_pass_filter":   [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
    "low_pass_filter":    [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
    "sample_suppression": [0.5, 1, 2.5, 5, 10, 25],
    "resampling":         [2, 4, 8, 16, 22.05],
    "reverberation":      ["bin_aula_carolina", "bin_booth", "bin_lecture",
                           "bin_meeting", "bin_office", "bin_stairway",
                           "phone_bathroom", "phone_corridor", "phone_kitchen",
                           "phone_lecture", "phone_lecture1", "phone_meeting",
                           "phone_office", "phone_stairway", "phone_stairway1",
                           "phone_stairway2"],
}

_RIR_MAP = {
    "bin_aula_carolina": ("binaural rooms", "aula_carolina"),
    "bin_booth":         ("binaural rooms", "booth"),
    "bin_lecture":       ("binaural rooms", "lecture"),
    "bin_meeting":       ("binaural rooms", "meeting"),
    "bin_office":        ("binaural rooms", "office"),
    "bin_stairway":      ("binaural rooms", "stairway"),
    "phone_bathroom":    ("phone_mock-up_rooms", "bathroom"),
    "phone_corridor":    ("phone_mock-up_rooms", "corridor"),
    "phone_kitchen":     ("phone_mock-up_rooms", "kitchen"),
    "phone_lecture":     ("phone_mock-up_rooms", "lecture"),
    "phone_lecture1":    ("phone_mock-up_rooms", "lecture1"),
    "phone_meeting":     ("phone_mock-up_rooms", "meeting"),
    "phone_office":      ("phone_mock-up_rooms", "office"),
    "phone_stairway":    ("phone_mock-up_rooms", "stairway"),
    "phone_stairway1":   ("phone_mock-up_rooms", "stairway1"),
    "phone_stairway2":   ("phone_mock-up_rooms", "stairway2"),
}

# ─── Distortion functions ─────────────────────────────────────────────────────

def _pitch_shift(y, sr, cents):
    return librosa.effects.pitch_shift(y.astype(np.float32), sr=sr, n_steps=cents / 100.0)

def _time_stretch(y, sr, rate):
    return librosa.effects.time_stretch(y.astype(np.float32), rate=rate)

def _gaussian_noise(y, sr, snr_db, rng=None):
    sig_power = float(np.mean(y.astype(np.float64) ** 2))
    if sig_power == 0.0:
        return y.copy()
    _rng = rng if rng is not None else np.random
    noise = _rng.randn(len(y)) * np.sqrt(sig_power / (10.0 ** (snr_db / 10.0)))
    return (y.astype(np.float64) + noise).astype(np.float32)

def _bitcrush(y, sr, bits):
    scale = 2 ** (bits - 1)
    return np.clip(np.round(y.astype(np.float64) * scale) / scale, -1.0, 1.0).astype(np.float32)

def _mp3_compression(y, sr, kbps):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        in_wav, mp3_file, out_wav = tmp / "in.wav", tmp / "c.mp3", tmp / "out.wav"
        sf.write(str(in_wav), y, sr)
        subprocess.run(["ffmpeg", "-y", "-i", str(in_wav), "-b:a", f"{kbps}k", str(mp3_file)],
                       capture_output=True, check=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(mp3_file), str(out_wav)],
                       capture_output=True, check=True)
        y_out, _ = sf.read(str(out_wav))
    if y_out.ndim > 1:
        y_out = y_out.mean(axis=1)
    return y_out.astype(np.float32)

def _load_demand_noise(noise_type, length, sr):
    noise_path = NOISE_DIR / noise_type / "ch01.wav"
    noise_raw, noise_sr = sf.read(str(noise_path))
    if noise_raw.ndim > 1:
        noise_raw = noise_raw[:, 0]
    noise_raw = noise_raw.astype(np.float64)
    if noise_sr != sr:
        noise_raw = librosa.resample(noise_raw.astype(np.float32),
                                     orig_sr=noise_sr, target_sr=sr).astype(np.float64)
    if len(noise_raw) < length:
        noise_raw = np.tile(noise_raw, int(np.ceil(length / len(noise_raw))))
    return noise_raw[:length]

def _background_noise(y, sr, noise_type):
    sig_power = float(np.mean(y.astype(np.float64) ** 2))
    if sig_power == 0.0:
        return y.copy()
    noise = _load_demand_noise(noise_type, len(y), sr)
    noise_power = float(np.mean(noise ** 2))
    if noise_power == 0.0:
        return y.copy()
    noise *= np.sqrt((sig_power / (10.0 ** (BG_NOISE_SNR_DB / 10.0))) / noise_power)
    return (y.astype(np.float64) + noise).astype(np.float32)

def _load_rir(room_key, sr):
    rtype, rname = _RIR_MAP[room_key]
    mat_file = sorted((RIR_DIR / rtype / rname).glob("*.mat"))[0]
    m = scipy.io.loadmat(str(mat_file))
    h = m["h_air"].flatten().astype(np.float64)
    fs_rir = int(m["air_info"]["fs"][0, 0].flat[0])
    if fs_rir != sr:
        h = librosa.resample(h.astype(np.float32), orig_sr=fs_rir, target_sr=sr).astype(np.float64)
    return h

def _reverberation(y, sr, room_key):
    h = _load_rir(room_key, sr)
    return fftconvolve(y.astype(np.float64), h)[:len(y)].astype(np.float32)

def _cutting_audio(y, sr, pct):
    y_out = y.copy()
    cut_len = int(len(y) * pct / 100.0)
    start = (len(y) - cut_len) // 2
    y_out[start:start + cut_len] = 0.0
    return y_out

def _high_pass_filter(y, sr, cutoff_hz):
    sos = butter(4, min(cutoff_hz / (sr / 2.0), 0.9999), btype="high", output="sos")
    return sosfilt(sos, y).astype(np.float32)

def _low_pass_filter(y, sr, cutoff_hz):
    sos = butter(4, min(cutoff_hz / (sr / 2.0), 0.9999), btype="low", output="sos")
    return sosfilt(sos, y).astype(np.float32)

def _sample_suppression(y, sr, pct, rng=None):
    y_out = y.copy()
    n_zero = max(1, int(len(y) * pct / 100.0))
    _rng = rng if rng is not None else np.random
    y_out[_rng.choice(len(y), size=n_zero, replace=False)] = 0.0
    return y_out

def _resampling(y, sr, target_sr_khz):
    target_sr = int(round(target_sr_khz * 1000))
    y_down = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    y_up   = librosa.resample(y_down, orig_sr=target_sr, target_sr=sr)
    return y_up[:len(y)].astype(np.float32)

_DISTORTION_FNS = {
    "pitch_shift":        _pitch_shift,
    "time_stretch":       _time_stretch,
    "gaussian_noise":     _gaussian_noise,
    "bitcrush":           _bitcrush,
    "mp3_compression":    _mp3_compression,
    "background_noise":   _background_noise,
    "cutting_audio":      _cutting_audio,
    "high_pass_filter":   _high_pass_filter,
    "low_pass_filter":    _low_pass_filter,
    "sample_suppression": _sample_suppression,
    "resampling":         _resampling,
    "reverberation":      _reverberation,
}

_RANDOM_DISTORTIONS = {"gaussian_noise", "sample_suppression"}


def apply_distortion(y, sr, name, setting, stem=""):
    """Apply a named distortion.  stem makes the random seed file-specific so
    that each file gets a distinct-but-reproducible noise/suppression draw.
    Stochastic distortions use a local RandomState — no global seed mutation."""
    seed = int(hashlib.md5(f"{stem}:{name}:{setting}".encode()).hexdigest()[:8], 16) % (2 ** 31)
    fn = _DISTORTION_FNS[name]
    if name in _RANDOM_DISTORTIONS:
        return fn(y, sr, setting, rng=np.random.RandomState(seed))
    return fn(y, sr, setting)

def setting_label(value):
    s = str(value)
    return s.replace("-", "neg").replace(".", "p")

# ─── Quality metrics ──────────────────────────────────────────────────────────

def compute_si_snr(ref: np.ndarray, deg: np.ndarray) -> float:
    """Scale-Invariant SNR (dB). ref = clean original, deg = distorted."""
    n = min(len(ref), len(deg))
    s = ref[:n].astype(np.float64)
    s_hat = deg[:n].astype(np.float64)
    s = s - s.mean()
    s_hat = s_hat - s_hat.mean()
    dot = np.dot(s_hat, s)
    s_norm_sq = np.dot(s, s) + 1e-8
    s_target = (dot / s_norm_sq) * s
    noise = s_hat - s_target
    si_snr = 10.0 * np.log10((np.dot(s_target, s_target) + 1e-8) /
                              (np.dot(noise, noise) + 1e-8))
    return float(si_snr)


def compute_pesq(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """PESQ score. Resamples to 16 kHz if needed. Returns NaN on failure."""
    try:
        from pesq import pesq as pesq_fn
        target_sr = 16000
        n = min(len(ref), len(deg))
        r = ref[:n].astype(np.float32)
        d = deg[:n].astype(np.float32)
        if sr != target_sr:
            r = librosa.resample(r, orig_sr=sr, target_sr=target_sr)
            d = librosa.resample(d, orig_sr=sr, target_sr=target_sr)
        n2 = min(len(r), len(d))
        return float(pesq_fn(target_sr, r[:n2], d[:n2], "wb"))
    except Exception:
        return float("nan")


def compute_estoi(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """Extended STOI score [0, 1]. Returns NaN on failure."""
    try:
        from pystoi import stoi
        n = min(len(ref), len(deg))
        return float(stoi(ref[:n].astype(np.float64),
                          deg[:n].astype(np.float64), sr, extended=True))
    except Exception:
        return float("nan")


# ViSQOL target sample rates per mode (per Google ViSQOL docs)
_VISQOL_SR = {"speech": 16000, "audio": 48000}

# Thread-local storage: each worker thread gets its own VisqolApi instances.
# This avoids the GIL bottleneck and is required for parallel distortion processing.
_visqol_local = threading.local()


def compute_visqol(ref: np.ndarray, deg: np.ndarray, sr: int,
                   mode: str = "speech") -> float:
    """ViSQOL MOS-LQO score [1–5].
    mode='speech' operates at 16 kHz; mode='audio' operates at 48 kHz.
    Returns NaN on failure (e.g. env missing visqol-python).
    Uses thread-local VisqolApi instances — safe for ThreadPoolExecutor."""
    try:
        attr = f"api_{mode}"
        if not hasattr(_visqol_local, attr):
            from visqol import VisqolApi
            api = VisqolApi()
            api.create(mode=mode)
            setattr(_visqol_local, attr, api)
        api = getattr(_visqol_local, attr)
        target_sr = _VISQOL_SR[mode]
        n = min(len(ref), len(deg))
        r = ref[:n].astype(np.float32)
        d = deg[:n].astype(np.float32)
        if sr != target_sr:
            r = librosa.resample(r, orig_sr=sr, target_sr=target_sr)
            d = librosa.resample(d, orig_sr=sr, target_sr=target_sr)
        n2 = min(len(r), len(d))
        result = api.measure_from_arrays(r[:n2], d[:n2], target_sr)
        return float(result.moslqo)
    except Exception:
        return float("nan")


def compute_all_metrics(ref: np.ndarray, deg: np.ndarray, sr: int,
                        visqol_mode: str = "speech") -> dict:
    """Compute SI-SNR, PESQ, ESTOI, ViSQOL between ref (original) and deg (distorted).
    visqol_mode: 'speech' for speech datasets (16 kHz), 'audio' for music/event (48 kHz)."""
    return {
        "si_snr": compute_si_snr(ref, deg),
        "pesq":   compute_pesq(ref, deg, sr),
        "estoi":  compute_estoi(ref, deg, sr),
        "visqol": compute_visqol(ref, deg, sr, mode=visqol_mode),
    }


# ─── Audio loading helpers ────────────────────────────────────────────────────

MAX_AUDIO_DURATION = 20.0   # seconds — clips longer audio to this length


def load_audio(path):
    """Load mono float32 audio, return (array, sr)."""
    y, sr = sf.read(str(path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32), sr


def clip_audio(y: np.ndarray, sr: int, stem: str = "",
               max_duration: float = MAX_AUDIO_DURATION) -> np.ndarray:
    """
    If audio is longer than max_duration seconds, randomly select a contiguous
    segment of exactly max_duration seconds.  The random offset is seeded from
    the stem name so the same file always yields the same segment across runs
    and across algorithms.
    """
    max_samples = int(max_duration * sr)
    if len(y) <= max_samples:
        return y
    seed  = int(hashlib.md5(stem.encode()).hexdigest()[:8], 16) % (2 ** 31)
    rng   = np.random.RandomState(seed)
    start = rng.randint(0, len(y) - max_samples)
    return y[start : start + max_samples]


def load_audio_clipped(path, stem: str = "", max_duration: float = MAX_AUDIO_DURATION):
    """Load mono float32 audio and clip to max_duration if needed."""
    y, sr = load_audio(path)
    return clip_audio(y, sr, stem=stem, max_duration=max_duration), sr


def get_stems():
    return [f.stem for f in sorted(DATA_DIR.glob("*.wav"))]


def get_original(stem):
    return load_audio(DATA_DIR / f"{stem}.wav")


def get_watermarked(algo, stem):
    return load_audio(OUT_DIR / algo / f"{stem}_watermarked.wav")
