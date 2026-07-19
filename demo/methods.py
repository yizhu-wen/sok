"""
Watermark method registry for the interactive Gradio demo.

Wraps the subset of benchmark methods that can run inside a single Docker
environment (single Python + CPU PyTorch):

  - AudioSeal      (pip `audioseal`,     16-bit payload, fixed)
  - WavMark        (pip `wavmark`,       16-bit payload, fixed)
  - SilentCipher   (pip `silentcipher`,  40-bit payload = 5 bytes, fixed)
  - audiowmark     (CLI binary,          128-bit payload, fixed)
  - FSVC           (in-repo,             variable payload, default 40 bits)
  - Patchwork      (in-repo,             variable payload, default 40 bits)
  - Norm-space     (in-repo,             variable payload, default 40 bits)

Timbre, AWARE and DNN-WM from the paper are NOT included: they require
external upstream repositories, private checkpoints, and incompatible
framework versions (TensorFlow 2.12 / old PyTorch), so they cannot be
packaged into one portable image.  See demo/README.md.

Each method exposes:
  embed(y, sr, bits)      -> (watermarked float32 array, sample_rate)
  decode(y, sr, n_bits)   -> np.ndarray of decoded bits, or None when the
                             method reports that it cannot decode anything.

Default payloads match the ones used by the scripts/13_large_*.py benchmarks.
"""
from __future__ import annotations

import os

# AudioSeal's vendored moshi modules wrap model forwards in torch.compile,
# which torch 2.0.0 (pinned for silentcipher) rejects on Python 3.11+.
# NO_TORCH_COMPILE is moshi's official escape hatch; it is read when the
# audioseal modules are imported, so it must be set before any backend loads.
os.environ.setdefault("NO_TORCH_COMPILE", "1")

import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import librosa  # noqa: E402


# ─── Bit-string helpers ───────────────────────────────────────────────────────

def parse_bits(text: str) -> np.ndarray:
    """Parse a user-supplied bit string ('0'/'1', separators ignored)."""
    cleaned = re.sub(r"[\s,_\-]", "", text or "")
    if not cleaned:
        raise ValueError("payload is empty — enter a string of 0s and 1s")
    if not re.fullmatch(r"[01]+", cleaned):
        raise ValueError("payload may only contain 0s and 1s (spaces allowed)")
    return np.array([int(c) for c in cleaned], dtype=np.int32)


def bits_to_str(bits) -> str:
    return "".join(str(int(b)) for b in bits)


def bits_to_bytes(bits: np.ndarray) -> list[int]:
    return [int("".join(str(int(b)) for b in bits[i:i + 8]), 2)
            for i in range(0, len(bits), 8)]


def bytes_to_bits(byte_vals) -> np.ndarray:
    return np.array([int(c) for b in byte_vals for c in f"{int(b) & 0xFF:08b}"],
                    dtype=np.int32)


def bits_to_hex(bits: np.ndarray) -> str:
    return "".join(f"{b:02x}" for b in bits_to_bytes(bits))


def hex_to_bits(hex_str: str) -> np.ndarray:
    return bytes_to_bits(bytes.fromhex(hex_str))


def _resample(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return y.astype(np.float32)
    return librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)


def _torch_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Method definition ────────────────────────────────────────────────────────

@dataclass
class WatermarkMethod:
    key: str
    name: str
    kind: str                    # "AI-based" or "Traditional"
    min_bits: int
    max_bits: int                # == min_bits for fixed-length payloads
    default_bits: str
    bits_help: str
    load_backend: Callable[[], dict] = field(repr=False)
    unavailable_reason: Optional[str] = None
    _backend: Optional[dict] = field(default=None, repr=False)

    @property
    def fixed_bits(self) -> bool:
        return self.min_bits == self.max_bits

    def validate_bits(self, text: str) -> np.ndarray:
        bits = parse_bits(text)
        if self.fixed_bits:
            if len(bits) != self.min_bits:
                raise ValueError(
                    f"{self.name} requires exactly {self.min_bits} bits "
                    f"(got {len(bits)})")
        elif not (self.min_bits <= len(bits) <= self.max_bits):
            raise ValueError(
                f"{self.name} supports {self.min_bits}–{self.max_bits} bits "
                f"(got {len(bits)})")
        return bits

    def backend(self) -> dict:
        if self.unavailable_reason:
            raise RuntimeError(f"{self.name} is unavailable: {self.unavailable_reason}")
        if self._backend is None:
            self._backend = self.load_backend()
        return self._backend

    def embed(self, y: np.ndarray, sr: int, bits: np.ndarray):
        return self.backend()["embed"](y, sr, bits)

    def decode(self, y: np.ndarray, sr: int, n_bits: int):
        return self.backend()["decode"](y, sr, n_bits)


# ─── AudioSeal ────────────────────────────────────────────────────────────────

def _load_audioseal():
    import torch
    from audioseal import AudioSeal

    device = _torch_device()
    generator = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
    detector = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
    generator.eval()
    detector.eval()
    sr_model = 16000

    def embed(y, sr, bits):
        y16 = _resample(y, sr, sr_model)
        t = torch.from_numpy(y16).float().unsqueeze(0).unsqueeze(0).to(device)
        msg = torch.tensor([bits.tolist()], dtype=torch.int32, device=device)
        with torch.no_grad():
            wm = generator.get_watermark(t, message=msg)
            out = (t + wm).squeeze(0).squeeze(0).cpu().numpy()
        return out.astype(np.float32), sr_model

    def decode(y, sr, n_bits):
        y16 = _resample(y, sr, sr_model)
        t = torch.from_numpy(y16).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            _, msg_out = detector.detect_watermark(t)
        return (msg_out.cpu().numpy()[0] > 0.5).astype(np.int32)

    return {"embed": embed, "decode": decode}


# ─── WavMark ──────────────────────────────────────────────────────────────────

def _load_wavmark():
    import torch
    import wavmark

    device = _torch_device()
    model = wavmark.load_model().to(device)
    model.eval()
    sr_model = 16000

    def embed(y, sr, bits):
        y16 = _resample(y, sr, sr_model)
        if len(y16) < sr_model:
            raise ValueError("WavMark needs at least 1 second of audio at 16 kHz")
        wmed, _ = wavmark.encode_watermark(
            model, y16, bits.astype(np.int32), show_progress=False)
        if wmed is None:
            raise RuntimeError("WavMark could not embed (audio too short?)")
        return wmed.astype(np.float32), sr_model

    def decode(y, sr, n_bits):
        y16 = _resample(y, sr, sr_model)
        if len(y16) < sr_model:
            return None
        decoded, _ = wavmark.decode_watermark(model, y16, show_progress=False)
        if decoded is None:
            return None
        return np.asarray(decoded, dtype=np.int32)

    return {"embed": embed, "decode": decode}


# ─── SilentCipher ─────────────────────────────────────────────────────────────

def _load_silentcipher():
    import torch
    import silentcipher

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = silentcipher.get_model(model_type="44.1k", device=device)

    def embed(y, sr, bits):
        message = bits_to_bytes(bits)          # 5 bytes, each 0–255
        encoded, _ = model.encode_wav(y.astype(np.float32), sr, message,
                                      calc_sdr=False)
        encoded = np.array(encoded, dtype=np.float32)
        if encoded.ndim > 1:
            encoded = encoded[0]
        return encoded, sr

    def decode(y, sr, n_bits):
        try:
            result = model.decode_wav(y.astype(np.float32), sr,
                                      phase_shift_decoding=False)
        except Exception:
            return None
        if result.get("status") and result.get("messages"):
            return bytes_to_bits(result["messages"][0])
        return None

    return {"embed": embed, "decode": decode}


# ─── audiowmark (CLI) ─────────────────────────────────────────────────────────

def _audiowmark_bin() -> Optional[str]:
    cand = os.environ.get("SOK_AUDIOWMARK_BIN", "").strip()
    if cand and Path(cand).exists():
        return cand
    return shutil.which("audiowmark")


def _load_audiowmark():
    import soundfile as sf

    binary = _audiowmark_bin()
    if binary is None:
        raise RuntimeError("audiowmark binary not found")
    env = dict(os.environ)
    lib = os.environ.get("SOK_AUDIOWMARK_LIB", "").strip()
    if lib:
        env["LD_LIBRARY_PATH"] = ":".join(
            [lib, env.get("LD_LIBRARY_PATH", "")]).strip(":")

    def embed(y, sr, bits):
        payload_hex = bits_to_hex(bits)        # 128 bits -> 32 hex chars
        with tempfile.TemporaryDirectory() as tmp:
            in_wav = Path(tmp) / "in.wav"
            out_wav = Path(tmp) / "out.wav"
            sf.write(str(in_wav), y, sr)
            result = subprocess.run(
                [binary, "add", str(in_wav), str(out_wav), payload_hex],
                capture_output=True, env=env)
            if result.returncode != 0:
                raise RuntimeError(
                    f"audiowmark add failed: {result.stderr.decode()[:300]}")
            wmed, wmed_sr = sf.read(str(out_wav))
        if wmed.ndim > 1:
            wmed = wmed.mean(axis=1)
        return wmed.astype(np.float32), wmed_sr

    def decode(y, sr, n_bits):
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "test.wav"
            sf.write(str(wav_path), y, sr)
            result = subprocess.run([binary, "get", str(wav_path)],
                                    capture_output=True, text=True, env=env)
        # Output lines look like:  "pattern  0:05 <32 hex chars> 1.32 0.059 A"
        # and, when several blocks agree:  "pattern    all <32 hex chars> ..."
        best = None
        for line in result.stdout.splitlines():
            m = re.match(r"\s*pattern\s+(\S+)\s+([0-9a-fA-F]{32})\b", line)
            if m:
                if m.group(1) == "all":
                    best = m.group(2)
                    break
                if best is None:
                    best = m.group(2)
        if best is None:
            return None
        return hex_to_bits(best)

    return {"embed": embed, "decode": decode}


# ─── FSVC / Patchwork / Norm-space (in-repo classic methods) ─────────────────

_KOSTA_SR = 16000
# Minimum watermarked-signal samples per payload bit for each classic method.
# Below these, the per-bit frames become too short for the transform bands.
_KOSTA_MIN_SAMPLES_PER_BIT = {"fsvc": 128, "patchwork": 32, "normspace": 32}


def _kosta_guard(algo: str, y16: np.ndarray, n_bits: int):
    need = _KOSTA_MIN_SAMPLES_PER_BIT[algo] * n_bits
    if len(y16) < need:
        raise ValueError(
            f"audio too short for {n_bits} bits: needs at least "
            f"{need / _KOSTA_SR:.2f}s at 16 kHz, got {len(y16) / _KOSTA_SR:.2f}s. "
            f"Use fewer bits or longer audio.")


def _load_kosta(algo: str):
    from fsvc_watermarking_gpu import (fsvc_watermark_detection,
                                       fsvc_watermark_embedding)
    from norm_space_watermarking_gpu import (norm_space_watermark_detection,
                                             norm_space_watermark_embedding)
    from patchwork_multylayer_watermarking_gpu import (
        patchwork_multilayer_watermark_detection,
        patchwork_multilayer_watermark_embedding)

    device = _torch_device()

    def embed(y, sr, bits):
        y16 = _resample(y, sr, _KOSTA_SR)
        _kosta_guard(algo, y16, len(bits))
        if algo == "fsvc":
            wmed = fsvc_watermark_embedding(y16, bits, _KOSTA_SR, device=device)
        elif algo == "patchwork":
            wmed = patchwork_multilayer_watermark_embedding(
                y16, bits, _KOSTA_SR, device=device)
        else:
            wmed = norm_space_watermark_embedding(y16, bits, device=device)
        return np.array(wmed, dtype=np.float32), _KOSTA_SR

    def decode(y, sr, n_bits):
        y16 = _resample(y, sr, _KOSTA_SR)
        if algo == "fsvc":
            bits = fsvc_watermark_detection(y16, n_bits, _KOSTA_SR, device=device)
        elif algo == "patchwork":
            bits = patchwork_multilayer_watermark_detection(
                y16, n_bits, _KOSTA_SR, device=device)
        else:
            bits = norm_space_watermark_detection(y16, n_bits, device=device)
        return np.asarray(bits, dtype=np.int32)

    return {"embed": embed, "decode": decode}


# ─── Default payloads (identical to the scripts/13_large_*.py benchmarks) ────

_AUDIOSEAL_MSG = [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0]
_WAVMARK_MSG = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
_SILENTCIPHER_BYTES = [123, 45, 67, 89, 12]
_AUDIOWMARK_HEX = "0102030405060708090a0b0c0d0e0f10"
_KOSTA_MSG = [
    1, 0, 1, 0, 1, 1, 0, 0, 1, 0,
    1, 1, 0, 1, 0, 0, 1, 0, 1, 1,
    0, 0, 1, 1, 0, 1, 0, 1, 1, 0,
    0, 1, 0, 1, 0, 1, 1, 0, 0, 1,
]


def build_registry() -> list[WatermarkMethod]:
    methods = [
        WatermarkMethod(
            key="audioseal", name="AudioSeal", kind="AI-based",
            min_bits=16, max_bits=16,
            default_bits=bits_to_str(_AUDIOSEAL_MSG),
            bits_help="exactly 16 bits",
            load_backend=_load_audioseal),
        WatermarkMethod(
            key="wavmark", name="WavMark", kind="AI-based",
            min_bits=16, max_bits=16,
            default_bits=bits_to_str(_WAVMARK_MSG),
            bits_help="exactly 16 bits (pattern mode)",
            load_backend=_load_wavmark),
        WatermarkMethod(
            key="silentcipher", name="SilentCipher", kind="AI-based",
            min_bits=40, max_bits=40,
            default_bits=bits_to_str(bytes_to_bits(_SILENTCIPHER_BYTES)),
            bits_help="exactly 40 bits (5 bytes)",
            load_backend=_load_silentcipher),
        WatermarkMethod(
            key="audiowmark", name="audiowmark", kind="Traditional",
            min_bits=128, max_bits=128,
            default_bits=bits_to_str(hex_to_bits(_AUDIOWMARK_HEX)),
            bits_help="exactly 128 bits (16 bytes)",
            load_backend=_load_audiowmark),
        WatermarkMethod(
            key="fsvc", name="FSVC", kind="Traditional",
            min_bits=8, max_bits=64,
            default_bits=bits_to_str(_KOSTA_MSG),
            bits_help="8–64 bits (default 40)",
            load_backend=lambda: _load_kosta("fsvc")),
        WatermarkMethod(
            key="patchwork", name="Patchwork", kind="Traditional",
            min_bits=8, max_bits=64,
            default_bits=bits_to_str(_KOSTA_MSG),
            bits_help="8–64 bits (default 40)",
            load_backend=lambda: _load_kosta("patchwork")),
        WatermarkMethod(
            key="normspace", name="Norm-space", kind="Traditional",
            min_bits=8, max_bits=64,
            default_bits=bits_to_str(_KOSTA_MSG),
            bits_help="8–64 bits (default 40)",
            load_backend=lambda: _load_kosta("normspace")),
    ]
    for m in methods:
        if m.key == "audiowmark" and _audiowmark_bin() is None:
            m.unavailable_reason = ("audiowmark binary not found — set "
                                    "SOK_AUDIOWMARK_BIN or install audiowmark")
    return methods


REGISTRY = build_registry()
METHODS_BY_NAME = {m.name: m for m in REGISTRY}
