from __future__ import annotations

import numpy as np
import torch

from torch_watermarking_utils import dct_ortho, idct_ortho, resolve_device, to_tensor_1d


d0 = 0.83
delta = 3.5793656
D = 0.65
gamma1 = 0.136
gamma2 = 0.181
alpha = 3


def encrypt_watermark(w):
  n = len(w)
  d = np.empty(w.shape)
  d[0] = d0
  e = np.empty(w.shape, dtype="int")
  for i in range(1, n):
    d[i] = delta * d[i - 1] * (1 - d[i - 1])

  e = (d >= D).astype("int")
  return w ^ e


def decrypt_watermark(wp, e):
  return wp ^ e


def modify_svd_pair(l1, l2, watermark_bit):
  if watermark_bit == 0:
    if l1 / l2 > 1 / (1 + alpha):
      l1 = (l1 + l2 * (1 + alpha)) / (alpha**2 + 2 * alpha + 2)
      l2 = (1 + alpha) * l1
  else:
    if l1 / l2 < 1 + alpha:
      l2 = (l2 + l1 * (1 + alpha)) / (alpha**2 + 2 * alpha + 2)
      l1 = (1 + alpha) * l2

  return l1, l2


def fsvc_watermark_embedding(signal, watermark, sr=16000, device=None):
  device = resolve_device(device)
  signal_t = to_tensor_1d(signal, device=device)
  watermark = np.asarray(watermark, dtype=np.int8)

  gamma1_scaled = gamma1 * sr / 44100
  gamma2_scaled = gamma2 * sr / 44100

  frames = torch.tensor_split(signal_t, len(watermark))
  watermarked_frames = []

  for i, frame in enumerate(frames):
    x1, x2 = torch.tensor_split(frame, 2)

    X1 = dct_ortho(x1)
    X2 = dct_ortho(x2)

    low = int(gamma1_scaled * len(frame))
    high = int(gamma2_scaled * len(frame) + 1)

    X1p = X1[low:high].unsqueeze(-1)
    X2p = X2[low:high].unsqueeze(-1)

    u1, s1, vh1 = torch.linalg.svd(X1p, full_matrices=False)
    u2, s2, vh2 = torch.linalg.svd(X2p, full_matrices=False)

    l1p, l2p = modify_svd_pair(float(s1[0]), float(s2[0]), int(watermark[i]))
    s1p = torch.tensor([l1p], dtype=signal_t.dtype, device=device)
    s2p = torch.tensor([l2p], dtype=signal_t.dtype, device=device)

    X1p_em = (u1 @ torch.diag(s1p) @ vh1).squeeze(-1)
    X2p_em = (u2 @ torch.diag(s2p) @ vh2).squeeze(-1)

    X1 = X1.clone()
    X2 = X2.clone()
    X1[low:high] = X1p_em
    X2[low:high] = X2p_em

    x1_em = idct_ortho(X1)
    x2_em = idct_ortho(X2)
    watermarked_frames.append(torch.cat([x1_em, x2_em]))

  return torch.cat(watermarked_frames).cpu().numpy()


def fsvc_watermark_detection(watermarked_signal, watermark_length=40, sr=16000, device=None):
  device = resolve_device(device)
  signal_t = to_tensor_1d(watermarked_signal, device=device)

  gamma1_scaled = gamma1 * sr / 44100
  gamma2_scaled = gamma2 * sr / 44100

  frames = torch.tensor_split(signal_t, watermark_length)
  watermark_bits = []

  for frame in frames:
    x1, x2 = torch.tensor_split(frame, 2)

    X1 = dct_ortho(x1)
    X2 = dct_ortho(x2)

    low = int(gamma1_scaled * len(frame))
    high = int(gamma2_scaled * len(frame) + 1)

    X1p = X1[low:high].unsqueeze(-1)
    X2p = X2[low:high].unsqueeze(-1)

    _, s1, _ = torch.linalg.svd(X1p, full_matrices=False)
    _, s2, _ = torch.linalg.svd(X2p, full_matrices=False)

    if float(s1[0] / s2[0]) < 1:
      watermark_bits.append(0)
    else:
      watermark_bits.append(1)

  return np.asarray(watermark_bits, dtype=np.int8)
