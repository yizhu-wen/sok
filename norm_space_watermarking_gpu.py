from __future__ import annotations

import numpy as np
import torch

from torch_watermarking_utils import (
  db1_dwt_symmetric,
  db1_idwt,
  dct_ortho,
  idct_ortho,
  resolve_device,
  to_tensor_1d,
)


def norm_space_watermark_embedding(signal, watermark, delta=0.03, device=None):
  device = resolve_device(device)
  signal_t = to_tensor_1d(signal, device=device)
  watermark = np.asarray(watermark, dtype=np.int8)

  segments = torch.tensor_split(signal_t, len(watermark))
  reconstructed_segments = []

  for index, segment in enumerate(segments):
    c_a1, c_d1 = db1_dwt_symmetric(segment)
    values = dct_ortho(c_a1)

    v1 = values[::2]
    v2 = values[1::2]

    norm_v1 = torch.linalg.norm(v1, ord=2)
    norm_v2 = torch.linalg.norm(v2, ord=2)

    u1 = v1 / norm_v1
    u2 = v2 / norm_v2

    mean_norm = (norm_v1 + norm_v2) / 2
    if int(watermark[index]) == 1:
      norm_v1 = mean_norm + delta
      norm_v2 = mean_norm - delta
    else:
      norm_v1 = mean_norm - delta
      norm_v2 = mean_norm + delta

    reconstructed_values = torch.zeros_like(values)
    reconstructed_values[::2] = norm_v1 * u1
    reconstructed_values[1::2] = norm_v2 * u2

    reconstructed_c_a1 = idct_ortho(reconstructed_values)
    reconstructed_segments.append(db1_idwt(reconstructed_c_a1, c_d1))

  return torch.cat(reconstructed_segments).cpu().numpy()


def norm_space_watermark_detection(watermarked_signal, watermark_length=512, delta=0.03, device=None):
  device = resolve_device(device)
  signal_t = to_tensor_1d(watermarked_signal, device=device)

  segments = torch.tensor_split(signal_t, watermark_length)
  watermark_bits = []

  for segment in segments:
    c_a1, _ = db1_dwt_symmetric(segment)
    values = dct_ortho(c_a1)

    norm_v1 = torch.linalg.norm(values[::2], ord=2)
    norm_v2 = torch.linalg.norm(values[1::2], ord=2)

    watermark_bits.append(1 if norm_v1 > norm_v2 else 0)

  return np.asarray(watermark_bits, dtype=np.int8)
