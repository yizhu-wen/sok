from __future__ import annotations

import numpy as np
import torch

from torch_watermarking_utils import dct_ortho, idct_ortho, resolve_device, to_tensor_1d


fs = 3000
fe = 7000
k1 = 0.195
k2 = 0.08


def patchwork_multilayer_watermark_embedding(signal, watermark, sr=16000, device=None):
  device = resolve_device(device)
  signal_t = to_tensor_1d(signal, device=device)
  watermark = np.asarray(watermark, dtype=np.int8)

  signal_length = len(signal_t)
  start_index = int(fs / (sr / signal_length))
  end_index = int(fe / (sr / signal_length))

  spectrum = dct_ortho(signal_t)
  selected = spectrum[start_index : (end_index + 1)]
  selected_length = len(selected)

  if selected_length % (len(watermark) * 2) != 0:
    selected_length -= selected_length % (len(watermark) * 2)
    selected = selected[:selected_length]

  interleaved = torch.stack(
    [selected[: selected_length // 2], selected[selected_length // 2 :].flip(0)], dim=1
  ).reshape(-1)

  segments = torch.tensor_split(interleaved, len(watermark) * 2)
  watermarked_segments = []

  for i in range(0, len(segments), 2):
    j = i // 2 + 1
    rj = k1 * np.exp(-k2 * j)

    mean1 = torch.mean(torch.abs(segments[i]))
    mean2 = torch.mean(torch.abs(segments[i + 1]))
    mean_avg = (mean1 + mean2) / 2
    mean_min = torch.minimum(mean1, mean2)

    mean1_prime = mean1
    mean2_prime = mean2

    if int(watermark[j - 1]) == 0 and float(mean1 - mean2) < float(rj * mean_min):
      mean1_prime = mean_avg + (rj * mean_min / 2)
      mean2_prime = mean_avg - (rj * mean_min / 2)
    elif int(watermark[j - 1]) == 1 and float(mean2 - mean1) < float(rj * mean_min):
      mean1_prime = mean_avg - (rj * mean_min / 2)
      mean2_prime = mean_avg + (rj * mean_min / 2)

    watermarked_segments.append(segments[i] * mean1_prime / mean1)
    watermarked_segments.append(segments[i + 1] * mean2_prime / mean2)

  interleaved_watermarked = torch.cat(watermarked_segments)
  restored = torch.cat([interleaved_watermarked[::2], interleaved_watermarked.flip(0)[::2]])

  watermarked_spectrum = spectrum.clone()
  watermarked_spectrum[start_index : (start_index + selected_length)] = restored
  return idct_ortho(watermarked_spectrum).cpu().numpy()


def patchwork_multilayer_watermark_detection(
  watermarked_signal,
  watermark_length=40,
  sr=16000,
  device=None,
):
  device = resolve_device(device)
  signal_t = to_tensor_1d(watermarked_signal, device=device)

  signal_length = len(signal_t)
  start_index = int(fs / (sr / signal_length))
  end_index = int(fe / (sr / signal_length))

  spectrum = dct_ortho(signal_t)
  selected = spectrum[start_index : (end_index + 1)]
  selected_length = len(selected)

  if selected_length % (watermark_length * 2) != 0:
    selected_length -= selected_length % (watermark_length * 2)
    selected = selected[:selected_length]

  interleaved = torch.stack(
    [selected[: selected_length // 2], selected[selected_length // 2 :].flip(0)], dim=1
  ).reshape(-1)

  segments = torch.tensor_split(interleaved, watermark_length * 2)
  watermark_bits = []

  for i in range(0, len(segments), 2):
    mean1 = torch.mean(torch.abs(segments[i]))
    mean2 = torch.mean(torch.abs(segments[i + 1]))
    watermark_bits.append(0 if float(mean1 - mean2) >= 0 else 1)

  return np.asarray(watermark_bits, dtype=np.int8)
