import math
from typing import Optional, Union

import torch


SQRT2 = math.sqrt(2.0)


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
  if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

  resolved = torch.device(device)
  if resolved.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

  return resolved


def to_tensor_1d(
  signal,
  device: Optional[Union[str, torch.device]] = None,
  dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
  return torch.as_tensor(signal, dtype=dtype, device=resolve_device(device)).contiguous()


def dct_ortho(x: torch.Tensor) -> torch.Tensor:
  original_shape = x.shape
  n = original_shape[-1]
  flat = x.contiguous().reshape(-1, n)

  v = torch.cat([flat[:, ::2], flat[:, 1::2].flip([1])], dim=1)
  vc = torch.fft.fft(v, dim=1)

  k = -torch.arange(n, device=x.device, dtype=x.dtype).unsqueeze(0) * math.pi / (2 * n)
  wr = torch.cos(k)
  wi = torch.sin(k)

  transformed = vc[:, :n].real * wr - vc[:, :n].imag * wi
  transformed[:, 0] /= math.sqrt(n) * 2
  if n > 1:
    transformed[:, 1:] /= math.sqrt(n / 2) * 2

  return (2 * transformed).reshape(original_shape)


def idct_ortho(x: torch.Tensor) -> torch.Tensor:
  original_shape = x.shape
  n = original_shape[-1]
  flat = x.contiguous().reshape(-1, n) / 2

  flat[:, 0] *= math.sqrt(n) * 2
  if n > 1:
    flat[:, 1:] *= math.sqrt(n / 2) * 2

  k = torch.arange(n, device=x.device, dtype=x.dtype).unsqueeze(0) * math.pi / (2 * n)
  wr = torch.cos(k)
  wi = torch.sin(k)

  vt_real = flat
  vt_imag = torch.cat([torch.zeros_like(flat[:, :1]), -flat.flip([1])[:, :-1]], dim=1)

  vr = vt_real * wr - vt_imag * wi
  vi = vt_real * wi + vt_imag * wr

  v = torch.complex(vr, vi)
  values = torch.fft.ifft(v, dim=1).real

  reconstructed = torch.zeros_like(values)
  reconstructed[:, ::2] = values[:, : n - (n // 2)]
  reconstructed[:, 1::2] = values.flip([1])[:, : n // 2]
  return reconstructed.reshape(original_shape)


def db1_dwt_symmetric(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  if x.shape[-1] % 2 == 1:
    x = torch.cat([x, x[-1:]], dim=0)

  even = x[::2]
  odd = x[1::2]
  c_a = (even + odd) / SQRT2
  c_d = (even - odd) / SQRT2
  return c_a, c_d


def db1_idwt(c_a: torch.Tensor, c_d: torch.Tensor) -> torch.Tensor:
  even = (c_a + c_d) / SQRT2
  odd = (c_a - c_d) / SQRT2

  reconstructed = torch.empty(c_a.shape[0] * 2, dtype=c_a.dtype, device=c_a.device)
  reconstructed[::2] = even
  reconstructed[1::2] = odd
  return reconstructed
