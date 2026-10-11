"""Differentiable JPEG approximation (used inside the EOT layer).

Pipeline: RGB -> YCbCr -> 4:2:0 chroma subsampling -> 8x8 DCT -> quantise with
IJG tables scaled for `quality` -> dequantise -> IDCT -> upsample -> RGB.

Rounding uses a straight-through estimator (forward = real rounding, backward =
identity), which is enough for EOT-style optimisation. It is an *approximation*
of libjpeg: real platforms (WhatsApp, Instagram) also resize, use different
chroma upsampling and sometimes custom tables – evaluate with `i3.attacks`
(real PIL encoders) and calibrate with `i3 inspect-jpeg`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .color import rgb_to_ycbcr, ycbcr_to_rgb
from .dct import dct2_blocks, idct2_blocks, pad_to_multiple

# Standard IJG / Annex K tables (row-major).
LUMA_Q = torch.tensor(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=torch.float32,
)
CHROMA_Q = torch.tensor(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=torch.float32,
)


def quality_scale(quality: float) -> float:
    q = float(min(max(quality, 1.0), 100.0))
    return 5000.0 / q if q < 50 else 200.0 - 2.0 * q


def scaled_table(base: torch.Tensor, quality: float) -> torch.Tensor:
    s = quality_scale(quality)
    return torch.clamp(torch.floor((base * s + 50.0) / 100.0), 1.0, 255.0)


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


def _quantise(channel: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    coef = dct2_blocks(channel)
    t = table.to(coef)
    return idct2_blocks(ste_round(coef / t) * t)


def diff_jpeg(x: torch.Tensor, quality: float = 75.0, subsample: bool = True) -> torch.Tensor:
    """x: (B,3,H,W) in [0,1]. Returns the JPEG-approximated image, same shape."""
    x16, (h, w) = pad_to_multiple(x, 16)
    ycc = rgb_to_ycbcr(x16) * 255.0 - 128.0
    y, cb, cr = ycc[:, :1], ycc[:, 1:2], ycc[:, 2:3]
    if subsample:
        cb, cr = F.avg_pool2d(cb, 2), F.avg_pool2d(cr, 2)
    lt, ct = scaled_table(LUMA_Q, quality), scaled_table(CHROMA_Q, quality)
    y = _quantise(y, lt)
    cb = _quantise(cb, ct)
    cr = _quantise(cr, ct)
    if subsample:
        size = y.shape[-2:]
        cb = F.interpolate(cb, size=size, mode="bilinear", align_corners=False)
        cr = F.interpolate(cr, size=size, mode="bilinear", align_corners=False)
    out = ycbcr_to_rgb((torch.cat([y, cb, cr], dim=1) + 128.0) / 255.0)
    out = ste_round(out.clamp(0, 1) * 255.0) / 255.0
    return out[..., :h, :w]
