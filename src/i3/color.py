"""Colour-space helpers (BT.601 full-range, the same matrix JPEG uses).

All tensors are (B, 3, H, W) in [0, 1] unless stated otherwise.
"""

from __future__ import annotations

import torch

_RGB2YCC = torch.tensor(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ]
)
_YCC2RGB = torch.linalg.inv(_RGB2YCC)
_OFFSET = torch.tensor([0.0, 0.5, 0.5])


def rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    m = _RGB2YCC.to(x)
    out = torch.einsum("ij,bjhw->bihw", m, x)
    return out + _OFFSET.to(x).view(1, 3, 1, 1)


def ycbcr_to_rgb(x: torch.Tensor) -> torch.Tensor:
    m = _YCC2RGB.to(x)
    return torch.einsum("ij,bjhw->bihw", m, x - _OFFSET.to(x).view(1, 3, 1, 1))


def ycbcr_delta_to_rgb(d: torch.Tensor) -> torch.Tensor:
    """Map a *difference* in YCbCr space to a difference in RGB space (no offset)."""
    m = _YCC2RGB.to(d)
    return torch.einsum("ij,bjhw->bihw", m, d)


def luma(x: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) RGB -> (B,1,H,W) luma."""
    w = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)
