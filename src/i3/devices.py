"""Device-portability helpers (Apple MPS, CUDA, CPU)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pool_to(mask: torch.Tensor, size) -> torch.Tensor:
    """Adaptive average pooling of a (no-grad) mask to `size`, safe on every device.

    Apple MPS only implements adaptive pooling when the input size is divisible by the
    output size (pytorch/pytorch#96056). Masks never need gradients, so we fall back to
    the CPU for that case and move the result back.
    """
    size = tuple(int(s) for s in size)
    h, w = mask.shape[-2:]
    if mask.device.type == "mps" and (h % size[0] or w % size[1]):
        return F.adaptive_avg_pool2d(mask.detach().cpu(), size).to(mask.device)
    return F.adaptive_avg_pool2d(mask, size)
