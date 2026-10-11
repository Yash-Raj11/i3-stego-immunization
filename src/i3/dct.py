"""Blockwise 8x8 orthonormal DCT (differentiable, pure torch)."""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn.functional as F


@lru_cache(maxsize=4)
def _dct_matrix(n: int = 8) -> torch.Tensor:
    m = torch.zeros(n, n, dtype=torch.float64)
    for k in range(n):
        a = math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)
        for i in range(n):
            m[k, i] = a * math.cos(math.pi * (2 * i + 1) * k / (2 * n))
    return m.float()


def dct_matrix(n: int = 8, like: torch.Tensor | None = None) -> torch.Tensor:
    m = _dct_matrix(n)
    return m.to(like) if like is not None else m


def blockify(x: torch.Tensor, n: int = 8) -> torch.Tensor:
    """(B,C,H,W) -> (B,C,H/n,W/n,n,n). H and W must be multiples of n."""
    b, c, h, w = x.shape
    x = x.view(b, c, h // n, n, w // n, n)
    return x.permute(0, 1, 2, 4, 3, 5)


def unblockify(x: torch.Tensor) -> torch.Tensor:
    b, c, hb, wb, n, _ = x.shape
    return x.permute(0, 1, 2, 4, 3, 5).reshape(b, c, hb * n, wb * n)


def dct2_blocks(x: torch.Tensor, n: int = 8) -> torch.Tensor:
    d = dct_matrix(n, x)
    return d @ blockify(x, n) @ d.T


def idct2_blocks(coef: torch.Tensor) -> torch.Tensor:
    n = coef.shape[-1]
    d = dct_matrix(n, coef)
    return unblockify(d.T @ coef @ d)


def pad_to_multiple(x: torch.Tensor, m: int) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return x, (h, w)


@lru_cache(maxsize=2)
def zigzag_order(n: int = 8) -> tuple[int, ...]:
    """Flat indices (row*n+col) of an n x n block in JPEG zig-zag order."""
    order = sorted(
        ((r, c) for r in range(n) for c in range(n)),
        key=lambda rc: (rc[0] + rc[1], rc[0] if (rc[0] + rc[1]) % 2 else -rc[0]),
    )
    return tuple(r * n + c for r, c in order)
