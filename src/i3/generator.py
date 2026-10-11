"""Phase 2: amortised immunisation ("vaccine generator").

Per-image optimisation (immunize.py) costs ~100 encoder forward/backward passes.
A small feed-forward network trained offline over many faces can predict a good
perturbation in ONE forward pass – the speed of a Universal Adversarial Perturbation,
but image-specific.

Why not a single universal mask (UAP)?
  * If every protected photo carries the *same* pattern, an attacker can estimate it by
    averaging a few protected images and subtract it.
  * A fixed pattern is also a fingerprint ("this photo went through app X").
  * One mask cannot align with each face's landmarks, so it wastes the PSNR budget.
The generator keeps the speed while staying image-specific.  At inference we use its
output as a warm start and run a few joint steps to lock in the payload
(`i3 protect --generator models/generator.pt --steps 15`).

The network is deliberately tiny (~0.3 M params, plain convs) so it can later be
exported to TorchScript / ONNX / TFLite for on-device use (see docs/ROADMAP.md).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class PerturbationGenerator(nn.Module):
    """Small U-Net: (RGB image, W_PIR mask) -> YCbCr perturbation direction in [-1, 1]."""

    def __init__(self, ch: int = 24):
        super().__init__()
        self.e1 = _block(4, ch)
        self.e2 = _block(ch, ch * 2)
        self.e3 = _block(ch * 2, ch * 4)
        self.d2 = _block(ch * 4 + ch * 2, ch * 2)
        self.d1 = _block(ch * 2 + ch, ch)
        self.out = nn.Conv2d(ch, 3, 1)

    def forward(self, x: torch.Tensor, wpir: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        ph, pw = (-h) % 4, (-w) % 4
        inp = F.pad(torch.cat([x * 2 - 1, wpir], 1), (0, pw, 0, ph), mode="replicate")
        e1 = self.e1(inp)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        d2 = self.d2(torch.cat([F.interpolate(e3, scale_factor=2, mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), e1], 1))
        return torch.tanh(self.out(d1))[..., :h, :w]


def caps_from_masks(
    wpir: torch.Tensor, vis: torch.Tensor, eps: float, chroma_ratio: float, adv_floor: float = 0.0
) -> torch.Tensor:
    """Per-pixel caps for the adversarial part (shared by immunize.py and the generator).

    W_PIR is re-scaled so its background floor maps to `adv_floor` (default 0): background
    pixels outnumber face pixels ~25:1, so even a small background cap would soak up most
    of the PSNR budget.  The JND map then shifts energy towards texture within the face.
    """
    lo = wpir.amin()
    rel = ((wpir - lo) / (1 - lo).clamp_min(1e-6)).clamp(0, 1)
    rel = adv_floor + (1 - adv_floor) * rel
    face = wpir > 0.5
    ref = vis[face].median() if face.any() else vis.median()
    shaped = rel * (vis / ref).clamp(0.5, 1.5)
    return torch.cat([shaped, shaped * chroma_ratio, shaped * chroma_ratio], dim=1) * eps


def save_generator(g: PerturbationGenerator, path: str | Path, meta: dict | None = None) -> None:
    torch.save({"state_dict": g.state_dict(), "ch": g.e1[0].out_channels, "meta": meta or {}}, path)


def load_generator(path: str | Path, device="cpu") -> PerturbationGenerator:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    g = PerturbationGenerator(ck.get("ch", 24))
    g.load_state_dict(ck["state_dict"])
    return g.to(device).eval()
