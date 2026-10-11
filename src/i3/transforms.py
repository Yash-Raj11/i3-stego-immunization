"""Expectation-over-Transformation (EOT) layer.

Every optimisation step samples a random, *differentiable* "social-media pipeline"
(resize -> beauty-filter colour shift -> blur -> noise -> JPEG -> 8-bit quantise)
and back-propagates through it. A perturbation that wins on average over this
distribution survives the real thing much better than one tuned on the clean image.

Two outputs per sample:
  * `common` – what the payload decoder sees (no cropping: the decoder is not
               crop-invariant yet, see docs/ROADMAP.md).
  * `adv`    – `common` plus a random crop, fed to the surrogate encoder.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .color import luma
from .diff_jpeg import diff_jpeg, ste_round


@dataclass
class EOTConfig:
    p_jpeg: float = 0.8
    jpeg_quality: tuple[float, float] = (60.0, 90.0)  # WhatsApp/Instagram live around 70-85
    p_resize: float = 0.6
    resize_scale: tuple[float, float] = (0.5, 1.0)
    p_blur: float = 0.3
    blur_sigma: tuple[float, float] = (0.3, 1.0)
    p_color: float = 0.5
    brightness: float = 0.08
    contrast: float = 0.10
    saturation: float = 0.15
    gamma: tuple[float, float] = (0.9, 1.1)
    p_noise: float = 0.3
    noise_std: tuple[float, float] = (0.0, 0.01)
    p_crop: float = 0.5
    crop_frac: tuple[float, float] = (0.8, 1.0)
    quantize: bool = True

    @classmethod
    def none(cls) -> EOTConfig:
        return cls(p_jpeg=0, p_resize=0, p_blur=0, p_color=0, p_noise=0, p_crop=0, quantize=False)


@dataclass
class EOTSample:
    common: torch.Tensor
    adv: torch.Tensor
    params: dict = field(default_factory=dict)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    radius = max(1, int(math.ceil(3 * sigma)))
    k = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(k**2) / (2 * sigma**2))
    k = k / k.sum()
    c = x.shape[1]
    kx = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, kx, groups=c)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(x, ky, groups=c)


def resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    down = size[0] < x.shape[-2] or size[1] < x.shape[-1]
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=down)


def color_filter(x: torch.Tensor, b: float, c: float, s: float, g: float) -> torch.Tensor:
    """A crude, differentiable 'beauty filter': brightness, contrast, saturation, gamma."""
    x = x + b
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * (1 + c) + mean
    y = luma(x)
    x = y + (x - y) * (1 + s)
    return x.clamp(1e-4, 1.0) ** g


class EOT:
    """Draw random transform parameters, then apply them (replayable on another image)."""

    def __init__(self, cfg: EOTConfig | None = None, seed: int = 0):
        self.cfg = cfg or EOTConfig()
        self.rng = random.Random(seed)

    def _u(self, lo_hi: tuple[float, float]) -> float:
        return self.rng.uniform(*lo_hi)

    def draw(self, h: int, w: int) -> dict:
        c, r = self.cfg, self.rng
        p: dict = {}
        if r.random() < c.p_resize:
            s = self._u(c.resize_scale)
            p["resize"] = (max(32, int(round(h * s))), max(32, int(round(w * s))))
            h, w = p["resize"]
        if r.random() < c.p_color:
            p["color"] = (
                r.uniform(-c.brightness, c.brightness),
                r.uniform(-c.contrast, c.contrast),
                r.uniform(-c.saturation, c.saturation),
                self._u(c.gamma),
            )
        if r.random() < c.p_blur:
            p["blur"] = self._u(c.blur_sigma)
        if r.random() < c.p_noise:
            p["noise"] = (self._u(c.noise_std), r.randrange(2**31))
        if r.random() < c.p_jpeg:
            p["jpeg"] = self._u(c.jpeg_quality)
        if r.random() < c.p_crop:
            f = self._u(c.crop_frac)
            ch, cw = max(32, int(h * f)), max(32, int(w * f))
            p["crop"] = (r.randint(0, h - ch), r.randint(0, w - cw), ch, cw)
        return p

    def apply(self, x: torch.Tensor, p: dict) -> EOTSample:
        y = x
        if "resize" in p:
            y = resize(y, p["resize"])
        if "color" in p:
            y = color_filter(y, *p["color"])
        if "blur" in p:
            y = gaussian_blur(y, p["blur"])
        if "noise" in p:
            std, seed = p["noise"]
            g = torch.Generator(device="cpu").manual_seed(seed)
            y = y + torch.randn(y.shape, generator=g).to(y) * std
        y = y.clamp(0, 1)
        if "jpeg" in p:
            y = diff_jpeg(y, p["jpeg"])
        elif self.cfg.quantize:
            y = ste_round(y * 255.0) / 255.0
        adv = y
        if "crop" in p:
            top, left, ch, cw = p["crop"]
            adv = y[..., top : top + ch, left : left + cw]
        return EOTSample(common=y, adv=adv, params=p)

    @staticmethod
    def apply_geometry(mask: torch.Tensor, p: dict) -> torch.Tensor:
        """Apply only the geometric part (resize, crop) of `p` to a (1,1,H,W) mask."""
        if "resize" in p:
            mask = F.interpolate(mask, size=p["resize"], mode="bilinear", align_corners=False)
        if "crop" in p:
            top, left, ch, cw = p["crop"]
            mask = mask[..., top : top + ch, left : left + cw]
        return mask

    def sample(self, x: torch.Tensor) -> EOTSample:
        return self.apply(x, self.draw(*x.shape[-2:]))
