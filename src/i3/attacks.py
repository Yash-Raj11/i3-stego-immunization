"""Real (non-differentiable) image manipulations used for *evaluation*.

These use actual encoders (Pillow's libjpeg / libwebp), so they are the ground truth
the differentiable EOT layer is trying to approximate.

Platform presets are APPROXIMATIONS. Platforms change their pipelines without notice.
Calibrate them yourself: send a test image to yourself on WhatsApp/Instagram/Telegram,
download it back, and run `i3 inspect-jpeg file.jpg` to read off the real size and
estimated JPEG quality. Then edit PLATFORM_PRESETS below (or configs/default.yaml).
"""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

PLATFORM_PRESETS = {
    # name: (max_long_side, format, quality)
    "whatsapp_like": (1600, "JPEG", 72),
    "whatsapp_hd_like": (4096, "JPEG", 80),
    "instagram_like": (1080, "JPEG", 78),
    "telegram_like": (1280, "JPEG", 82),
}


def jpeg(img: Image.Image, q: int, subsampling: int | str = 2) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=int(q), subsampling=subsampling)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def webp(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "WEBP", quality=int(q))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def limit_long_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    s = max_side / max(w, h)
    if s >= 1:
        return img
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.LANCZOS)


def platform(img: Image.Image, name: str) -> Image.Image:
    side, fmt, q = PLATFORM_PRESETS[name]
    out = limit_long_side(img, side)
    return jpeg(out, q) if fmt == "JPEG" else webp(out, q)


def rescale(img: Image.Image, s: float) -> Image.Image:
    w, h = img.size
    return img.resize((max(8, round(w * s)), max(8, round(h * s))), Image.Resampling.BICUBIC)


def down_up(img: Image.Image, s: float = 0.5) -> Image.Image:
    """Classic purification attempt: downscale then upscale back."""
    return rescale(img, s).resize(img.size, Image.Resampling.BICUBIC)


def beauty_filter(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(1.08)
    img = ImageEnhance.Contrast(img).enhance(1.1)
    img = ImageEnhance.Color(img).enhance(1.2)
    return img.filter(ImageFilter.GaussianBlur(0.6))  # "skin smoothing"


def center_crop(img: Image.Image, frac: float) -> Image.Image:
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def gaussian_noise(img: Image.Image, std: float = 4.0, seed: int = 0) -> Image.Image:
    a = np.asarray(img, dtype=np.float32)
    a = a + np.random.default_rng(seed).normal(0, std, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


ATTACKS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "identity": lambda im: im,
    "jpeg_q90": lambda im: jpeg(im, 90),
    "jpeg_q75": lambda im: jpeg(im, 75),
    "jpeg_q60": lambda im: jpeg(im, 60),
    "jpeg_q50": lambda im: jpeg(im, 50),
    "webp_q75": lambda im: webp(im, 75),
    "whatsapp_like": lambda im: platform(im, "whatsapp_like"),
    "instagram_like": lambda im: platform(im, "instagram_like"),
    "telegram_like": lambda im: platform(im, "telegram_like"),
    "resize_0.5": lambda im: rescale(im, 0.5),
    "resize_0.5_jpeg75": lambda im: jpeg(rescale(im, 0.5), 75),
    "beauty_filter_jpeg80": lambda im: jpeg(beauty_filter(im), 80),
    "blur_1.0": lambda im: im.filter(ImageFilter.GaussianBlur(1.0)),
    "median_3": lambda im: im.filter(ImageFilter.MedianFilter(3)),
    "noise_std4": lambda im: gaussian_noise(im, 4.0),
    "down_up_0.5": lambda im: down_up(im, 0.5),
    "double_whatsapp": lambda im: platform(platform(im, "whatsapp_like"), "whatsapp_like"),
    "crop_90": lambda im: center_crop(im, 0.9),  # expected to FAIL for the payload (no sync yet)
}

# Attacks an adversary would use to *strip* adversarial noise before editing.
PURIFICATION = ["jpeg_q75", "blur_1.0", "median_3", "down_up_0.5", "noise_std4", "resize_0.5_jpeg75"]
