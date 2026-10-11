"""Image-quality and robustness metrics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from .diff_jpeg import CHROMA_Q, LUMA_Q, scaled_table


def to_np(img) -> np.ndarray:
    return np.asarray(img.convert("RGB") if isinstance(img, Image.Image) else img, dtype=np.float64)


def psnr(a, b) -> float:
    mse = np.mean((to_np(a) - to_np(b)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255.0**2 / mse)


def _gauss_filter(x: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    import cv2

    return cv2.GaussianBlur(x, (11, 11), sigma)


def ssim(a, b) -> float:
    """Mean SSIM on luma (Wang et al. 2004, gaussian window)."""
    w = np.array([0.299, 0.587, 0.114])
    x, y = to_np(a) @ w, to_np(b) @ w
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = _gauss_filter(x), _gauss_filter(y)
    sxx = _gauss_filter(x * x) - mx**2
    syy = _gauss_filter(y * y) - my**2
    sxy = _gauss_filter(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2))
    return float(s.mean())


def bit_accuracy(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.asarray(a) == np.asarray(b)))


def estimate_jpeg_quality(path: str | Path) -> dict:
    """Estimate the IJG quality factor of a JPEG from its quantisation tables.

    Use this to calibrate the EOT layer against what WhatsApp/Instagram actually do:
    send yourself a photo, save the received file, run `i3 inspect-jpeg`.
    """
    img = Image.open(path)
    info: dict = {"format": img.format, "size": img.size, "mode": img.mode}
    q = getattr(img, "quantization", None)
    if img.format != "JPEG" or not q:
        info["quality_estimate"] = None
        return info
    luma_tab = np.array(q[0], dtype=np.float64).reshape(8, 8)
    # Pillow returns tables in natural (row-major) order
    best, best_err = None, float("inf")
    for qual in range(1, 101):
        err = np.abs(scaled_table(LUMA_Q, qual).numpy() - luma_tab).mean()
        if 1 in q:
            err += np.abs(scaled_table(CHROMA_Q, qual).numpy() - np.array(q[1]).reshape(8, 8)).mean()
        if err < best_err:
            best, best_err = qual, err
    info["quality_estimate"] = best
    info["table_mismatch"] = round(best_err, 3)  # >1 means non-IJG (custom) tables
    try:
        from PIL import JpegImagePlugin

        info["subsampling"] = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}.get(JpegImagePlugin.get_sampling(img), "unknown")
    except Exception:  # pragma: no cover
        pass
    info["has_exif"] = bool(img.info.get("exif"))
    return info
