"""Just-Noticeable-Difference (visibility) map, values in [vmin, 1].

Combines two classic masking effects (Chou & Li, 1995 style):
  * luminance adaptation – very dark and very bright backgrounds hide more change;
  * texture masking      – busy regions (hair, eyebrows, fabric) hide more than smooth skin.

Why it matters for South Asian skin tones: smooth, darker skin in low light is exactly
where naive L-inf noise shows up as grain / colour blotches after JPEG. The map keeps
the budget low there, and `dark_boost_cap` stops the luminance term from over-spending
in deep shadows (which turns into banding after recompression).
"""

from __future__ import annotations

import cv2
import numpy as np


def visibility_map(
    rgb: np.ndarray,
    vmin: float = 0.35,
    texture_gain: float = 0.6,
    dark_boost_cap: float = 12.0,
) -> np.ndarray:
    y = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    bg = cv2.blur(y, (5, 5))
    t_lum = np.where(bg <= 127, 17.0 * (1.0 - np.sqrt(bg / 127.0)) + 3.0, 3.0 / 128.0 * (bg - 127.0) + 3.0)
    t_lum = np.minimum(t_lum, dark_boost_cap)
    mean = cv2.blur(y, (5, 5))
    std = np.sqrt(np.maximum(cv2.blur(y * y, (5, 5)) - mean * mean, 0))
    jnd = np.maximum(t_lum, texture_gain * std)
    jnd = cv2.GaussianBlur(jnd, (0, 0), 2.0)
    v = jnd / (np.percentile(jnd, 95) + 1e-6)
    return np.clip(v, vmin, 1.0).astype(np.float32)
