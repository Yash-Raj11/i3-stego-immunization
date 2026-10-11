from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "assets" / "sample_astronaut.png"
MODELS = ROOT / "models"


@pytest.fixture(scope="session")
def sample_rgb() -> np.ndarray:
    return np.asarray(Image.open(SAMPLE).convert("RGB"))


@pytest.fixture(scope="session")
def face_crop(sample_rgb) -> np.ndarray:
    """256x256 crop around the face - small enough for fast tests."""
    return np.ascontiguousarray(sample_rgb[40:296, 100:356])


def to_t(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(rgb)).permute(2, 0, 1)[None].float() / 255


def to_pil(t: torch.Tensor) -> Image.Image:
    return Image.fromarray((t[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy())


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)
