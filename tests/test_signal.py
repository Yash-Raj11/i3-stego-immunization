"""DCT, differentiable JPEG and EOT layer."""

import numpy as np
import torch

from i3 import attacks
from i3.dct import dct2_blocks, idct2_blocks, zigzag_order
from i3.diff_jpeg import diff_jpeg
from i3.metrics import psnr
from i3.transforms import EOT, EOTConfig

from .conftest import to_pil, to_t


def test_dct_roundtrip():
    x = torch.rand(2, 1, 32, 48)
    assert torch.allclose(idct2_blocks(dct2_blocks(x)), x, atol=1e-5)


def test_zigzag_starts_like_jpeg():
    assert zigzag_order(8)[:6] == (0, 1, 8, 16, 9, 2)


def test_diff_jpeg_matches_libjpeg(face_crop):
    x = to_t(face_crop)
    for q in (50, 75, 90):
        ours = to_pil(diff_jpeg(x, q))
        real = attacks.jpeg(to_pil(x), q)
        # same algorithm up to chroma-upsampling details -> very close
        assert psnr(ours, real) > 33, q


def test_diff_jpeg_has_gradient(face_crop):
    x = to_t(face_crop).requires_grad_(True)
    diff_jpeg(x, 70).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_eot_replay_is_deterministic(face_crop):
    x = to_t(face_crop)
    eot = EOT(EOTConfig(p_jpeg=1, p_resize=1, p_noise=1, p_crop=1, p_color=1, p_blur=1), seed=3)
    p = eot.draw(*x.shape[-2:])
    a, b = eot.apply(x, p), eot.apply(x.clone(), p)
    assert torch.equal(a.common, b.common) and torch.equal(a.adv, b.adv)
    m = EOT.apply_geometry(torch.ones(1, 1, *x.shape[-2:]), p)
    assert m.shape[-2:] == a.adv.shape[-2:]


def test_eot_none_is_identity(face_crop):
    x = to_t(face_crop)
    s = EOT(EOTConfig.none()).sample(x)
    assert torch.equal(s.common, x)


def test_attacks_return_images(face_crop):
    im = to_pil(to_t(face_crop))
    for name, fn in attacks.ATTACKS.items():
        out = fn(im)
        assert out.mode == "RGB" and min(out.size) > 0, name
    assert np.asarray(attacks.platform(im, "whatsapp_like")).dtype == np.uint8
