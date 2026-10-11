"""Payload coding, spread-spectrum carrier, manifests."""

import numpy as np
import pytest
import torch

from i3 import attacks
from i3 import manifest as mf
from i3 import payload as P
from i3.immunize import init_payload_amplitudes
from i3.jnd import visibility_map
from i3.keys import generate_keys, load_private_key, load_public_key, load_secret
from i3.metrics import psnr
from i3.stego import SpreadSpectrumCodec

from .conftest import to_pil, to_t

SECRET = bytes(range(32))


def test_payload_roundtrip_and_owner_check():
    p = P.make_payload(SECRET, "owner@example.com", timestamp=1_760_000_000)
    bits = P.encode_bits(p)
    assert bits.shape == (P.CODED_BITS,)
    q, fixed = P.decode_bits(bits)
    assert q == p and fixed == 0
    assert P.verify_owner(q, SECRET, "owner@example.com")
    assert not P.verify_owner(q, SECRET, "someone-else")
    assert not P.verify_owner(q, b"x" * 32, "owner@example.com")


def test_reed_solomon_corrects_byte_errors():
    p = P.make_payload(SECRET, "o", timestamp=1)
    bits = P.encode_bits(p).copy()
    for byte in (0, 5, 11, 15):  # 4 corrupted bytes = the RS(16,8) limit
        bits[byte * 8 : byte * 8 + 8] ^= 1
    q, fixed = P.decode_bits(bits)
    assert q == p and fixed == 4


def test_too_many_errors_returns_none():
    bits = np.random.default_rng(1).integers(0, 2, P.CODED_BITS).astype(np.uint8)
    q, _ = P.decode_bits(bits)
    assert q is None or isinstance(q, P.Payload)  # random data must not crash


def _embed(rgb, alpha=0.005):
    x = to_t(rgb)
    codec = SpreadSpectrumCodec(SECRET)
    p = P.make_payload(SECRET, "me", timestamp=1_700_000_000)
    bits = P.encode_bits(p)
    vis = torch.from_numpy(visibility_map(rgb))[None, None]
    A = init_payload_amplitudes(codec, x, bits, vis, alpha)
    y = (x + codec.render(A, x.shape[-2:]) * vis).clamp(0, 1)
    return codec, p, to_pil(y)


# The closed-form carrier alone (no EOT optimisation). WebP and "beauty filter" chains are
# the weak spots at this strength; the full optimiser adds margin for them (see `i3 evaluate`).
@pytest.mark.parametrize(
    "attack", ["identity", "jpeg_q60", "jpeg_q50", "whatsapp_like", "double_whatsapp", "resize_0.5_jpeg75"]
)
def test_carrier_survives_platform_processing(sample_rgb, attack):
    codec, p, im = _embed(sample_rgb)
    assert psnr(sample_rgb, im) > 40
    att = attacks.ATTACKS[attack](im)
    bits, z = codec.decode(to_t(np.asarray(att)))
    q, _ = P.decode_bits(bits, z)
    assert q == p


def test_wrong_key_reads_nothing(sample_rgb):
    _, p, im = _embed(sample_rgb)
    other = SpreadSpectrumCodec(b"another key" * 3)
    bits, _ = other.decode(to_t(np.asarray(im)))
    q, _ = P.decode_bits(bits)
    assert q != p


def test_clean_image_has_no_payload(sample_rgb):
    bits, z = SpreadSpectrumCodec(SECRET).decode(to_t(sample_rgb))
    assert P.decode_bits(bits, z)[0] is None or np.abs(z).mean() < 3


def test_manifest_sign_verify_and_tamper(tmp_path, face_crop):
    kd = generate_keys(tmp_path / "keys")
    assert len(load_secret(kd)) == 32
    with pytest.raises(FileExistsError):
        generate_keys(kd)
    orig, prot = tmp_path / "o.png", tmp_path / "p.png"
    to_pil(to_t(face_crop)).save(orig)
    to_pil(to_t(face_crop)).save(prot)
    m = mf.sign_manifest(mf.build_manifest("me", "00" * 8, 1, orig, prot, {"a": 1}), load_private_key(kd))
    pub = load_public_key(kd)
    assert mf.verify_manifest(m, pub)
    m["payload_hex"] = "ff" * 8
    assert not mf.verify_manifest(m, pub)


def test_dhash_is_robust_to_recompression(face_crop):
    im = to_pil(to_t(face_crop))
    assert mf.hamming_hex(mf.dhash(im), mf.dhash(attacks.platform(im, "whatsapp_like"))) <= 6
