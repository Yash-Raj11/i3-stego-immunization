"""Saliency, optimiser constraints, generator, CLI end-to-end (fast: random surrogate)."""

import json

import numpy as np
import pytest
import torch

from i3 import payload as P
from i3.cli import main
from i3.generator import PerturbationGenerator, caps_from_masks, load_generator, save_generator
from i3.immunize import ImmunizeConfig, immunize
from i3.jnd import visibility_map
from i3.metrics import estimate_jpeg_quality, psnr
from i3.saliency import build_wpir, detect_faces, normalize_illumination
from i3.stego import SpreadSpectrumCodec
from i3.surrogates import get_surrogate

from .conftest import MODELS, SAMPLE, to_pil, to_t


def test_face_detection_and_wpir(sample_rgb):
    faces = detect_faces(sample_rgb, "auto", MODELS)
    assert len(faces) == 1
    x, y, w, h = faces[0].box
    assert 150 < x + w / 2 < 260 and 60 < y + h / 2 < 200  # the astronaut's face
    m = build_wpir(sample_rgb, faces)
    assert m.dtype == np.float32 and m.min() >= 0.1 - 1e-6 and m.max() <= 1 + 1e-6
    assert m[y + h // 2, x + w // 2] > 0.4 > m[10, 10]  # face weighted above background


def test_low_light_detection(sample_rgb):
    dark = (255 * ((sample_rgb / 255.0) ** 1.8) * 0.3).astype(np.uint8)
    assert normalize_illumination(dark).mean() > dark.mean()
    assert len(detect_faces(dark, "auto", MODELS)) == 1


def test_visibility_map_range(sample_rgb):
    v = visibility_map(sample_rgb)
    assert v.shape == sample_rgb.shape[:2] and 0.3 <= v.min() and v.max() <= 1.0


@pytest.mark.parametrize("mode", ["untargeted", "targeted"])
def test_immunize_respects_constraints(face_crop, mode):
    rgb = face_crop[:128, 64:192].copy()
    x = to_t(rgb)
    secret = b"s" * 32
    codec = SpreadSpectrumCodec(secret)
    bits = P.encode_bits(P.make_payload(secret, "me", timestamp=1))
    w = torch.from_numpy(build_wpir(rgb, detect_faces(rgb, "auto", MODELS)))[None, None]
    v = torch.from_numpy(visibility_map(rgb))[None, None]
    cfg = ImmunizeConfig(steps=6, mode=mode, psnr_min=38.0, face_psnr_min=33.0)
    r = immunize(x, get_surrogate("random"), codec, bits, w, v, cfg, progress=False)
    assert r.psnr >= 38.0
    assert r.face_psnr is None or r.face_psnr >= 33.0
    assert r.payload_ok
    q = r.protected
    assert q.min() >= 0 and q.max() <= 1 and torch.allclose(q * 255, (q * 255).round(), atol=1e-4)
    assert len(r.history) == 6


def test_generator_shapes_and_roundtrip(tmp_path, face_crop):
    g = PerturbationGenerator(ch=8)
    x = to_t(face_crop[:100, :90])  # odd size on purpose
    w = torch.rand(1, 1, 100, 90)
    out = g(x, w)
    assert out.shape == (1, 3, 100, 90) and out.abs().max() <= 1
    cap = caps_from_masks(w, torch.rand(1, 1, 100, 90), 16 / 255, 0.25)
    assert (cap[:, 1:] <= cap[:, :1] * 0.25 + 1e-7).all()
    save_generator(g, tmp_path / "g.pt")
    assert torch.allclose(load_generator(tmp_path / "g.pt")(x, w), out, atol=1e-6)


def test_jpeg_quality_estimator(tmp_path, face_crop):
    p = tmp_path / "q.jpg"
    to_pil(to_t(face_crop)).save(p, quality=72)
    info = estimate_jpeg_quality(p)
    assert info["quality_estimate"] == 72 and info["subsampling"] == "4:2:0"


def test_cli_end_to_end(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    keys = tmp_path / "keys"
    assert main(["keygen", "--keys", str(keys)]) == 0
    out = tmp_path / "p.png"
    rc = main(
        [
            "protect",
            str(SAMPLE),
            "-o",
            str(out),
            "--owner",
            "me@example.com",
            "--keys",
            str(keys),
            "--models",
            str(MODELS),
            "--surrogate",
            "random",
            "--steps",
            "3",
            "--max-side",
            "512",
            "--debug",
        ]
    )
    assert rc == 0 and out.exists() and out.with_suffix(".i3.json").exists()
    capsys.readouterr()
    rc = main(
        [
            "verify",
            str(out),
            "--owner",
            "me@example.com",
            "--keys",
            str(keys),
            "--manifest",
            str(out.with_suffix(".i3.json")),
        ]
    )
    res = json.loads(capsys.readouterr().out)
    assert rc == 0 and res["payload_found"] and res["owner_match"]
    assert res["manifest_signature_valid"] and res["manifest_payload_match"]
    from PIL import Image

    from i3 import attacks

    shared = tmp_path / "shared.jpg"
    attacks.platform(Image.open(out), "whatsapp_like").save(shared, quality=95)
    assert main(["verify", str(shared), "--owner", "someone@else.com", "--keys", str(keys)]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["payload_found"] and res["owner_match"] is False
    assert (
        psnr(np.asarray(Image.open(out)), np.asarray(Image.open(tmp_path / "p_debug" / "prepared_original.png"))) >= 38
    )


@pytest.mark.slow
@pytest.mark.skipif(not (MODELS / "taesd_encoder.pth").exists(), reason="run scripts/download_models.py")
def test_taesd_surrogate_loads(face_crop):
    s = get_surrogate("taesd", MODELS)
    with torch.no_grad():
        z = s.encode(to_t(face_crop))
        r = s.decode(z)
    assert z.shape == (1, 4, 32, 32) and r.shape == (1, 3, 256, 256)
    assert psnr(face_crop, to_pil(r)) > 20  # TAESD reconstructs a clean face reasonably
