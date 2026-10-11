"""Morph-test harness with a fake diffusion backend (no downloads, no GPU)."""

import json

import numpy as np
from PIL import Image, ImageFilter

from i3.cli import build_parser
from i3.morph import MorphConfig, run_morph_test, scene_mask

from .conftest import MODELS, SAMPLE

MARK = (1, 2, 3)  # top-left pixel marks the "protected" input for the fake backend


def fake_backend(cfg, kind):
    """img2img = blur by strength; inpaint = grey background. The protected input gets extra
    damage, mimicking an immunised photo, so the drop metric must come out positive."""

    def factory(spec):
        def run(img, mask):
            is_protected = img.getpixel((0, 0)) == MARK
            out = img.filter(ImageFilter.GaussianBlur(4 * spec.strength))
            if mask is not None:
                out = Image.composite(Image.new("RGB", img.size, (128, 128, 128)), out, mask.convert("L"))
            if is_protected:
                out = out.filter(ImageFilter.GaussianBlur(6))
            return out

        return run

    return factory, lambda: None


def test_morph_test_produces_figures_and_metrics(tmp_path):
    orig = Image.open(SAMPLE).convert("RGB")
    prot = orig.copy()
    prot.putpixel((0, 0), MARK)
    cfg = MorphConfig(img2img_strengths=(0.3,), steps=1, models_dir=str(MODELS), size=512)
    rep = run_morph_test(orig, prot, tmp_path, cfg, edit_backend=fake_backend)

    assert set(rep["edits"]) == {"img2img_s0.3", "inpaint_scene"}
    for r in rep["edits"].values():
        assert r["face_psnr_drop_db"] > 0  # protected edits are worse
    for f in ("before_after.png", "grid.png", "report.json", "report.md", "edits/inpaint_mask.png"):
        assert (tmp_path / f).exists(), f
    assert json.loads((tmp_path / "report.json").read_text())["hero_edit"] in rep["edits"]
    assert Image.open(tmp_path / "before_after.png").size[0] == 2 * 384


def test_scene_mask_keeps_the_face(sample_rgb):
    from i3.saliency import detect_faces

    faces = detect_faces(sample_rgb, "auto", MODELS)
    mask = np.asarray(scene_mask(sample_rgb, faces))
    x, y, w, h = faces[0].box
    assert mask[y + h // 2, x + w // 2] < 10  # face centre: keep
    assert mask[5, 5] > 245  # corner: regenerate


def test_cli_parses_demo_and_morph_test():
    p = build_parser()
    a = p.parse_args(["demo", "x.png", "--owner", "me", "--strengths", "0.2,0.4", "--no-inpaint", "--steps", "5"])
    assert a.cmd == "demo" and a.strengths == "0.2,0.4" and a.no_inpaint and a.steps == 5
    a = p.parse_args(["morph-test", "--original", "a.png", "--protected", "b.png"])
    assert a.out == "reports/morph_test" and a.sd_steps == 30
