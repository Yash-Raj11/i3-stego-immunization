"""Command-line interface:  i3 <command> ...

i3 keygen                                   create keys/ (secret + Ed25519 pair)
i3 protect photo.jpg --owner you@mail.com   -> photo.protected.png + .i3.json manifest
i3 verify  received.jpg --owner you@mail.com --manifest photo.protected.i3.json
i3 evaluate --original photo.jpg --protected photo.protected.png
i3 inspect-jpeg whatsapp_download.jpg       estimate the platform's JPEG quality
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_config


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/default.yaml", help="YAML config (default: %(default)s)")
    p.add_argument("--keys", dest="keys_dir", help="key directory (default from config: keys/)")
    p.add_argument("--models", dest="models_dir", help="model directory (default: models/)")
    p.add_argument("--device", help="auto | cpu | cuda | mps")
    p.add_argument("-v", "--verbose", action="store_true")


def _apply_overrides(cfg, a) -> None:
    for k in (
        "keys_dir",
        "models_dir",
        "device",
        "surrogate",
        "max_side",
        "generator",
        "refine_surrogate",
        "refine_steps",
    ):
        v = getattr(a, k, None)
        if v is not None:
            setattr(cfg, k, v)
    if getattr(a, "refine_surrogate", None) in ("none", ""):
        cfg.refine_surrogate = None
    im = cfg.immunize
    if getattr(a, "steps", None) is not None:
        im.steps = a.steps
    if getattr(a, "eps", None) is not None:
        im.eps = a.eps / 255.0
    if getattr(a, "psnr", None) is not None:
        im.psnr_min = a.psnr
    if getattr(a, "face_psnr", None) is not None:
        im.face_psnr_min = a.face_psnr if a.face_psnr > 0 else None
    if getattr(a, "eot_samples", None) is not None:
        im.eot_samples = a.eot_samples
    if getattr(a, "mode", None) is not None:
        im.mode = a.mode
    if getattr(a, "payload_alpha", None) is not None:
        im.payload_alpha = a.payload_alpha
    if getattr(a, "face_backend", None) is not None:
        cfg.saliency.backend = a.face_backend


def _protect_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--surrogate", help="taesd | sd-vae | taesd+sd-vae")
    p.add_argument("--refine-surrogate", help="stage 2 encoder after the main steps, e.g. sd-vae ('none' disables)")
    p.add_argument("--refine-steps", type=int, help="number of stage-2 steps (default 20)")
    p.add_argument("--steps", type=int)
    p.add_argument("--eps", type=float, help="per-pixel cap in 0-255 units where W_PIR=1 (default 16)")
    p.add_argument("--psnr", type=float, help="minimum whole-image PSNR in dB (default 38)")
    p.add_argument("--face-psnr", type=float, help="minimum PSNR on the face region (default 34; 0 disables)")
    p.add_argument("--eot-samples", type=int)
    p.add_argument("--mode", choices=["targeted", "untargeted"])
    p.add_argument("--payload-alpha", type=float)
    p.add_argument("--max-side", type=int)
    p.add_argument("--generator", help="Phase-2 generator checkpoint for a fast warm start (then use ~15 steps)")
    p.add_argument("--face-backend", choices=["auto", "mediapipe", "yunet", "haar"])
    p.add_argument("--debug", action="store_true", help="also save masks and the amplified perturbation")


def _morph_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--strengths", default="0.3,0.5", help="img2img strengths, comma-separated (default 0.3,0.5)")
    p.add_argument("--no-inpaint", action="store_true", help="skip the background-inpainting edit")
    p.add_argument("--prompt", help="img2img prompt")
    p.add_argument("--scene-prompt", help="inpainting prompt for the new background")
    p.add_argument("--sd-steps", type=int, default=30, help="diffusion steps per edit (default 30)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--size", type=int, default=512, help="edit resolution, long side (default 512)")


def _morph_config(a, cfg):
    from .morph import MorphConfig

    mc = MorphConfig(
        img2img_strengths=tuple(float(x) for x in a.strengths.split(",") if x.strip()),
        inpaint=not a.no_inpaint,
        steps=a.sd_steps,
        seed=a.seed,
        size=a.size,
        device=cfg.device,
        models_dir=cfg.models_dir,
    )
    if a.prompt:
        mc.prompt = a.prompt
    if a.scene_prompt:
        mc.scene_prompt = a.scene_prompt
    return mc


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="i3", description="Imperceptible Image Immunization + steganographic payload")
    sub = ap.add_subparsers(dest="cmd", required=True)

    k = sub.add_parser("keygen", help="create a secret key and an Ed25519 signing key pair")
    _common(k)
    k.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("protect", help="immunise an image and embed the ownership payload")
    _common(p)
    p.add_argument("input")
    p.add_argument("-o", "--output", help="output path (default: <input>.protected.png)")
    p.add_argument("--owner", required=True, help="owner id (e-mail, phone, handle) - only an HMAC tag is embedded")
    _protect_opts(p)

    v = sub.add_parser("verify", help="read the payload back from a (possibly re-shared) image")
    _common(v)
    v.add_argument("image")
    v.add_argument("--owner", help="check the HMAC tag against this owner id")
    v.add_argument("--manifest", help="sidecar .i3.json to cross-check")
    v.add_argument("--pubkey", help="Ed25519 public key PEM (default: keys/ed25519_public.pem)")

    e = sub.add_parser("evaluate", help="quality, payload survival and immunity report")
    _common(e)
    e.add_argument("--original", required=True)
    e.add_argument("--protected", required=True)
    e.add_argument("--surrogate")
    e.add_argument("--max-side", type=int)
    e.add_argument("--out", default="reports/latest")

    mt = sub.add_parser("morph-test", help="run real Stable Diffusion edits on original vs protected")
    _common(mt)
    mt.add_argument("--original", required=True)
    mt.add_argument("--protected", required=True)
    mt.add_argument("--out", default="reports/morph_test")
    _morph_opts(mt)

    d = sub.add_parser("demo", help="protect an image, then run the Stable Diffusion morph test on it")
    _common(d)
    d.add_argument("input")
    d.add_argument("--owner", required=True)
    d.add_argument("--out", default="reports/demo", help="output folder (protected image + figures)")
    _protect_opts(d)
    _morph_opts(d)

    j = sub.add_parser("inspect-jpeg", help="estimate JPEG quality/size of files downloaded from a platform")
    j.add_argument("files", nargs="+")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(a, "verbose", False) else logging.INFO, format="%(levelname)s %(message)s"
    )
    if a.cmd == "inspect-jpeg":
        from .metrics import estimate_jpeg_quality

        for f in a.files:
            print(json.dumps({"file": f, **estimate_jpeg_quality(f)}, default=str))
        return 0

    cfg = load_config(a.config)
    _apply_overrides(cfg, a)

    if a.cmd == "keygen":
        from .keys import generate_keys

        d = generate_keys(cfg.keys_dir, overwrite=a.overwrite)
        print(f"keys written to {d}/  - keep i3_secret.key and ed25519_private.pem private (they are git-ignored)")
        return 0

    from . import pipeline

    if a.cmd == "protect":
        src = Path(a.input)
        dst = Path(a.output) if a.output else src.with_name(src.stem + ".protected.png")
        res = pipeline.protect_file(src, dst, a.owner, cfg, debug=a.debug)
        print(json.dumps(res, indent=2))
        return 0 if res["payload_ok"] else 2
    if a.cmd == "verify":
        res = pipeline.verify_file(a.image, cfg, a.owner, a.manifest, a.pubkey)
        print(json.dumps(res, indent=2))
        return 0 if res["payload_found"] else 1
    if a.cmd == "evaluate":
        res = pipeline.evaluate_file(a.original, a.protected, cfg, a.out)
        print(pipeline.render_markdown(res))
        print(f"full report: {a.out}/report.json")
        return 0
    if a.cmd in ("morph-test", "demo"):
        from .morph import render_markdown as morph_md
        from .morph import run_morph_test

        out = Path(a.out)
        if a.cmd == "demo":
            protected = out / (Path(a.input).stem + ".protected.png")
            res = pipeline.protect_file(a.input, protected, a.owner, cfg, debug=True)
            print(json.dumps(res, indent=2))
            original = out / f"{Path(a.input).stem}.protected_debug" / "prepared_original.png"
            if not original.exists():
                original = Path(a.input)
        else:
            original, protected = Path(a.original), Path(a.protected)
        rep = run_morph_test(pipeline.load_rgb(original), pipeline.load_rgb(protected), out, _morph_config(a, cfg))
        print(morph_md(rep))
        print(f"figures: {out}/before_after.png  {out}/grid.png")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
