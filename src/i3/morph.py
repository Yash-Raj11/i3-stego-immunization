"""End-to-end Stable Diffusion "morph test": does immunisation actually spoil real edits?

For one original photo and its protected version we run the same Stable Diffusion edits
with the same seed and prompt, then compare:

  * img2img        - re-renders the whole photo (strengths 0.3 / 0.5 by default). This is
                     how most "AI filter" and morphing bots work.
  * inpaint-scene  - keeps the person, regenerates the background from a text prompt
                     (the classic "put this person in another scene" misuse).

Outputs (in --out):
  before_after.png   2x2 hero figure for slides: original vs protected, edited vs edited
  grid.png           every edit: inputs, edited original, edited protected + metrics
  edits/*.png        every individual output image
  report.json / .md  metrics

Metrics are measured on the face region (from W_PIR) because that is what we protect:
  face_psnr_vs_input  similarity of the edited face to the input face (higher = edit kept
                      the identity; we want this LOWER for the protected photo)
  face_ssim_vs_input  same, structural similarity
  face_score          YuNet face-detector confidence on the edited image (0 = no face found)

Requires `pip install -e ".[sd]"`. First run downloads two SD-1.5 checkpoints (~2 GB each)
into the Hugging Face cache. Runs on CUDA, Apple MPS or (slowly) CPU.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import metrics as M
from .saliency import YUNET_MODEL, build_wpir, detect_faces

log = logging.getLogger(__name__)

IMG2IMG_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
INPAINT_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
DEFAULT_PROMPT = "a portrait photo of a person, high quality, detailed face"
DEFAULT_SCENE_PROMPT = "a photo of a person standing in a busy market street, daylight, high quality"
NEGATIVE = "blurry, low quality"

# An edit function takes (image, mask_or_None) and returns the edited image.
EditFn = Callable[[Image.Image, Image.Image | None], Image.Image]


@dataclass
class EditSpec:
    name: str
    kind: str  # "img2img" | "inpaint"
    strength: float
    prompt: str


@dataclass
class MorphConfig:
    img2img_strengths: tuple[float, ...] = (0.3, 0.5)
    inpaint: bool = True
    prompt: str = DEFAULT_PROMPT
    scene_prompt: str = DEFAULT_SCENE_PROMPT
    steps: int = 30
    guidance: float = 7.5
    seed: int = 1234
    size: int = 512
    device: str = "auto"
    half: bool = False  # fp16 weights on MPS/CPU too (half the memory); CUDA always uses fp16
    models_dir: str = "models"
    img2img_model: str = IMG2IMG_MODEL
    inpaint_model: str = INPAINT_MODEL
    edits: list[EditSpec] = field(default_factory=list)

    def build_edits(self) -> list[EditSpec]:
        if self.edits:
            return self.edits
        out = [EditSpec(f"img2img_s{s:.1f}", "img2img", s, self.prompt) for s in self.img2img_strengths]
        if self.inpaint:
            out.append(EditSpec("inpaint_scene", "inpaint", 1.0, self.scene_prompt))
        return out


# --------------------------------------------------------------------------- masks & metrics
def face_region(rgb: np.ndarray, models_dir: str | Path) -> tuple[np.ndarray | None, list]:
    faces = detect_faces(rgb, "auto", models_dir)
    if not faces:
        return None, []
    return build_wpir(rgb, faces) > 0.5, faces


def scene_mask(rgb: np.ndarray, faces: list, grow: float = 1.6) -> Image.Image:
    """White = regenerate. Keeps an enlarged ellipse around each face (head + hair) untouched."""
    h, w = rgb.shape[:2]
    keep = np.zeros((h, w), np.uint8)
    for f in faces:
        x, y, fw, fh = f.box
        cx, cy = x + fw / 2, y + fh / 2
        cv2.ellipse(
            keep, (int(cx), int(cy + 0.15 * fh)), (int(fw * grow / 2), int(fh * grow * 0.65)), 0, 0, 360, 255, -1
        )
    keep = cv2.GaussianBlur(keep, (0, 0), 3)
    return Image.fromarray(255 - keep)


def _region_psnr(a: np.ndarray, b: np.ndarray, region: np.ndarray | None) -> float:
    if region is None or not region.any():
        return round(M.psnr(a, b), 2)
    mse = np.mean((a.astype(np.float64)[region] - b.astype(np.float64)[region]) ** 2)
    return float("inf") if mse == 0 else round(10 * np.log10(255.0**2 / mse), 2)


def _region_ssim(a: np.ndarray, b: np.ndarray, region: np.ndarray | None) -> float:
    if region is None or not region.any():
        return round(M.ssim(a, b), 4)
    ys, xs = np.nonzero(region)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return round(M.ssim(a[y0:y1, x0:x1], b[y0:y1, x0:x1]), 4)


def face_score(rgb: np.ndarray, models_dir: str | Path) -> float:
    """Best YuNet confidence (0 if no face). Uses a low threshold so we see weak faces too."""
    path = Path(models_dir) / YUNET_MODEL
    if not path.exists():
        return float("nan")
    h, w = rgb.shape[:2]
    det = cv2.FaceDetectorYN.create(str(path), "", (w, h), 0.3, 0.3, 50)
    det.setInputSize((w, h))
    _, rows = det.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return 0.0 if rows is None else round(float(rows[:, -1].max()), 3)


# --------------------------------------------------------------------------- pipelines
def _device_and_dtype(name: str, half: bool = False):
    import torch

    from .pipeline import pick_device

    dev = pick_device(name)
    # fp16 on CUDA. On MPS fp32 is the safe default; --low-mem/--half switches to fp16 (half the
    # memory; on some setups this can give black images - run_morph_test warns if it happens).
    use_half = dev.type == "cuda" or (half and dev.type == "mps")
    return dev, (torch.float16 if use_half else torch.float32)


def free_device_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _load(cls, repo: str, dev, dtype):
    kw = dict(torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
    try:
        pipe = cls.from_pretrained(repo, variant="fp16", **kw)  # smaller download
    except (OSError, ValueError):
        pipe = cls.from_pretrained(repo, **kw)
    pipe = pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)
    if dev.type != "cuda":
        pipe.enable_attention_slicing()  # lower peak memory on Apple Silicon / CPU
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
    return pipe


def sd_edit_fn(cfg: MorphConfig, spec_kind: str) -> tuple[Callable[[EditSpec], EditFn], Callable[[], None]]:
    """Return (factory(spec) -> edit function, release()) for one pipeline type."""
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline

    dev, dtype = _device_and_dtype(cfg.device, cfg.half)
    if spec_kind == "img2img":
        pipe = _load(StableDiffusionImg2ImgPipeline, cfg.img2img_model, dev, dtype)
    else:
        pipe = _load(StableDiffusionInpaintPipeline, cfg.inpaint_model, dev, dtype)

    def factory(spec: EditSpec) -> EditFn:
        def run(img: Image.Image, mask: Image.Image | None) -> Image.Image:
            g = torch.Generator("cpu").manual_seed(cfg.seed)  # same noise for both images
            common = dict(
                prompt=spec.prompt,
                negative_prompt=NEGATIVE,
                num_inference_steps=cfg.steps,
                guidance_scale=cfg.guidance,
                generator=g,
            )
            if spec.kind == "img2img":
                return pipe(image=img, strength=spec.strength, **common).images[0]
            return pipe(
                image=img, mask_image=mask, height=img.height, width=img.width, strength=spec.strength, **common
            ).images[0]

        return run

    def release():
        nonlocal pipe
        del pipe
        free_device_memory()

    return factory, release


# --------------------------------------------------------------------------- main routine
def _fit(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(64, round(w * s / 8) * 8), max(64, round(h * s / 8) * 8)
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def run_morph_test(
    original: Image.Image,
    protected: Image.Image,
    out_dir: str | Path,
    cfg: MorphConfig | None = None,
    edit_backend: Callable[[MorphConfig, str], tuple] | None = None,
) -> dict:
    """Run every edit on both images. `edit_backend` is injectable for tests (default: diffusers)."""
    cfg = cfg or MorphConfig()
    backend = edit_backend or sd_edit_fn
    out_dir = Path(out_dir)
    (out_dir / "edits").mkdir(parents=True, exist_ok=True)

    prot = _fit(protected.convert("RGB"), cfg.size)
    orig = original.convert("RGB").resize(prot.size, Image.Resampling.LANCZOS)
    o_np, p_np = np.asarray(orig), np.asarray(prot)
    region, faces = face_region(o_np, cfg.models_dir)
    mask = scene_mask(o_np, faces) if faces else None
    specs = cfg.build_edits()
    if mask is None:
        specs = [s for s in specs if s.kind != "inpaint"]
        log.warning("no face found - skipping the inpainting edit")
    orig.save(out_dir / "edits" / "input_original.png")
    prot.save(out_dir / "edits" / "input_protected.png")
    if mask is not None:
        mask.save(out_dir / "edits" / "inpaint_mask.png")

    report: dict = {
        "config": {k: v for k, v in cfg.__dict__.items() if k != "edits"},
        "input": {
            "size": list(prot.size),
            "face_psnr_protected_vs_original": _region_psnr(o_np, p_np, region),
            "psnr_protected_vs_original": round(M.psnr(o_np, p_np), 2),
        },
        "edits": {},
    }
    images: dict[str, tuple[Image.Image, Image.Image]] = {}
    for kind in ("img2img", "inpaint"):
        kind_specs = [s for s in specs if s.kind == kind]
        if not kind_specs:
            continue
        t0 = time.time()
        log.info("loading %s pipeline...", kind)
        factory, release = backend(cfg, kind)
        try:
            for spec in kind_specs:
                run = factory(spec)
                log.info("edit %s: original...", spec.name)
                eo = run(orig, mask).resize(orig.size)
                log.info("edit %s: protected...", spec.name)
                ep = run(prot, mask).resize(orig.size)
                eo.save(out_dir / "edits" / f"{spec.name}_original.png")
                ep.save(out_dir / "edits" / f"{spec.name}_protected.png")
                eo_np, ep_np = np.asarray(eo), np.asarray(ep)
                if min(eo_np.mean(), ep_np.mean()) < 3:
                    log.warning(
                        "edit %s produced a (near) black image - if you used --low-mem/--half, rerun without it",
                        spec.name,
                    )
                r = {
                    "kind": spec.kind,
                    "strength": spec.strength,
                    "prompt": spec.prompt,
                    "original": {
                        "face_psnr_vs_input": _region_psnr(o_np, eo_np, region),
                        "face_ssim_vs_input": _region_ssim(o_np, eo_np, region),
                        "face_score": face_score(eo_np, cfg.models_dir),
                    },
                    "protected": {
                        "face_psnr_vs_input": _region_psnr(o_np, ep_np, region),
                        "face_ssim_vs_input": _region_ssim(o_np, ep_np, region),
                        "face_score": face_score(ep_np, cfg.models_dir),
                    },
                    "face_psnr_between_edits": _region_psnr(eo_np, ep_np, region),
                }
                r["face_psnr_drop_db"] = round(
                    r["original"]["face_psnr_vs_input"] - r["protected"]["face_psnr_vs_input"], 2
                )
                report["edits"][spec.name] = r
                images[spec.name] = (eo, ep)
        finally:
            release()
        log.info("%s done in %.0fs", kind, time.time() - t0)

    if images:
        make_grid(orig, prot, images, report, out_dir / "grid.png")
        best = max(report["edits"], key=lambda k: report["edits"][k]["face_psnr_drop_db"])
        make_before_after(orig, prot, *images[best], best, report, out_dir / "before_after.png")
        report["hero_edit"] = best
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    (out_dir / "report.md").write_text(render_markdown(report))
    return report


# --------------------------------------------------------------------------- figures
def _caption(im: Image.Image, text: str, sub: str = "", bar: int = 34) -> Image.Image:
    canvas = Image.new("RGB", (im.width, im.height + bar), "white")
    canvas.paste(im, (0, bar))
    d = ImageDraw.Draw(canvas)
    d.text((6, 4), text, fill=(0, 0, 0))
    if sub:
        d.text((6, 18), sub, fill=(90, 90, 90))
    return canvas


def make_grid(orig, prot, images: dict, report: dict, path: Path, tile: int = 256) -> None:
    def t(im):
        return im.resize((tile, round(tile * im.height / im.width)), Image.Resampling.LANCZOS)

    rows = []
    for name, (eo, ep) in images.items():
        r = report["edits"][name]
        o, p = r["original"], r["protected"]
        rows.append(
            [
                _caption(t(orig), "input: original"),
                _caption(
                    t(prot), "input: protected", f"face PSNR {report['input']['face_psnr_protected_vs_original']} dB"
                ),
                _caption(t(eo), f"{name} of ORIGINAL", f"face PSNR {o['face_psnr_vs_input']}  det {o['face_score']}"),
                _caption(t(ep), f"{name} of PROTECTED", f"face PSNR {p['face_psnr_vs_input']}  det {p['face_score']}"),
            ]
        )
    w, h = rows[0][0].size
    grid = Image.new("RGB", (4 * w, len(rows) * h), "white")
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            grid.paste(im, (j * w, i * h))
    grid.save(path)


def make_before_after(orig, prot, eo, ep, name, report, path: Path, tile: int = 384) -> None:
    def t(im):
        return im.resize((tile, round(tile * im.height / im.width)), Image.Resampling.LANCZOS)

    r = report["edits"][name]
    cells = [
        _caption(t(orig), "BEFORE  - original photo"),
        _caption(t(eo), f"AI edit of original ({name})", "edit works: face preserved"),
        _caption(
            t(prot),
            "AFTER  - immunised photo",
            f"looks the same (face PSNR {report['input']['face_psnr_protected_vs_original']} dB)",
        ),
        _caption(t(ep), f"AI edit of immunised ({name})", f"face PSNR drop {r['face_psnr_drop_db']} dB"),
    ]
    w, h = cells[0].size
    fig = Image.new("RGB", (2 * w, 2 * h), "white")
    for i, c in enumerate(cells):
        fig.paste(c, ((i % 2) * w, (i // 2) * h))
    fig.save(path)


def render_markdown(r: dict) -> str:
    lines = [
        "# Stable Diffusion morph test",
        "",
        f"- input size {r['input']['size']}, protected vs original: PSNR {r['input']['psnr_protected_vs_original']} dB,"
        f" face PSNR {r['input']['face_psnr_protected_vs_original']} dB",
        "- same prompt, seed and settings for both images; metrics on the face region of the ORIGINAL input",
        "- face PSNR vs input: higher = the edit kept the face. A large drop for the protected photo = immunisation works.",
        "- det: YuNet face-detector confidence on the edited image (0 = no face found).",
        "",
        "| edit | face PSNR (orig → edit) | face PSNR (protected → edit) | drop dB | SSIM orig / prot | det orig / prot |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in r["edits"].items():
        o, p = v["original"], v["protected"]
        lines.append(
            f"| {k} | {o['face_psnr_vs_input']} | {p['face_psnr_vs_input']} | **{v['face_psnr_drop_db']}** | "
            f"{o['face_ssim_vs_input']} / {p['face_ssim_vs_input']} | {o['face_score']} / {p['face_score']} |"
        )
    if "hero_edit" in r:
        lines += ["", f"Hero figure (`before_after.png`) uses: **{r['hero_edit']}**"]
    return "\n".join(lines) + "\n"
