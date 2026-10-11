"""High-level operations used by the CLI: protect, verify, evaluate."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from . import attacks as atk
from . import manifest as mf
from . import metrics as M
from . import payload as P
from .config import AppConfig
from .devices import pool_to
from .immunize import immunize
from .jnd import visibility_map
from .keys import load_private_key, load_public_key, load_secret
from .saliency import build_wpir, detect_faces
from .stego import SpreadSpectrumCodec
from .surrogates import get_surrogate, members

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- io
def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_rgb(path: str | Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def prepare(img: Image.Image, max_side: int) -> np.ndarray:
    """Downscale to max_side and centre-crop to a multiple of 8 (needed by the VAE)."""
    img = atk.limit_long_side(img, max_side)
    w, h = img.size
    nw, nh = w - w % 8, h - h % 8
    left, top = (w - nw) // 2, (h - nh) // 2
    return np.asarray(img.crop((left, top, left + nw, top + nh)), dtype=np.uint8)


def to_tensor(rgb: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.array(rgb, dtype=np.uint8)).permute(2, 0, 1)[None].float().div(255).to(device)


def to_uint8(x: torch.Tensor) -> np.ndarray:
    return (x[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()


def save_image(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(rgb)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        log.warning("Saving as JPEG adds a lossy step before the platform does - PNG is recommended.")
        im.save(path, quality=95, subsampling=0)
    else:
        im.save(path)


def make_codec(cfg: AppConfig, secret: bytes, device) -> SpreadSpectrumCodec:
    return SpreadSpectrumCodec(secret, cfg.stego, device=device)


# --------------------------------------------------------------------------- protect
def protect_file(src: str | Path, dst: str | Path, owner_id: str, cfg: AppConfig, debug: bool = False) -> dict:
    src, dst = Path(src), Path(dst)
    device = pick_device(cfg.device)
    secret = load_secret(cfg.keys_dir)
    rgb = prepare(load_rgb(src), cfg.max_side)
    if min(rgb.shape[:2]) < 1.5 * cfg.stego.canonical:
        log.warning(
            "image short side %d px < 1.5x payload grid (%d): payload will be fragile under JPEG. "
            "Use larger images or stego.canonical: 128.",
            min(rgb.shape[:2]),
            cfg.stego.canonical,
        )
    faces = detect_faces(
        rgb, cfg.saliency.backend, cfg.models_dir, illumination_normalize=cfg.saliency.illumination_normalize
    )
    wpir = build_wpir(rgb, faces, floor=cfg.saliency.floor, edge_mix=cfg.saliency.edge_mix)
    vis = visibility_map(rgb, vmin=cfg.saliency.vis_min, dark_boost_cap=cfg.saliency.dark_boost_cap)
    pl = P.make_payload(secret, owner_id)
    bits = P.encode_bits(pl)
    codec = make_codec(cfg, secret, device)
    surrogate = get_surrogate(cfg.surrogate, cfg.models_dir, device)
    log.info(
        "image %sx%s | faces: %s | surrogate: %s | device: %s",
        rgb.shape[1],
        rgb.shape[0],
        [f.source for f in faces] or "none",
        surrogate.name,
        device,
    )
    x = to_tensor(rgb, device)
    wpir_t = torch.from_numpy(wpir)[None, None].to(device)
    vis_t = torch.from_numpy(vis)[None, None].to(device)
    init_adv = None
    if cfg.generator:
        from .generator import caps_from_masks, load_generator

        gen = load_generator(cfg.generator, device)
        with torch.no_grad():
            init_adv = gen(x, wpir_t) * caps_from_masks(wpir_t, vis_t, cfg.immunize.eps, cfg.immunize.chroma_ratio)
        log.info("generator warm start from %s", cfg.generator)
    res = immunize(x, surrogate, codec, bits, wpir_t, vis_t, cfg.immunize, init_adv=init_adv)
    out = to_uint8(res.protected)
    save_image(out, dst)
    params = {
        "surrogate": surrogate.name,
        "mode": cfg.immunize.mode,
        "eps_255": round(cfg.immunize.eps * 255, 2),
        "steps": cfg.immunize.steps,
        "eot_samples": cfg.immunize.eot_samples,
        "psnr_min": cfg.immunize.psnr_min,
        "face_psnr_min": cfg.immunize.face_psnr_min,
        "payload_alpha": cfg.immunize.payload_alpha,
        "stego": asdict(cfg.stego),
        "generator": cfg.generator,
        "faces": [f.source for f in faces],
        "result": res.summary(),
    }
    m = mf.build_manifest(owner_id, pl.hex, pl.timestamp, src, dst, params)
    m = mf.sign_manifest(m, load_private_key(cfg.keys_dir))
    man_path = dst.with_suffix(".i3.json")
    mf.save(m, man_path)
    if debug:
        dbg = dst.parent / f"{dst.stem}_debug"
        dbg.mkdir(exist_ok=True)
        Image.fromarray((wpir * 255).astype(np.uint8)).save(dbg / "w_pir.png")
        Image.fromarray((vis * 255).astype(np.uint8)).save(dbg / "visibility.png")
        diff = np.clip((out.astype(np.int16) - rgb.astype(np.int16)) * 10 + 128, 0, 255).astype(np.uint8)
        Image.fromarray(diff).save(dbg / "perturbation_x10.png")
        Image.fromarray(rgb).save(dbg / "prepared_original.png")
        (dbg / "history.json").write_text(json.dumps(res.history))
    return {"output": str(dst), "manifest": str(man_path), "payload": pl.hex, "timestamp": pl.iso_time, **res.summary()}


# --------------------------------------------------------------------------- verify
def verify_file(
    path: str | Path,
    cfg: AppConfig,
    owner_id: str | None = None,
    manifest_path: str | Path | None = None,
    pubkey: str | Path | None = None,
) -> dict:
    secret = load_secret(cfg.keys_dir)
    img = load_rgb(path)
    codec = make_codec(cfg, secret, "cpu")
    bits, z = codec.decode(to_tensor(np.asarray(img), "cpu"))
    pl, nfix = P.decode_bits(bits, z)
    out: dict = {
        "file": str(path),
        "payload_found": pl is not None,
        "mean_abs_z": round(float(np.abs(z).mean()), 2),
    }
    if pl is not None:
        out.update(payload=pl.hex, timestamp=pl.iso_time, corrected_bytes=nfix)
        if owner_id is not None:
            out["owner_match"] = P.verify_owner(pl, secret, owner_id)
    if manifest_path:
        m = mf.load(manifest_path)
        key = load_public_key(pubkey or Path(cfg.keys_dir) / "ed25519_public.pem")
        out["manifest_signature_valid"] = mf.verify_manifest(m, key)
        out["manifest_payload_match"] = pl is not None and m.get("payload_hex") == pl.hex
        out["dhash_distance"] = mf.hamming_hex(mf.dhash(img), m["protected"]["dhash"])
    return out


# --------------------------------------------------------------------------- evaluate
def _label(im: Image.Image, text: str) -> Image.Image:
    im = im.copy()
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 18], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 255))
    return im


def _region_psnr(a: Image.Image, b: Image.Image, region: np.ndarray | None) -> float:
    if region is None or not region.any():
        return round(M.psnr(a, b), 2)
    da = np.asarray(a, np.float64)[region]
    db = np.asarray(b, np.float64)[region]
    mse = np.mean((da - db) ** 2)
    return float("inf") if mse == 0 else round(10 * np.log10(255.0**2 / mse), 2)


def evaluate_file(original: str | Path, protected: str | Path, cfg: AppConfig, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)
    prot = load_rgb(protected)
    orig = Image.fromarray(prepare(load_rgb(original), cfg.max_side))
    if orig.size != prot.size:
        orig = orig.resize(prot.size, Image.Resampling.LANCZOS)
    orig_np = np.asarray(orig)
    faces = detect_faces(orig_np, cfg.saliency.backend, cfg.models_dir)
    region = build_wpir(orig_np, faces, floor=cfg.saliency.floor) > 0.5 if faces else None
    secret = load_secret(cfg.keys_dir)
    codec = make_codec(cfg, secret, "cpu")
    ref_bits, _ = codec.decode(to_tensor(np.asarray(prot), "cpu"))
    ref_pl, _ = P.decode_bits(ref_bits)
    report: dict = {
        "quality": {
            "psnr_db": round(M.psnr(orig, prot), 2),
            "face_psnr_db": _region_psnr(orig, prot, region),
            "ssim": round(M.ssim(orig, prot), 4),
        },
        "faces": [f.source for f in faces],
        "payload_reference_ok": ref_pl is not None,
        "payload_survival": {},
        "immunity": {},
    }
    for name, fn in atk.ATTACKS.items():
        a = fn(prot)
        b, z = codec.decode(to_tensor(np.asarray(a), "cpu"))
        pl, _ = P.decode_bits(b, z)
        report["payload_survival"][name] = {
            "bit_acc": round(M.bit_accuracy(b, ref_bits), 3),
            "decoded": pl is not None and ref_pl is not None and pl.hex == ref_pl.hex,
        }

    surrogate = get_surrogate(cfg.surrogate, cfg.models_dir, device)
    grid_rows = []
    for name in ["identity", "whatsapp_like"] + atk.PURIFICATION:
        fn = atk.ATTACKS[name]
        ao, ap = fn(orig), fn(prot)
        if ap.size != orig.size:
            ao, ap = ao.resize(orig.size, Image.Resampling.BICUBIC), ap.resize(orig.size, Image.Resampling.BICUBIC)
        xo, xp = to_tensor(np.asarray(ao), device), to_tensor(np.asarray(ap), device)
        row = {}
        with torch.no_grad():
            for mm in members(surrogate):
                zo, zp = mm.encode(xo), mm.encode(xp)
                if region is not None:
                    w = pool_to(torch.from_numpy(region.astype(np.float32))[None, None].to(device), zo.shape[-2:])
                    shift = (((zp - zo) ** 2 * w).sum() / ((zo**2) * w).sum()).sqrt().item()
                else:
                    shift = ((zp - zo).norm() / zo.norm()).item()
                r = {"latent_shift": round(shift, 3)}
                ro, rp = mm.decode(zo), mm.decode(zp)
                if ro is not None:
                    io_, ip_ = Image.fromarray(to_uint8(ro)), Image.fromarray(to_uint8(rp))
                    r["recon_psnr_original"] = _region_psnr(orig, io_, region)
                    r["recon_psnr_protected"] = _region_psnr(orig, ip_, region)
                    r["recon_drop_db"] = round(r["recon_psnr_original"] - r["recon_psnr_protected"], 2)
                    if mm is members(surrogate)[0] and name in ("identity", "whatsapp_like", "down_up_0.5"):
                        grid_rows.append((name, ao, ap, Image.fromarray(to_uint8(ro)), Image.fromarray(to_uint8(rp))))
                row[mm.name] = r
        report["immunity"][name] = row

    # comparison grid
    if grid_rows:
        tw = 256
        th = round(tw * orig.height / orig.width)
        diff = np.clip((np.asarray(prot, np.int16) - np.asarray(orig, np.int16)) * 10 + 128, 0, 255).astype(np.uint8)
        tiles = [
            [
                _label(orig.resize((tw, th)), "original"),
                _label(prot.resize((tw, th)), f"protected {report['quality']['psnr_db']} dB"),
                _label(Image.fromarray(diff).resize((tw, th)), "perturbation x10"),
            ]
        ]
        for name, _ao, ap, ro, rp in grid_rows:
            tiles.append(
                [
                    _label(ap.resize((tw, th)), f"protected -> {name}"),
                    _label(ro.resize((tw, th)), f"VAE recon of original ({name})"),
                    _label(rp.resize((tw, th)), f"VAE recon of protected ({name})"),
                ]
            )
        grid = Image.new("RGB", (3 * tw, len(tiles) * th), "white")
        for r, row in enumerate(tiles):
            for c, t in enumerate(row):
                grid.paste(t, (c * tw, r * th))
        grid.save(out_dir / "comparison.png")
        report["comparison_png"] = str(out_dir / "comparison.png")

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "report.md").write_text(render_markdown(report))
    return report


def render_markdown(r: dict) -> str:
    q = r["quality"]
    lines = [
        "# I3 evaluation report",
        "",
        f"- PSNR (whole image): **{q['psnr_db']} dB**  ·  PSNR (face region): **{q['face_psnr_db']} dB**"
        f"  ·  SSIM: **{q['ssim']}**",
        f"- faces detected with: {', '.join(r.get('faces') or ['none'])}",
        "",
    ]
    lines += ["## Payload survival", "", "| attack | bit accuracy | payload decoded |", "|---|---|---|"]
    for k, v in r["payload_survival"].items():
        lines.append(f"| {k} | {v['bit_acc']:.3f} | {'yes' if v['decoded'] else 'no'} |")
    lines += [
        "",
        "## Immunity (surrogate encoder, measured on the face region)",
        "",
        "latent_shift = ||E(attacked protected) - E(attacked original)|| / ||E(attacked original)||;",
        "recon_drop_db = how much worse the VAE reconstruction of the protected face is (higher = stronger).",
        "Purification rows show what survives when an attacker tries to wash the perturbation out.",
        "",
    ]
    lines += ["| attack | surrogate | latent shift | recon drop (dB) |", "|---|---|---|---|"]
    for k, row in r["immunity"].items():
        for s, v in row.items():
            lines.append(f"| {k} | {s} | {v['latent_shift']} | {v.get('recon_drop_db', '-')} |")
    return "\n".join(lines) + "\n"
