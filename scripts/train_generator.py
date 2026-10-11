"""Train the Phase-2 amortised "vaccine" generator.

    python scripts/train_generator.py --data data/faces --epochs 20 --out models/generator.pt
    i3 protect photo.jpg --owner me --generator models/generator.pt --steps 15

Data: a folder of face photos (jpg/png, any size). Use images you have the right to
use: your own consented photos, FairFace (has an "Indian" group - useful for the
skin-tone audit), FFHQ (check licences). Keep data/ out of git (.gitignore does this).

Loss per batch  = untargeted, W_PIR-weighted latent loss through EOT
                + lam_psnr * relu(MSE - MSE_budget)   (soft version of the PSNR constraint)
Training on a GPU (Colab / Kaggle free tier) is strongly recommended.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from i3.color import ycbcr_delta_to_rgb  # noqa: E402
from i3.generator import PerturbationGenerator, caps_from_masks, save_generator  # noqa: E402
from i3.immunize import latent_loss  # noqa: E402
from i3.jnd import visibility_map  # noqa: E402
from i3.pipeline import pick_device  # noqa: E402
from i3.saliency import build_wpir, detect_faces  # noqa: E402
from i3.surrogates import get_surrogate  # noqa: E402
from i3.transforms import EOT, EOTConfig  # noqa: E402

log = logging.getLogger("train")
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_items(
    folder: Path, size: int, models_dir: str, limit: int | None
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Pre-compute (image, W_PIR, visibility) at `size` x `size` (centre crop)."""
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS)[: limit or None]
    if not files:
        raise SystemExit(f"no images found in {folder}")
    items = []
    for i, f in enumerate(files):
        im = ImageOps.fit(ImageOps.exif_transpose(Image.open(f)).convert("RGB"), (size, size), Image.Resampling.LANCZOS)
        rgb = np.asarray(im, dtype=np.uint8)
        faces = detect_faces(rgb, "auto", models_dir)
        items.append((rgb, build_wpir(rgb, faces), visibility_map(rgb)))
        if (i + 1) % 50 == 0:
            log.info("prepared %d/%d", i + 1, len(files))
    return items


def batches(items, bs: int, rng: random.Random):
    idx = list(range(len(items)))
    rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [items[j] for j in idx[i : i + bs]]
        x = torch.from_numpy(np.stack([c[0] for c in chunk])).permute(0, 3, 1, 2).float() / 255
        w = torch.from_numpy(np.stack([c[1] for c in chunk]))[:, None]
        v = torch.from_numpy(np.stack([c[2] for c in chunk]))[:, None]
        yield x, w, v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/generator.pt")
    ap.add_argument("--surrogate", default="taesd")
    ap.add_argument("--models", default="models")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eps", type=float, default=16, help="0-255 units, same meaning as i3 protect --eps")
    ap.add_argument("--psnr", type=float, default=38.0)
    ap.add_argument("--lam-psnr", type=float, default=50.0)
    ap.add_argument("--limit", type=int, help="use only the first N images (smoke tests)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = pick_device(a.device)

    items = load_items(Path(a.data), a.size, a.models, a.limit)
    log.info("%d training images, device=%s", len(items), dev)
    surr = get_surrogate(a.surrogate, a.models, dev)
    gen = PerturbationGenerator().to(dev)
    opt = torch.optim.Adam(gen.parameters(), lr=a.lr)
    eot = EOT(EOTConfig(), seed=a.seed)
    mse_budget = 10 ** (-a.psnr / 10)
    eps = a.eps / 255

    for ep in range(a.epochs):
        t0, tot, n = time.time(), 0.0, 0
        for x, w, v in batches(items, a.batch, rng):
            x, w, v = x.to(dev), w.to(dev), v.to(dev)
            cap = torch.cat([caps_from_masks(w[i : i + 1], v[i : i + 1], eps, 0.15) for i in range(len(x))])
            d = ycbcr_delta_to_rgb(gen(x, w) * cap)
            xp = (x + d).clamp(0, 1)
            params = eot.draw(*x.shape[-2:])
            s = eot.apply(xp, params)
            ref = eot.apply(x, params).adv
            l_adv = latent_loss(surr, s.adv, EOT.apply_geometry(w, params), "untargeted", ref)
            mse = (d**2).mean(dim=(1, 2, 3))
            l_vis = torch.relu(mse / mse_budget - 1).mean()
            loss = l_adv + a.lam_psnr * l_vis
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, n = tot + loss.item(), n + 1
        psnr_est = 10 * math.log10(1 / max(mse.mean().item(), 1e-12))
        log.info(
            "epoch %d/%d  loss %.4f  batch-PSNR %.1f dB  (%.0fs)",
            ep + 1,
            a.epochs,
            tot / max(n, 1),
            psnr_est,
            time.time() - t0,
        )
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        save_generator(gen, a.out, {"epochs": ep + 1, "surrogate": a.surrogate, "eps_255": a.eps, "psnr": a.psnr})
    log.info("saved %s", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
