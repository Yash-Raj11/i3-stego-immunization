"""Surrogate image encoders the immunisation is optimised against.

Generative editors (Stable Diffusion img2img / inpainting, many face-morph pipelines)
first map the photo into a latent space with a VAE encoder. If we push the photo's
latent towards a meaningless target, everything downstream is built on garbage
(PhotoGuard "encoder attack", Salman et al. 2023).

Available surrogates
  * "taesd"   – Tiny AutoEncoder for SD (MIT, ~5 MB). Distilled to mimic the SD-1.x
                VAE latent space; fast enough for CPU / mobile-class hardware.
                Weights: scripts/download_models.py -> models/taesd_*.pth
  * "sd-vae"  – the real SD VAE via diffusers (stabilityai/sd-vae-ft-mse, ~330 MB).
                Slower; use it (or the ensemble) for final results / on a GPU.
  * "taesd+sd-vae" – ensemble (better transfer). Any "a+b" combination works.
  * "random"  – randomly initialised conv net. ONLY for unit tests; meaningless.

Face-swap engines (e.g. InSwapper/SimSwap) key on an *identity* embedding (ArcFace),
not a VAE latent - adding an identity surrogate is Phase 3 in docs/ROADMAP.md.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ._taesd import Decoder as TAEDecoder
from ._taesd import Encoder as TAEEncoder


class Surrogate(nn.Module):
    name = "base"

    def encode(self, x: torch.Tensor) -> torch.Tensor:  # x in [0,1]
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor | None:
        return None

    def freeze(self) -> Surrogate:
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        return self


class TAESDSurrogate(Surrogate):
    name = "taesd"

    def __init__(self, models_dir: str | Path = "models"):
        super().__init__()
        d = Path(models_dir)
        enc, dec = d / "taesd_encoder.pth", d / "taesd_decoder.pth"
        if not enc.exists():
            raise FileNotFoundError(f"{enc} missing - run: python scripts/download_models.py")
        self.encoder, self.decoder = TAEEncoder(4), TAEDecoder(4)
        self.encoder.load_state_dict(torch.load(enc, map_location="cpu", weights_only=True))
        if dec.exists():
            self.decoder.load_state_dict(torch.load(dec, map_location="cpu", weights_only=True))
        else:
            self.decoder = None

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return None if self.decoder is None else self.decoder(z).clamp(0, 1)


class SDVAESurrogate(Surrogate):
    name = "sd-vae"

    def __init__(self, repo: str = "stabilityai/sd-vae-ft-mse", vae=None):
        super().__init__()
        if vae is None:
            from diffusers import AutoencoderKL  # optional dependency: pip install -e ".[sd]"

            vae = AutoencoderKL.from_pretrained(repo)
        self.vae = vae
        self.scale = getattr(vae.config, "scaling_factor", 0.18215)

    def encode(self, x):
        return self.vae.encode(x * 2 - 1).latent_dist.mean * self.scale

    def decode(self, z):
        return ((self.vae.decode(z / self.scale).sample + 1) / 2).clamp(0, 1)


class RandomSurrogate(Surrogate):
    """Test-only stand-in with the same interface (stride-8 latent, 4 channels)."""

    name = "random"

    def __init__(self, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 4, 3, 2, 1)
        )
        for p in self.net.parameters():
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g) * 0.2)

    def encode(self, x):
        return self.net(x)


class Ensemble(Surrogate):
    def __init__(self, members: list[Surrogate]):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.name = "+".join(m.name for m in members)


def get_surrogate(spec: str, models_dir: str | Path = "models", device: str | torch.device = "cpu") -> Surrogate:
    parts = [s.strip() for s in spec.split("+") if s.strip()]
    built: list[Surrogate] = []
    for p in parts:
        if p == "taesd":
            built.append(TAESDSurrogate(models_dir))
        elif p == "sd-vae":
            built.append(SDVAESurrogate())
        elif p == "random":
            built.append(RandomSurrogate())
        else:
            raise ValueError(f"unknown surrogate {p!r} (taesd | sd-vae | random, joined with '+')")
    s = built[0] if len(built) == 1 else Ensemble(built)
    return s.to(device).freeze()


def members(s: Surrogate) -> list[Surrogate]:
    return list(s.members) if isinstance(s, Ensemble) else [s]
