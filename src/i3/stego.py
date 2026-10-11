"""Keyed spread-spectrum payload carrier ("passive forensic layer").

Why not classic steganography (LSB, S-UNIWARD/HUGO cost maps)?  Those hide data in
the least significant / highest-frequency detail, which is precisely what WhatsApp's
resize + JPEG Q~70 throws away. To survive the platform, the payload has to be a
*robust watermark*: low-to-mid frequency, redundant, error-corrected.

Design
------
1. Canonical domain: luma is resized to C x C (default 256x256) before embedding and
   decoding.  Any proportional resize of the shared image therefore maps back to the
   same grid (crop / aspect change does not – see ROADMAP "sync template").
2. 8x8 block DCT of the canonical luma; a mid band of zig-zag positions is used.
3. Every (block, coefficient) slot is assigned - by a key-seeded permutation - to one
   coded bit and gets a pseudo-random +-1 chip.  Each coded bit is spread over
   ~200 slots, interleaved across the whole image.
4. Decoding = correlation:  s_i = mean_j chip_j * X_j over the slots of bit i.
   Bit = sign(s_i).  The correlation is differentiable, so the immuniser can
   optimise margins *through* the EOT layer (joint optimisation in immunize.py).
5. The carrier is parameterised by free coefficient amplitudes A (one per slot),
   rendered to a C x C luma delta by inverse DCT and upsampled to image size.

Without the key, the slot assignment and chips are unknown, so the payload is hard
to read or to subtract (but NOT impossible to destroy - see docs/THREAT_MODEL.md).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .color import luma
from .dct import dct2_blocks, idct2_blocks, zigzag_order
from .transforms import resize


@dataclass
class StegoConfig:
    canonical: int = 256
    band: tuple[int, int] = (6, 28)  # zig-zag positions [lo, hi)
    n_bits: int = 128  # coded bits (payload.CODED_BITS)


class SpreadSpectrumCodec:
    def __init__(self, key: bytes, cfg: StegoConfig | None = None, device: str | torch.device = "cpu"):
        self.cfg = cfg or StegoConfig()
        c = self.cfg
        if c.canonical % 8:
            raise ValueError("canonical size must be a multiple of 8")
        seed = int.from_bytes(hashlib.sha256(b"i3-spread-spectrum-v1|" + key).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        nb = c.canonical // 8
        zz = np.array(zigzag_order(8)[c.band[0] : c.band[1]])
        n_blocks = nb * nb
        # flat index into a (nb, nb, 64) coefficient array
        slots = (np.arange(n_blocks)[:, None] * 64 + zz[None, :]).reshape(-1)
        slots = slots[rng.permutation(slots.size)]
        self.n_slots = slots.size
        if self.n_slots < 16 * c.n_bits:
            raise ValueError("too few slots per bit; enlarge canonical size or band")
        bit_of = np.arange(self.n_slots) % c.n_bits
        chips = rng.choice([-1.0, 1.0], size=self.n_slots)
        self.device = torch.device(device)
        self.slots = torch.as_tensor(slots, dtype=torch.long, device=self.device)
        self.bit_of = torch.as_tensor(bit_of, dtype=torch.long, device=self.device)
        self.chips = torch.as_tensor(chips, dtype=torch.float32, device=self.device)
        self.count = torch.bincount(self.bit_of, minlength=c.n_bits).float()

    # ------------------------------------------------------------------ analysis
    def canonical_luma(self, img: torch.Tensor) -> torch.Tensor:
        c = self.cfg.canonical
        return resize(luma(img), (c, c))

    def slot_values(self, img: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B, n_slots) DCT coefficients at the keyed slots."""
        coef = dct2_blocks(self.canonical_luma(img) - 0.5)  # (B,1,nb,nb,8,8)
        flat = coef.reshape(coef.shape[0], -1)
        return flat[:, self.slots]

    def soft_bits(self, img: torch.Tensor) -> torch.Tensor:
        """(B, n_bits) correlation scores; sign = decoded bit (+1 <-> bit 1)."""
        v = self.slot_values(img) * self.chips
        out = torch.zeros(v.shape[0], self.cfg.n_bits, device=v.device, dtype=v.dtype)
        out.index_add_(1, self.bit_of, v)
        return out / self.count.to(v)

    def decode(self, img: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Return (bits uint8 (n_bits,), z-scores (n_bits,)) for the first image."""
        with torch.no_grad():
            v = (self.slot_values(img) * self.chips)[0]
            s = self.soft_bits(img)[0]
            # per-bit noise estimate from chip-product spread -> z-score
            mean = s[self.bit_of]
            var = torch.zeros_like(s).index_add_(0, self.bit_of, (v - mean) ** 2) / (self.count - 1)
            z = s / torch.sqrt(var / self.count + 1e-12)
        return (s > 0).cpu().numpy().astype(np.uint8), z.cpu().numpy()

    # ------------------------------------------------------------------ synthesis
    def init_amplitudes(self, bits: np.ndarray, strength: float) -> torch.Tensor:
        b = torch.as_tensor(bits, dtype=torch.float32, device=self.device) * 2 - 1
        return (b[self.bit_of] * self.chips * strength).clone()

    def render(self, amplitudes: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Amplitudes (n_slots,) -> luma delta (1,1,H,W)."""
        nb = self.cfg.canonical // 8
        flat = torch.zeros(nb * nb * 64, device=amplitudes.device, dtype=amplitudes.dtype)
        flat = flat.index_put((self.slots,), amplitudes)
        coef = flat.view(1, 1, nb, nb, 8, 8)
        delta_c = idct2_blocks(coef)
        return F.interpolate(delta_c, size=size, mode="bilinear", align_corners=False)

    @staticmethod
    def margin_loss(soft: torch.Tensor, bits: torch.Tensor, margin: float) -> torch.Tensor:
        sign = bits.to(soft) * 2 - 1
        return F.relu(margin - sign * soft).mean()
