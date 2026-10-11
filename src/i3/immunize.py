"""Joint "steganographic immunisation" optimiser.

    x_protected = clip( x + R(delta_adv) + vis * Carrier(A) )

    delta_adv : adversarial component, parameterised in YCbCr.  Per-pixel caps:
                luma <= eps_max * W_PIR * vis_norm,  chroma <= chroma_ratio * that.
    Carrier(A): keyed spread-spectrum payload, luma only, smooth, masked by vis.

Both are optimised together through the EOT layer:

    L = L_adv( E(T(x_p)), weighted by W_PIR )  +  lambda_wm * L_margin( D(T(x_p)), bits )

so the adversarial part learns not to erase the payload, and the payload carrier is
free to shape itself in directions that *also* hurt the encoder – the "adversarial
stego-payload" idea, implemented as a single optimisation problem.

Why L2 steps instead of classic sign-PGD?  The binding constraint is PSNR, which is
an L2 budget. Sign steps spread energy uniformly inside the L-inf box and waste that
budget on pixels that barely move the latent; normalised-gradient (L2) steps spend it
where the encoder is most sensitive. (In the prototype's experiments on the sample face,
sign-PGD at PSNR >= 38 dB left almost no effect after JPEG; the L2 + EOT + untargeted
variant kept a visible VAE-reconstruction degradation after WhatsApp-like compression.)

Hard constraints, enforced every step and re-checked after 8-bit rounding:
    PSNR(x, x_p)            >= psnr_min       (global, default 38 dB)
    PSNR on the face region >= face_psnr_min  (optional; local visibility matters more)
    |delta_adv| <= per-pixel caps
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .color import ycbcr_delta_to_rgb
from .generator import caps_from_masks
from .stego import SpreadSpectrumCodec
from .surrogates import Surrogate, members
from .transforms import EOT, EOTConfig

log = logging.getLogger(__name__)


@dataclass
class ImmunizeConfig:
    eps: float = 16 / 255  # per-pixel cap for the adversarial part where W_PIR = 1 (PSNR is the real budget)
    chroma_ratio: float = 0.15  # chroma cap relative to luma (colour noise is the most visible artefact)
    steps: int = 100
    step_frac: float = 0.15  # L2 step as a fraction of the L2 budget (cosine-decayed)
    momentum: float = 0.5
    eot_samples: int = 1
    psnr_min: float = 38.0
    face_psnr_min: float | None = 34.0  # local visibility bound on the face (None = global PSNR only)
    mode: str = "untargeted"  # untargeted: push latent away from the clean one | targeted: pull to grey latent
    payload_alpha: float = 0.005  # target correlation per coded bit (higher = more robust, more visible)
    payload_lr: float = 2e-4  # the carrier may re-shape itself (to also hurt the encoder) ...
    payload_energy_slack: float = 1.1  # ... but never grow beyond 1.1x its initial energy
    adv_floor: float = 0.0  # adversarial cap in the background, relative to the face (0 = face/halo only)
    lambda_wm: float = 1.0
    random_start: bool = True
    seed: int = 0
    eot: EOTConfig = field(default_factory=EOTConfig)


@dataclass
class ImmunizeResult:
    protected: torch.Tensor  # (1,3,H,W) in [0,1], already 8-bit quantised
    psnr: float
    face_psnr: float | None
    latent_shift: float  # ||E(x_p) - E(x)|| / ||E(x)||  on the face-weighted latent (clean, no EOT)
    payload_ok: bool
    seconds: float
    history: list = field(default_factory=list)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("protected")
        d.pop("history")
        return d


def _psnr(a: torch.Tensor, b: torch.Tensor, region: torch.Tensor | None = None) -> float:
    se = (a - b) ** 2
    mse = se.mean().item() if region is None else (se * region).sum().item() / (region.sum().item() * 3 + 1e-9)
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def latent_loss(
    surr: Surrogate, x_adv: torch.Tensor, weight: torch.Tensor, mode: str, x_ref: torch.Tensor | None
) -> torch.Tensor:
    """Mask-weighted latent objective, normalised per surrogate. Lower is better for the defender."""
    total = 0.0
    for m in members(surr):
        z = m.encode(x_adv)
        with torch.no_grad():
            z_t = m.encode(torch.full_like(x_adv, 0.5)) if mode == "targeted" else m.encode(x_ref)
        w = F.adaptive_avg_pool2d(weight, z.shape[-2:])
        l = ((z - z_t) ** 2 * w).sum() / (w.sum() * z.shape[1] + 1e-9) / (torch.mean(z_t**2) + 1e-6)
        total = total + (l if mode == "targeted" else -l)
    return total


def init_payload_amplitudes(
    codec: SpreadSpectrumCodec, x: torch.Tensor, bits: np.ndarray, vis: torch.Tensor, alpha: float, iters: int = 4
) -> torch.Tensor:
    """Closed-loop host-interference cancellation (Improved Spread Spectrum style).

    Choose A so that every coded bit's correlation on the *clean* image equals
    +-alpha exactly, i.e. the host image's own projection is cancelled.
    """
    size = x.shape[-2:]
    b = torch.as_tensor(bits, dtype=x.dtype, device=x.device) * 2 - 1
    with torch.no_grad():
        unit = codec.init_amplitudes(bits, 1.0).to(x)
        gain = (codec.soft_bits(codec.render(unit, size).expand_as(x) * vis)[0] * b).clamp_min(1e-3)
        A = torch.zeros(codec.n_slots, device=x.device, dtype=x.dtype)
        for _ in range(iters):
            s = codec.soft_bits(x + codec.render(A, size) * vis)[0]
            A = A + codec.chips.to(x) * ((alpha * b - s) / gain)[codec.bit_of]
    return A


def _max_scale(a: torch.Tensor, c: torch.Tensor, budget_sse: float) -> tuple[float, float]:
    """Largest s<=1 with sum((s*a + c)^2) <= budget; returns (s_adv, s_carrier)."""
    aa, ac, cc = (a * a).sum().item(), (a * c).sum().item(), (c * c).sum().item()
    if cc >= budget_sse:  # carrier alone is over budget -> shrink it, drop the adversarial part
        return 0.0, math.sqrt(budget_sse / (cc + 1e-12)) * 0.98
    if aa + 2 * ac + cc <= budget_sse:
        return 1.0, 1.0
    disc = ac * ac - aa * (cc - budget_sse)
    s = (-ac + math.sqrt(max(disc, 0.0))) / (aa + 1e-12)
    return max(0.0, min(1.0, s)), 1.0


def immunize(
    x: torch.Tensor,
    surrogate: Surrogate,
    codec: SpreadSpectrumCodec,
    bits: np.ndarray,
    wpir: torch.Tensor,
    vis: torch.Tensor,
    cfg: ImmunizeConfig | None = None,
    progress: bool = True,
    init_adv: torch.Tensor | None = None,
) -> ImmunizeResult:
    """x: (1,3,H,W) in [0,1]; wpir/vis: (1,1,H,W).

    init_adv: optional YCbCr warm start for the adversarial part (e.g. from the
    Phase-2 generator); then a handful of steps is usually enough.
    """
    cfg = cfg or ImmunizeConfig()
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    eot = EOT(cfg.eot, seed=cfg.seed)
    size = x.shape[-2:]
    bits_t = torch.as_tensor(bits, device=x.device)
    face = (wpir > 0.5).to(x)  # region used for the optional local PSNR constraint / report

    # ---- caps & budgets -------------------------------------------------------
    cap = caps_from_masks(wpir, vis, cfg.eps, cfg.chroma_ratio, cfg.adv_floor)  # JND shapes *where* energy goes
    slack = 0.15  # dB, absorbs 8-bit rounding
    sse_global = 10 ** (-(cfg.psnr_min + slack) / 10) * x.numel()
    sse_face = None
    if cfg.face_psnr_min is not None and face.sum() > 0:
        sse_face = 10 ** (-(cfg.face_psnr_min + slack) / 10) * face.sum().item() * 3
    l2_budget = math.sqrt(sse_global)

    # ---- parameters -----------------------------------------------------------
    A = init_payload_amplitudes(codec, x, bits, vis, cfg.payload_alpha).requires_grad_(True)
    with torch.no_grad():
        carrier_sse_max = ((codec.render(A, size) * vis) ** 2).sum().item() * 3 * cfg.payload_energy_slack
    if init_adv is not None:
        d_adv = init_adv.detach().clone().to(cap)
    elif cfg.random_start:
        d_adv = (torch.rand_like(cap) * 2 - 1) * cap * 0.25
    else:
        d_adv = torch.zeros_like(cap)
    d_adv.requires_grad_(True)
    opt_A = torch.optim.Adam([A], lr=cfg.payload_lr)
    g_mom = torch.zeros_like(d_adv)

    def compose(d, amp):
        carrier = codec.render(amp, size) * vis
        return ycbcr_delta_to_rgb(d), carrier.expand(-1, 3, -1, -1)

    def project():
        with torch.no_grad():
            d_adv.copy_(torch.maximum(torch.minimum(d_adv, cap), -cap))
            c_sse = ((codec.render(A, size) * vis) ** 2).sum().item() * 3
            if c_sse > carrier_sse_max:
                A.mul_(math.sqrt(carrier_sse_max / c_sse))
            a_rgb, c_rgb = compose(d_adv, A)
            s_a, s_c = _max_scale(a_rgb, c_rgb, sse_global)
            if sse_face is not None:
                fa, fc = _max_scale(a_rgb * face, c_rgb * face, sse_face)
                s_a, s_c = min(s_a, fa), min(s_c, fc)
            if s_a < 1.0:
                d_adv.mul_(s_a)
            if s_c < 1.0:
                A.mul_(s_c)

    project()
    history = []
    margin = cfg.payload_alpha * 0.8
    for step in range(cfg.steps):
        a_rgb, c_rgb = compose(d_adv, A)
        x_p = (x + a_rgb + c_rgb).clamp(0, 1)
        l_adv_tot, l_wm_tot = 0.0, 0.0
        for _ in range(cfg.eot_samples):
            params = eot.draw(*size)
            s = eot.apply(x_p, params)
            x_ref = eot.apply(x, params).adv if cfg.mode != "targeted" else None
            weight = EOT.apply_geometry(wpir, params)
            l_adv = latent_loss(surrogate, s.adv, weight, cfg.mode, x_ref)
            soft = codec.soft_bits(s.common)
            l_wm = torch.relu(1 - (bits_t.to(soft) * 2 - 1) * soft / margin).mean()
            l_adv_tot = l_adv_tot + l_adv / cfg.eot_samples
            l_wm_tot = l_wm_tot + l_wm / cfg.eot_samples
        loss = l_adv_tot + cfg.lambda_wm * l_wm_tot
        opt_A.zero_grad()
        d_adv.grad = None
        loss.backward()

        with torch.no_grad():
            g = d_adv.grad / (d_adv.grad.norm() + 1e-12)
            g_mom.mul_(cfg.momentum).add_(g)
            lr = cfg.step_frac * l2_budget * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / max(1, cfg.steps))))
            d_adv -= lr * g_mom / (g_mom.norm() + 1e-12)
        opt_A.step()
        project()
        rec = {"step": step, "adv": float(l_adv_tot.detach()), "wm": float(l_wm_tot.detach())}
        history.append(rec)
        if progress and (step % 10 == 0 or step == cfg.steps - 1):
            log.info("step %3d/%d  L_adv %+.4f  L_wm %.4f", step, cfg.steps, rec["adv"], rec["wm"])

    # ---- finalise: quantise to 8 bit, re-check constraints ---------------------
    with torch.no_grad():
        a_rgb, c_rgb = compose(d_adv, A)
        scale = 1.0
        for _ in range(30):
            out = torch.round((x + scale * a_rgb + c_rgb).clamp(0, 1) * 255) / 255
            p, fp = _psnr(out, x), (_psnr(out, x, face) if face.sum() > 0 else None)
            if p >= cfg.psnr_min and (cfg.face_psnr_min is None or fp is None or fp >= cfg.face_psnr_min):
                break
            scale *= 0.9
        shifts = []
        for mm in members(surrogate):
            zp, zc = mm.encode(out), mm.encode(x)
            w = F.adaptive_avg_pool2d(wpir, zc.shape[-2:])
            shifts.append((((zp - zc) ** 2 * w).sum() / ((zc**2) * w).sum()).sqrt().item())
        dec, _ = codec.decode(out)
        ok = bool(np.array_equal(dec, bits))
    return ImmunizeResult(
        protected=out,
        psnr=round(p, 2),
        face_psnr=None if fp is None else round(fp, 2),
        latent_shift=round(float(np.mean(shifts)), 4),
        payload_ok=ok,
        seconds=round(time.time() - t0, 1),
        history=history,
    )
