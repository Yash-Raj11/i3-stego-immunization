# CLAUDE.md — project memory for Claude Code

## What this is
**I3 — Imperceptible Image Immunization + steganographic payload.** A research prototype
(minor project) that protects face photos against generative editing/morphing *and*
embeds a robust ownership payload, tuned for Indian sharing habits (WhatsApp/Instagram
recompression, diverse skin tones, low light).

One perturbation, two layers, optimised jointly through an EOT ("social-media") layer:
- **Active**: W_PIR-masked adversarial perturbation against a surrogate VAE encoder
  (TAESD by default; SD-VAE / ensembles optional). Untargeted latent push by default.
- **Passive**: keyed spread-spectrum payload (64 bits → RS(16,8) → 128 coded bits) in the
  mid-band DCT of a canonical 256×256 luma grid, host-interference cancelled, plus an
  Ed25519-signed sidecar manifest.

## Commands
```bash
pip install -e ".[dev]"                 # once, inside .venv
python scripts/download_models.py       # TAESD + YuNet into models/
i3 keygen                               # keys/ (git-ignored, never read it)
i3 protect IMG --owner ID [--steps 20 for quick loops] [--debug]
i3 verify IMG --owner ID --manifest IMG.i3.json
i3 evaluate --original IMG --protected OUT --out reports/x
i3 inspect-jpeg FILE.jpg                # estimate a platform's JPEG quality
python -m pytest -m "not slow"          # fast tests (random surrogate, a few seconds)
ruff check src tests scripts && ruff format src tests scripts
```
Project skills: `/i3-eval [img]`, `/i3-experiment <flag> <v1,v2>`, `/i3-next-phase [item]`.

## Code map (src/i3/)
| file | role |
|---|---|
| `immunize.py` | joint optimiser: L2-normalised PGD + momentum on `delta_adv` (YCbCr), Adam on payload amplitudes, projections for caps / PSNR / face-PSNR / carrier energy |
| `stego.py` | `SpreadSpectrumCodec`: keyed slots+chips, `soft_bits` (differentiable), `render`, `decode` |
| `payload.py` | 64-bit payload (version, unix ts, 24-bit HMAC tag) + Reed-Solomon |
| `transforms.py` | EOT: `draw()` params, `apply()` (replayable), `apply_geometry()` for masks |
| `diff_jpeg.py` | differentiable JPEG (IJG tables, 4:2:0, STE rounding) |
| `saliency.py` | face detection (mediapipe → yunet → haar) on CLAHE copy, `build_wpir` |
| `jnd.py` | visibility (JND) map: luminance + texture masking |
| `generator.py` | Phase-2 U-Net warm start + shared `caps_from_masks` |
| `surrogates.py` | `taesd`, `sd-vae`, `random` (tests only), `a+b` ensembles |
| `attacks.py` | REAL (PIL) attacks + platform presets for evaluation |
| `pipeline.py` / `cli.py` | file-level protect / verify / evaluate, argparse CLI |
| `config.py` + `configs/default.yaml` | dataclass config, YAML overrides |

## Invariants — never break these
1. Whole-image PSNR ≥ `psnr_min` (38 dB) **after 8-bit rounding**; face PSNR ≥ `face_psnr_min` when set.
2. Payload decodes after `whatsapp_like` and `double_whatsapp` on the sample image.
3. `keys/` is secret: never read, print, log or commit it. Never commit `models/*` weights,
   `outputs/`, `reports/`, `data/` or any real person's photo.
4. Tests never download anything; use `--surrogate random` or skip-if-missing for models.
5. Evaluation uses REAL encoders (`attacks.py`), never the differentiable approximations.
6. Report face-region metrics, not just whole-image ones (global PSNR hides a noisy face).

## Conventions
- Python ≥ 3.10, type hints, `from __future__ import annotations`, ruff (line 120).
- Tensors: images `(B,3,H,W)` float in [0,1]; masks `(B,1,H,W)`; numpy images `uint8 HxWx3` RGB.
- New tunables go into a dataclass + `configs/default.yaml` + (if user-facing) a CLI flag.
- Commit messages: Conventional Commits (`feat(stego): ...`, `fix(eot): ...`, `exp: ...`).
- Record experiments in `docs/EXPERIMENTS.md` (the `/i3-experiment` skill does this).

## Gotchas
- CPU is slow: ~2 s per step for a 512 px image with TAESD. Use `--steps 20` while iterating,
  full runs on a GPU (Colab/Kaggle) — `--device cuda`.
- Images < 384 px short side make the payload fragile (canonical grid is 256).
- Cropping breaks payload sync (known, roadmap P4.1). WebP is the weakest codec.
- OpenCV 5 removed Haar cascades from the main package → pinned `<5`.
- MediaPipe ≥ 0.10.30 has no `mp.solutions`; we use the Tasks API (`FaceLandmarker`).
- The MediaPipe backend could not be exercised when the prototype was built (model host unreachable
  from the build sandbox); YuNet and Haar are tested. Verify MediaPipe first if you rely on it.
