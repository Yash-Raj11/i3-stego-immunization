# Experiment log

Append new runs at the bottom (the `/i3-experiment` skill does this for you). Always record the commit,
the image, the exact command and the numbers — your report/viva will need them.

All numbers below: sample image `assets/sample_astronaut.png` (512×512, public-domain NASA photo),
surrogate **TAESD**, CPU, face detected with YuNet. "Recon drop" = how many dB worse the TAESD
reconstruction of the *face region* gets for the protected image vs the original (higher = stronger
immunisation). "After WhatsApp" = real Pillow JPEG Q72 after resizing to ≤1600 px (`whatsapp_like`).

---

## 2026-10-11 — prototype bring-up

### E1. Sign-PGD (classic PhotoGuard-style) is too weak at PSNR ≥ 38 dB
256×256 face crop, L∞ ε = 8/255 on W_PIR, 40 steps.

| variant | crop PSNR | recon drop (clean) | recon drop after JPEG Q72 |
|---|---|---|---|
| targeted, no EOT | 42.5 | 2.10 | −0.10 |
| untargeted, no EOT | 41.6 | 1.91 | 0.76 |
| targeted + EOT | 43.7 | −0.42 | −0.05 |
| untargeted + EOT | 44.2 | 0.09 | 0.04 |

Without EOT the effect disappears after one JPEG; with EOT and sign steps it never builds up.

### E2. L2-normalised steps + EOT + face-weighted latent loss
Same crop, 100 steps, PSNR budget 33 dB on the crop (≈ 38 dB on the full image when the face is ~4 %
of it), JND-shaped caps (RGB chroma not yet reduced).

| variant | face PSNR | clean | JPEG Q75 | WhatsApp-like | down-up 0.5 |
|---|---|---|---|---|---|
| targeted, no EOT | 30.3 | 4.91 | 1.12 | 1.03 | 1.53 |
| targeted + EOT | 30.7 | 1.91 | 1.43 | 1.53 | 1.99 |
| **untargeted + EOT** | 30.7 | 3.12 | **2.78** | **2.95** | **2.36** |
| targeted + EOT + momentum 0.9 | 28.6 | 2.45 | 2.18 | 2.08 | 2.41 |

→ default switched to **untargeted, L2 steps, EOT**. Visible green tint on eyes/teeth → chroma budget
reduced (`chroma_ratio` 0.25 → 0.15).

### E3. Bug found: payload carrier drifted and ate the PSNR budget
First full-image run: whole-image PSNR 38.6 but face PSNR 37.3 and recon drop ≈ 0 — the payload carrier
(optimised with Adam at lr 2e-3) had grown, and the background cap let the adversarial part spread over
25× more background pixels. Fixed with `payload_energy_slack` (carrier may reshape, not grow),
`payload_lr` 2e-4 and `adv_floor` 0 (adversarial energy only on face + halo).

### E4. Full image — the visibility / strength trade-off (100 steps, 1 EOT sample, ~3.5 min CPU)

| constraint | whole PSNR | face PSNR | SSIM | latent shift after WhatsApp | recon drop after WhatsApp | visible? |
|---|---|---|---|---|---|---|
| global ≥ 38 dB only (chroma 0.25) | 38.2 | **27.0** | – | 0.88 | 1.83 | **yes, clearly** |
| **global ≥ 38 + face ≥ 34 dB (default)** | 41.4 | 34.1 | 0.977 | 0.58 | 0.59 | subtle texture when zoomed |

Lesson: *a global PSNR target says little when the perturbation is concentrated on the face.*
Report face PSNR (and later LPIPS / a user study) next to global PSNR.

Default run, full purification table (latent shift / recon drop dB):
identity 0.56/0.67 · whatsapp_like 0.58/0.59 · jpeg_q75 0.56/0.84 · blur_1.0 0.33/0.62 ·
median_3 0.38/0.90 · down_up_0.5 0.34/0.62 · noise_std4 0.52/0.80 · resize_0.5+jpeg75 0.33/0.50.

### E5. Payload robustness
Closed-form carrier only (no optimisation), α = per-bit correlation target, 3 keys averaged:

| band (zig-zag) | α | PSNR (carrier only) | WebP Q75 bit-acc | beauty filter + JPEG80 | WhatsApp-like |
|---|---|---|---|---|---|
| **6–28** | 0.005 | 43.2 | 0.914 | 0.966 | 1.000 |
| 3–20 | 0.005 | 41.0 | 0.833 | 0.839 | 1.000 |
| 4–24 | 0.005 | 41.4 | 0.852 | 0.862 | 1.000 |

Mid band 6–28 is best. After the joint EOT optimisation (default run) the payload decodes under **every
attack except cropping** (WebP Q75 bit-acc 0.992, corrected by Reed-Solomon). Crop 90 % → 0.50 (no sync
yet, roadmap P4.1).

### Open questions for next experiments
- Does the TAESD-optimised perturbation transfer to the real SD-VAE and to img2img? (P1.1, P1.2)
- More steps / EOT samples on a GPU: how far does recon drop go at face PSNR 34–36?
- Per-skin-tone and low-light behaviour on a real dataset (P5.1).
