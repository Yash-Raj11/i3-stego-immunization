# Threat model — what I3 does and does not protect against

Being precise here is what separates a credible project from an over-claiming one.

## Assets
1. The *face* in a published photo (against morphing, img2img/inpainting edits, later face-swap).
2. Evidence of *ownership and time* of the photo.

## Adversaries
| adversary | capability | I3's expected effect |
|---|---|---|
| **A1 – bot operator** (e.g. Telegram "undress"/morph bots) | runs an off-the-shelf pipeline on whatever it downloads; no adaptation | **Best case.** Perturbation is crafted against open encoders through a WhatsApp-like EOT; still must be validated against the specific pipeline (P1.2, P3.1) |
| **A2 – casual editor** | re-shares, crops, filters, screenshots | Payload survives recompression, resizing, filters; **not** cropping yet (P4.1) or heavy screenshots with UI borders |
| **A3 – informed attacker** | knows about protective perturbations; purifies (JPEG, blur, down-up scaling, DiffPure, noisy upscaling) before editing | **Partial at best.** Our purification rows show a reduced but non-zero effect for simple filters; strong diffusion-based purification is known to defeat protective perturbations in general (Hönig et al. 2025) |
| **A4 – adaptive attacker** | has the I3 code and trains a model to remove it | Not defended. Publishing your exact method + weights helps them – this is inherent to the approach |
| **A5 – key thief** | obtains `keys/i3_secret.key` | Can read, strip and forge payloads. Keep the key offline; rotate if leaked |

## What the payload proves – and what it doesn't
- A decoded payload whose HMAC tag matches your owner id shows that **someone holding your key**
  processed this image at that timestamp. It does not prove you are the person in it or the photographer.
- Watermarks can be removed by regeneration (re-encoding the image with a generative model) –
  Zhao et al. 2023. Absence of a payload therefore proves nothing.
- The signed manifest + original file hash + an independently timestamped storage location (email to
  yourself, versioned cloud bucket, public Git commit of the manifest *hash only*) is stronger evidence
  than the pixels alone.

## Known limitations of this prototype
- Surrogate is TAESD (CPU-friendly). Transfer to the full SD VAE, SDXL/Flux VAEs and identity models is
  **not yet measured**.
- Immunity metric is a VAE-reconstruction proxy, not an end-to-end img2img/face-swap test (P1.2/P3.1).
- One sample image so far. No skin-tone audit yet (P5.1) – the JND/CLAHE design *aims* at inclusivity,
  it has not been *shown* yet.
- Payload: no crop/rotation resilience; images < ~384 px are fragile; WebP is the weakest codec.

## Responsible-use notes
- Process photos locally; never upload users' faces to a server you don't control.
- Don't market it as "deepfake-proof". Say what you measured, against which models, under which attacks.
- For ID documents (Aadhaar/PAN), adversarial noise can break legitimate OCR/KYC; prefer masking and
  watermark-only modes, and test with the verifier you care about.
