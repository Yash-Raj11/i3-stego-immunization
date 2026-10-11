# Roadmap — with copy-paste prompts for Claude Code

Work through items in order; each is sized for one Claude Code session and one branch/PR.
Paste the prompt into Claude Code (or run `/i3-next-phase <id>`). Tick items off here as you go and
record numbers in `docs/EXPERIMENTS.md`.

Legend: 🖥 CPU is fine · ⚡ needs a GPU (Colab/Kaggle free tier is enough)

---

## Phase 0 — prototype ✅
- [x] W_PIR saliency (MediaPipe / YuNet / Haar on a CLAHE copy) + JND visibility map
- [x] Differentiable JPEG + EOT layer (resize, colour "beauty filter", blur, noise, JPEG, crop)
- [x] Spread-spectrum payload (64 bit, RS, HMAC tag) with host-interference cancellation
- [x] Joint optimiser (L2-PGD on the adversarial part + Adam on the carrier), PSNR + face-PSNR constraints
- [x] Signed manifest, CLI, evaluation harness with real codecs and purification attacks
- [x] Phase-2 generator scaffold + training script

---

## ★ Milestone 1 — prove it on Stable Diffusion (current)
- [x] `i3 demo` / `i3 morph-test`: SD-1.5 img2img + inpainting on original vs protected, same seed/prompt,
      before/after figures and face-region metrics (docs/DEMO.md)
- [x] `configs/sd_demo.yaml`: protect against the real SD VAE (+ TAESD ensemble)
- [ ] **M1.1** Run the demo on the sample + 5–10 consented / public-domain portraits (include darker
      skin tones and a low-light photo). Prompt for Claude Code: `/i3-sd-demo <photo>` for each.
- [ ] **M1.2** Tune for a clear, honest result.
  > Use /i3-experiment to sweep `--face-psnr 30,32,34` and then `--mode targeted,untargeted` with
  > `i3 demo ... --config configs/sd_demo.yaml`. Pick the setting with the largest img2img drop at
  > face PSNR ≥ 32 dB. Record everything in docs/EXPERIMENTS.md and update configs/sd_demo.yaml.
- [ ] **M1.3** Robustness of the demo: does the effect survive WhatsApp first?
  > Add an `--attack whatsapp_like` option to `i3 morph-test` that applies an `attacks.ATTACKS` entry to
  > both inputs before editing. Add a test with the fake backend. Then run the demo with it.
- [ ] **M1.4** Presentation pack.
  > Write `scripts/make_slides_figures.py` that collects before_after.png from every reports/demo_* folder
  > into one contact sheet plus a summary table (mean drop per edit, face PSNR of inputs).

Only after Milestone 1 is convincing, continue with other models below.

---

## Phase 1 — beyond SD-1.5 (after Milestone 1)

**P1.1 ⚡ Real SD-VAE + ensemble surrogate.** Today we optimise against TAESD (a distilled copy of the
SD VAE). Measure transfer and use an ensemble.
> Plan first. Add a Colab notebook `notebooks/gpu_eval.ipynb` that clones the repo, installs
> `.[sd]`, downloads models and runs `i3 protect` on the sample with `--surrogate taesd`, `sd-vae`
> and `taesd+sd-vae` (300 steps, 4 EOT samples, `--device cuda`). For each output run
> `i3 evaluate --surrogate sd-vae`. Summarise transfer (TAESD-only → evaluated on SD-VAE) in a table
> in docs/EXPERIMENTS.md. Don't change the optimiser.

**P1.2 ✅ done as Milestone 1 (`i3 morph-test`).** Original description kept for reference: Recon-PSNR is only a proxy; test the real threat.
> Add `scripts/sd_attack_test.py`: given original + protected images, run diffusers
> StableDiffusionImg2ImgPipeline (an SD-1.5 checkpoint such as stable-diffusion-v1-5/stable-diffusion-v1-5, strength 0.3/0.5/0.7, fixed seed,
> a neutral prompt like "a photo of a person") and an inpainting pipeline (mask = face region from
> W_PIR). Save side-by-side grids and compute: face-region PSNR/SSIM vs original, LPIPS if installed,
> and face-detection success with YuNet. Mark as GPU-only; no unit test should import diffusers.

**P1.3 🖥 Calibrate against real platforms.** (manual + small code change)
> I'll put real downloaded files in data/platform_samples/{whatsapp,whatsapp_hd,instagram,telegram}/.
> Write `scripts/calibrate_platforms.py` that runs `estimate_jpeg_quality` on every file, prints
> size/quality/subsampling statistics per platform, and proposes updated `PLATFORM_PRESETS` and EOT
> `jpeg_quality` / `resize_scale` ranges. Show me the diff before applying.

**P1.4 🖥 Perceptual metric.** PSNR is a weak proxy for "invisible".
> Add optional LPIPS (pip extra `.[perceptual]`) to `i3 evaluate` (whole image and face crop), and an
> optional `lpips_max` constraint in the optimiser implemented as a penalty term. Default off. Tests
> must skip when lpips is not installed.

---

## Phase 2 — speed (mobile-first)

**P2.1 ⚡ Train the amortised generator.**
> Using scripts/train_generator.py, write a Colab notebook that trains on FairFace (download
> instructions only — never commit data) for 20 epochs at 256 px, then compares on 50 held-out faces:
> (a) generator only (`--steps 0`), (b) generator + 15 refinement steps, (c) 100-step optimisation from
> scratch — time per image, face PSNR, latent shift after whatsapp_like. Results to docs/EXPERIMENTS.md.

**P2.2 🖥 Export for on-device use.**
> Add `scripts/export_generator.py` exporting the generator to TorchScript and ONNX (dynamic H/W),
> verify numerically against PyTorch on 3 images (max abs diff < 1e-4), and document how a mobile app
> would combine it with the closed-form payload carrier (`init_payload_amplitudes` needs no encoder).

---

## Phase 3 — face-swap / identity layer

**P3.1 ⚡ Identity surrogate.** Face-swap engines (InSwapper/SimSwap-style) key on an ArcFace identity
embedding, not a VAE latent.
> Plan first. Add an `identity` surrogate: load an ArcFace-family ONNX model from models/ via
> onnx2torch (optional extra), align the face crop with the YuNet 5 landmarks, and add a loss term
> maximising cosine distance between identity embeddings of clean and protected faces (through EOT).
> Make it combinable: `--surrogate taesd+identity`. Report identity cosine similarity before/after
> purification in `i3 evaluate`. Note the model licence in models/README.md (InsightFace models are
> non-commercial).

**P3.2 🖥 Detector-level disruption (optional).** Many bots first crop the face with a detector.
> Measure whether protected images reduce YuNet/MediaPipe detection confidence; if cheap, add an
> optional loss term on YuNet's score map. Report detection rate before/after attacks.

---

## Phase 4 — payload robustness

**P4.1 🖥 Survive cropping.** Today a 10 % crop desynchronises the grid.
> Implement crop-robust decoding: tile the payload grid (e.g. 2×2 periodic carrier) and at decode time
> search offsets/scales by maximising the mean |z|, or add a keyed synchronisation template. Add
> crop_90/crop_80 tests that must pass; keep PSNR and all existing attack results.

**P4.2 🖥 WebP in the EOT layer.** WebP (VP8, 4×4 transforms + loop filter) is the weakest codec today.
> Add a differentiable 'webp-like' transform (4×4 block DCT quantisation + mild deblocking blur) to
> transforms.py with probability p_webp, and show bit-accuracy under webp_q60/q75 improves.

**P4.3 🖥 Small images.**
> Make the codec multi-scale: choose canonical 128 for images with short side < 384, and let `decode`
> try 256 then 128. Add tests for 256 px and 320 px images after whatsapp_like.

**P4.4 🖥 Metadata provenance alongside pixels.**
> Write the manifest hash into XMP/EXIF of the protected PNG/JPEG and (optionally) emit a C2PA
> manifest with c2pa-python. Explain in README that platforms strip metadata, which is exactly why the
> pixel payload exists.

---

## Phase 5 — inclusivity & evaluation (this is what makes the project *yours*)

**P5.1 🖥 Skin-tone & lighting audit.**
> Add `scripts/audit.py`: for a folder of faces with a CSV of labels (e.g. FairFace group and a
> Monk-scale estimate), protect each image and report per group: detection backend used, face PSNR,
> payload survival after whatsapp_like, latent shift after whatsapp_like, and a low-light variant
> (gamma 1.8 × 0.35 exposure). Output a markdown table + bar chart. Flag any group whose detection rate
> or face PSNR differs by > 2 points from the mean.

**P5.2 🖥 Visibility user study.**
> Generate a 2-alternative-forced-choice web page (single HTML file) that shows original vs protected
> pairs in random order and records which one people think was modified; analyse with a binomial
> test. 20 pairs, 10 volunteers is enough for a minor project.

---

## Phase 6 — product ideas (pick one if time allows)
- Android app (Kotlin or Flutter) running the exported generator + closed-form payload on-device.
- Browser extension / web demo for matrimonial-profile photos (process locally, never upload).
- "Document mode" for ID scans – careful: perturbations can break legitimate OCR/KYC checks; masking
  (e.g. masked Aadhaar) is usually the better protection there. Evaluate before claiming anything.
