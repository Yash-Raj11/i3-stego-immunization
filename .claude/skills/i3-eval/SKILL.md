---
name: i3-eval
description: Protect an image with I3 and evaluate it (PSNR, face PSNR, payload survival, immunity after purification). Use after changing anything in src/i3/ to check for regressions.
argument-hint: "[image path] [extra i3 protect flags]"
allowed-tools: Bash(i3 *) Bash(python -m pytest *) Read
---

Run the I3 protect → evaluate loop and report what changed.

1. Image = the first argument in `$ARGUMENTS`, default `assets/sample_astronaut.png`.
   Any remaining arguments are extra flags for `i3 protect`.
2. If `models/taesd_encoder.pth` is missing, run `python scripts/download_models.py` first.
   If `keys/` is missing, run `i3 keygen`.
3. Run:
   `i3 protect <image> -o outputs/eval/<stem>.protected.png --owner eval@example.com --debug <extra flags>`
   then
   `i3 evaluate --original <image> --protected outputs/eval/<stem>.protected.png --out reports/eval-<stem>`
4. Read `reports/eval-<stem>/report.json` and, if it exists, the previous
   `reports/eval-<stem>/report.prev.json`.
5. Reply with a short table: whole-image PSNR, face PSNR, payload decoded on
   whatsapp_like / webp_q75 / double_whatsapp / crop_90, and recon_drop_db for identity,
   whatsapp_like and down_up_0.5. Mark each number better/worse than the previous run.
6. Look at `reports/eval-<stem>/comparison.png` and say in one sentence whether the
   perturbation is visible on the face.
7. Copy report.json to report.prev.json for the next comparison.

Hard rules: whole-image PSNR must be >= 38 dB and the payload must decode on
`identity` and `whatsapp_like`. If either fails, say so first.
