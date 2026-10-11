---
name: i3-sd-demo
description: Run the Stable Diffusion before/after demo (protect a photo, run img2img + inpainting on original and protected, build figures) and summarise whether the immunisation visibly worked. Use for the SD milestone or when the user asks for before/after results.
argument-hint: "[photo path] [extra i3 demo flags, e.g. --face-psnr 32 --mode targeted]"
allowed-tools: Bash(i3 *) Bash(pip install -e *) Read
---

Produce the Stable Diffusion before/after evidence for the current milestone.

1. Photo = first word of `$ARGUMENTS` (default `assets/sample_astronaut.png`); the rest are extra
   flags for `i3 demo`. Output folder: `reports/demo_<photo stem>` (add a suffix such as `_fp32` if the
   user is comparing settings, so earlier results are not overwritten).
2. If `python -c "import diffusers, transformers"` fails, run `pip install -e ".[sd]"`.
3. Run:
   `i3 demo <photo> --owner demo@example.com --config configs/sd_demo.yaml --out <folder> <extra flags>`
   This takes several minutes (model download on the first run). Don't interrupt it.
4. Read `<folder>/report.md` and look at `<folder>/before_after.png` and `<folder>/grid.png`.
5. Reply with:
   - the metrics table (face PSNR of the immunised input; per edit: face PSNR original→edit,
     protected→edit, drop dB, detector scores),
   - one honest sentence per edit: is the damage to the protected edit clearly visible, subtle, or absent?
   - whether the immunised input looks unchanged,
   - one concrete next setting to try if the effect is weak (lower `--face-psnr`, more `--steps`,
     `--mode targeted`, `--surrogate sd-vae`), and why.
6. Append a short entry to `docs/EXPERIMENTS.md` (date, commit `git log -1 --format=%h`, command, table).

Never claim protection against models that were not tested. Never commit files from `reports/`.
