# Stable Diffusion demo — before / after

Goal of the current milestone: **show that an immunised photo breaks real Stable Diffusion edits,
while looking the same to people.** Other models (SDXL, Flux, face-swap engines) come after this.

## One-time setup
```bash
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[sd]"             # diffusers + transformers + accelerate
```
The first run downloads from Hugging Face (cached afterwards):
- `stabilityai/sd-vae-ft-mse` (~330 MB) – the real SD VAE used as surrogate during protection
- `stable-diffusion-v1-5/stable-diffusion-v1-5` (~2 GB) – img2img edits
- `stable-diffusion-v1-5/stable-diffusion-inpainting` (~2 GB) – background-replacement edits

Hardware: Apple Silicon (MPS) or an NVIDIA GPU. 16 GB RAM recommended. CPU works but is very slow.

## Run it
**One command** (protect + edits + figures):
```bash
i3 demo assets/sample_astronaut.png --owner you@example.com --config configs/sd_demo.yaml --out reports/demo_sd
```
or, in VS Code: **Terminal → Run Task… → "i3: SD demo (sample)"**.

Already have a protected image? Run only the edits:
```bash
i3 morph-test --original assets/sample_astronaut.png --protected outputs/sample.protected.png --out reports/morph_test
```

Your own photo (use only photos you have permission to use):
```bash
i3 demo path/to/photo.jpg --owner you@example.com --config configs/sd_demo.yaml --out reports/my_photo
```

## What you get (`--out` folder)
| file | use |
|---|---|
| `before_after.png` | **the slide**: original vs immunised (top) and the AI edit of each (bottom) |
| `grid.png` | every edit with metrics under each image |
| `edits/*.png` | each individual image, full size |
| `report.md` / `report.json` | the numbers |
| `<name>.protected.png` | the immunised photo (+ `.i3.json` manifest, `_debug/` masks) |

## The edits
| edit | what it simulates |
|---|---|
| `img2img_s0.3`, `img2img_s0.5` | "AI filter"/morph bots that re-render the photo with a prompt |
| `inpaint_scene` | keep the person, replace the background ("put her somewhere else") |

Both photos get the **same prompt, seed and settings**; only the input differs. Change prompts with
`--prompt` / `--scene-prompt`, strengths with `--strengths 0.2,0.4,0.6`, quality with `--sd-steps 50`.

## Reading the numbers
- **face PSNR (input → edit)**: how much of the face survived the edit. For the original this should be
  high (the edit worked). For the immunised photo we want it **lower** – that is the protection.
- **drop dB** = original − protected. Larger = stronger protection. ~1 dB is measurable; several dB is
  clearly visible.
- **det**: face-detector confidence on the edited image. A drop means the AI output no longer looks
  like a clean face.
- Also check that the immunised input itself looks unchanged (face PSNR of the input in the header).

## Tuning for a convincing demo
Run a few variants and keep the best trade-off (the `/i3-experiment` command in Claude Code logs them):
| knob | effect |
|---|---|
| `--face-psnr 30 … 34` | lower = stronger protection, more visible |
| `--steps 150 … 300` | more optimisation = stronger, slower |
| `--mode targeted` vs `untargeted` | try both; the best choice can depend on the edit |
| `--surrogate sd-vae` | only the real VAE (slower, most direct for SD) |
| `--strengths` | protection is usually most visible at low/medium img2img strength |

Be honest in the presentation: show the visibility cost next to the effect, report the settings, and
show a case where it does *not* work (e.g. after heavy blur) – examiners trust results that show limits.
