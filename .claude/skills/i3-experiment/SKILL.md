---
name: i3-experiment
description: Run a small, logged parameter sweep for I3 (e.g. eps, face-psnr, payload-alpha, mode, surrogate) and append the results to docs/EXPERIMENTS.md.
argument-hint: "<parameter> <value1,value2,...> [image]"
allowed-tools: Bash(i3 *) Read Edit
---

Sweep one I3 parameter and log the outcome so results are reproducible.

Arguments: `$ARGUMENTS` = `<flag> <comma-separated values> [image]`, e.g.
`--face-psnr 30,32,34,36` or `--payload-alpha 0.004,0.005,0.007`.
Default image: `assets/sample_astronaut.png`.

For each value:
1. `i3 protect <image> -o outputs/sweep/<flag>-<value>.png --owner sweep@example.com <flag> <value>`
2. `i3 evaluate --original <image> --protected outputs/sweep/<flag>-<value>.png --out reports/sweep/<flag>-<value>`
3. Collect: psnr_db, face_psnr_db, payload decoded under whatsapp_like and webp_q75,
   recon_drop_db for identity / whatsapp_like / down_up_0.5, runtime from the protect output.

Then append to `docs/EXPERIMENTS.md` (create it if missing) a section:
- date, git commit (`git log -1 --format=%h`), image, the exact command template
- one markdown table, one row per value
- two or three sentences of interpretation: which value gives the best
  immunity-vs-visibility trade-off and whether any hard constraint was violated.

Do not change source code in this skill. Do not delete earlier experiment sections.
