---
name: i3-next-phase
description: Implement one roadmap item from docs/ROADMAP.md (plan first, then code + tests + docs), keeping all I3 invariants. Use when the user says "next phase" or names a roadmap item.
argument-hint: "[roadmap item id, e.g. P1.2]"
---

Implement roadmap item `$ARGUMENTS` (or, if empty, the first unchecked item) from `docs/ROADMAP.md`.

1. Read `CLAUDE.md`, `docs/ARCHITECTURE.md` and the roadmap item. Read every source file you will touch.
2. Write a short plan (files to change, new tests, how you will measure success) and
   wait for the user to approve it before editing.
3. Implement in small steps. Add or update tests in `tests/` for every behaviour change;
   tests must not download anything (use `--surrogate random` / skip-if-missing for models).
4. Run `python -m pytest -m "not slow"` and `ruff check src tests scripts` until clean.
5. If the change touches the optimiser, payload or masks, run the `i3-eval` skill and
   include its table in your summary.
6. Update `docs/ROADMAP.md` (tick the item, note the measured result) and any doc whose
   behaviour description changed.
7. Propose a commit message in Conventional Commits style (`feat(stego): ...`). Do not push.

Invariants you must not break: PSNR >= 38 dB after 8-bit rounding; payload decodes after
`whatsapp_like`; secrets in `keys/` are never read, printed or committed; no model weights
or face images in git.
