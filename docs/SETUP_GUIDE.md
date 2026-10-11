# Setup guide — VS Code, Git/GitHub and Claude Code

Written for **Windows 10/11** first (commands for macOS/Linux are given alongside).
Total time: ~30–45 minutes, most of it downloads.

---

## 0. What you need

| | |
|---|---|
| Disk | ~5 GB free (PyTorch is the big one) |
| Python | 3.10 – 3.12 recommended (3.13 also works) |
| Accounts | GitHub (free). Claude **Pro, Max, Team, Enterprise or a Console (API) account** – the free claude.ai plan does not include Claude Code |
| GPU | optional. Everything runs on CPU; a free Colab/Kaggle GPU is useful for big experiments and for training the Phase-2 generator |

> Put the project in a normal folder such as `C:\Projects\`. **Avoid OneDrive/Desktop/Documents** if
> OneDrive sync is on – it locks files inside `.venv` and slows everything down.

---

## 1. Install the tools

1. **Python** – https://www.python.org/downloads/ → run the installer → tick **"Add python.exe to PATH"**.
   Check in a new terminal: `py --version` (Windows) / `python3 --version` (macOS/Linux).
2. **Git** – https://git-scm.com/downloads/win (keep the defaults). This also installs **Git Bash**,
   which Claude Code uses as its shell on Windows. Then, once:
   ```bash
   git config --global user.name  "Your Name"
   git config --global user.email "you@example.com"
   git config --global init.defaultBranch main
   ```
3. **VS Code** – https://code.visualstudio.com/ (version 1.94 or newer is needed for the Claude Code extension).

---

## 2. Open the project in VS Code

1. Unzip `i3-stego-immunization.zip` to `C:\Projects\i3-stego-immunization`.
2. VS Code → **File → Open Folder…** → choose that folder → **Yes, I trust the authors**.
3. A pop-up offers to install the **recommended extensions** (Python, Pylance, Ruff, Claude Code,
   Jupyter, GitLens) – click **Install**. (They are listed in `.vscode/extensions.json`.)

---

## 3. Create the Python environment

Open **Terminal → New Terminal** (PowerShell on Windows) in VS Code:

```powershell
# Windows (PowerShell)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
# if you get "running scripts is disabled":  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip

# PyTorch – the CPU-only build is a few hundred MB instead of several GB:
pip install torch --index-url https://download.pytorch.org/whl/cpu
#   NVIDIA GPU? take the CUDA command from https://pytorch.org/get-started/locally/ instead

pip install -e ".[dev]"              # the i3 package + pytest + ruff
python scripts/download_models.py    # TAESD (5 MB x2) + YuNet face detector
i3 keygen                            # creates keys/  (secret – git-ignored)
python -m pytest -m "not slow"       # should end with "passed"
```

```bash
# macOS / Linux
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch                    # macOS wheels are CPU/MPS already; Linux: add --index-url .../whl/cpu
pip install -e ".[dev]"
python scripts/download_models.py && i3 keygen && python -m pytest -m "not slow"
```

Then tell VS Code to use the venv: **Ctrl+Shift+P → "Python: Select Interpreter" → `.venv`**.
The **Testing** panel (flask icon) now shows all tests; **Run and Debug** (Ctrl+Shift+D) has ready-made
launch configs for protect / verify / evaluate; **Terminal → Run Task…** has demo, tests and lint tasks.

Optional extras:
```bash
python scripts/download_models.py --mediapipe   # 478-landmark face mesh (more precise W_PIR)
pip install -e ".[mediapipe]"
pip install -e ".[sd]"      # real Stable Diffusion VAE surrogate (--surrogate sd-vae), ~330 MB download on first use
```

---

## 4. First run

```bash
i3 protect assets/sample_astronaut.png -o outputs/sample.protected.png --owner you@example.com --debug
i3 verify  outputs/sample.protected.png --owner you@example.com --manifest outputs/sample.protected.i3.json
i3 evaluate --original assets/sample_astronaut.png --protected outputs/sample.protected.png --out reports/sample
```
Open `reports/sample/comparison.png` and `reports/sample/report.md`. On a laptop CPU the protect step takes
a few minutes (use `--steps 20` while experimenting).

**The real-world test (do this!)** – send `outputs/sample.protected.png` to yourself on WhatsApp
(normal quality, not as a document), download it back, then:
```bash
i3 inspect-jpeg path\to\IMG-WA0001.jpg     # real size + estimated JPEG quality of the platform
i3 verify path\to\IMG-WA0001.jpg --owner you@example.com
```
If WhatsApp's quality/size differ from the presets, update `PLATFORM_PRESETS` in `src/i3/attacks.py`
and `immunize.eot.jpeg_quality` in `configs/default.yaml`. Repeat for Instagram and Telegram.

---

## 5. Git and GitHub

### First commit
```bash
git init
git add .
git status          # CHECK: no keys/, no models/*.pth, no outputs/ or reports/ in the list
git commit -m "feat: initial I3 prototype (immunisation + stego payload)"
```

### Put it on GitHub
Easiest: VS Code **Source Control** panel (Ctrl+Shift+G) → **Publish Branch** → choose **private
repository** (recommended: research code + face images). VS Code handles the GitHub sign-in.

Or by hand: create an *empty* repo at https://github.com/new (no README/licence – you already have them), then
```bash
git remote add origin https://github.com/<your-username>/i3-stego-immunization.git
git push -u origin main
```
On Windows, Git Credential Manager opens a browser window to log in the first time.

### Day-to-day workflow
```bash
git switch -c feat/identity-surrogate     # one branch per roadmap item
# ... work, run tests ...
git add -p                                # review what you stage, hunk by hunk
git commit -m "feat(surrogate): add ArcFace identity loss"
git push -u origin feat/identity-surrogate   # then open a Pull Request on GitHub and merge
git switch main && git pull
git tag exp-face34-v1 && git push --tags  # tag commits that produced results you report
```
The included GitHub Actions workflow (`.github/workflows/tests.yml`) runs the fast tests on every push.

### Things that must never go into git
- `keys/` – if a key ever gets committed, treat it as leaked: `i3 keygen --overwrite` (old images then
  become unreadable with the new key – keep a private backup of the old one) and rewrite history.
- model weights (`models/*.pth`, `.onnx`, `.task`) – re-downloadable; for your own trained generator use
  **GitHub Releases** or Git LFS (`git lfs install && git lfs track "*.pt"`).
- photos of real people, `data/`, `outputs/`, `reports/` – privacy (and India's DPDP Act). The
  `.gitignore` already covers all of these.

---

## 6. Claude Code

### Install
Claude Code works as a **VS Code extension** (graphical panel) and as a **CLI**. Install both:

1. **Extension**: VS Code → Extensions (Ctrl+Shift+X) → search **"Claude Code"** (publisher Anthropic) →
   Install. Open it with the **Spark icon** in the editor toolbar (top-right, visible when a file is open)
   or the Activity Bar. Sign in with your Claude account when asked.
2. **CLI** (needed if you want `claude` in a terminal – the extension does not add it to PATH):
   ```powershell
   irm https://claude.ai/install.ps1 | iex                 # Windows PowerShell
   # or: winget install Anthropic.ClaudeCode
   ```
   ```bash
   curl -fsSL https://claude.ai/install.sh | bash          # macOS / Linux / WSL
   ```
   Open a new terminal → `claude --version`, and `claude doctor` if anything looks wrong.

On native Windows, Claude Code uses Git Bash for shell commands when Git for Windows is installed,
otherwise PowerShell.

### What is already set up for you in this repo
- **`CLAUDE.md`** – loaded automatically at the start of every session: what the project is, commands,
  code map, invariants (PSNR ≥ 38 dB, never touch `keys/`, …). Keep it up to date – it is the
  single highest-leverage file for getting good results from Claude Code.
- **`.claude/settings.json`** – pre-approves safe commands (pytest, ruff, `i3`, `git status/diff/log`) and
  blocks reading `keys/`. Your personal overrides go in `.claude/settings.local.json` (git-ignored).
- **Project skills** (type `/` in the chat to see them):
  - `/i3-eval [image]` – protect + evaluate, compare with the previous run, flag regressions.
  - `/i3-experiment --face-psnr 30,32,34,36` – logged parameter sweep → `docs/EXPERIMENTS.md`.
  - `/i3-next-phase P1.2` – implement a roadmap item: plan → approval → code + tests → eval → docs.

### How to work with it effectively
1. **Plan first for anything non-trivial**: type `/plan` (or start with "Plan only, don't edit yet: …").
   Read the plan, push back, then let it implement.
2. **Give it context precisely**: select code in the editor and press **Alt+K** (Option+K on Mac) to insert an
   @-mention like `@src/i3/stego.py#40-80`. **Ctrl+Esc** toggles focus between editor and Claude.
3. **Make it verify**: end prompts with "run the fast tests and `/i3-eval`, show me the numbers".
   Claude Code is much better when it can check its own work against tests and metrics.
4. **Review every diff** before accepting; use the checkpoint/rewind button (hover a message) to undo a
   bad direction. Commit working states often so `git` is your second safety net.
5. **One task per conversation**; start a fresh conversation for an unrelated task so old context doesn't leak.
6. **Ask for explanations** – this is a learning project: "Explain `init_payload_amplitudes` line by line"
   or "Why does the payload break under cropping?" are great prompts for your viva/report.

Copy-paste prompts for every roadmap phase are in **[ROADMAP.md](ROADMAP.md)**.
