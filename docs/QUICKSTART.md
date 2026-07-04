# Quickstart

Two ways to run OTTO: install the pre-built app, or run it from source in your
browser — no Rust required. Either way, you configure your model and API key
from the in-app **Settings**, and you can pull MLX models straight from the
command line if you'd rather skip the On-Device UI.

---

## Option 1 — Install the app from the `.dmg` (fastest, no dev setup)

No Python, Node, or Rust required — this is the quickest way to try OTTO.

1. Grab the latest `OTTO.dmg` from [Releases](https://github.com/ottoappmac/otto/releases/latest).
2. Open the `.dmg` and drag **OTTO** into **Applications**.
3. Launch **OTTO** from Applications (or Spotlight).
   - The app is signed and notarized, so it should open normally. If macOS
     still shows an "unidentified developer" warning, right-click the app →
     **Open** once to bypass it.
4. On first launch, open **Settings** and configure a model — see
   [Configure a model](#configure-a-model) below.

---

## Option 2 — Run from source (no Rust required)

This runs the real UI in your web browser via the Vite dev server, talking to
the FastAPI backend. It needs neither Rust, Xcode, nor the packaged app build,
which makes it ideal for VMs, headless machines, or a quick dev loop.

> You do **not** need to run `./install.sh` first. That script is only for
> building the packaged desktop app (it downloads the Playwright browser and
> compiles a PyInstaller backend binary). To run from source you just need the
> Python venv and the frontend `npm` deps — the two steps below.

### Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Python | 3.12 | `python3 --version` |
| Node.js | 18+ | `node -v` |
| `uv` | any | `uv --version` |

Missing something?

```bash
# uv (Python package/venv manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Node.js — via Homebrew (macOS)
brew install node
```

### Step 1 — Install dependencies

```bash
# Python: creates .venv/ and installs backend deps from the lockfile
uv sync --frozen --python 3.12

# Frontend
npm install --prefix app
```

### Step 2 — Start the backend

```bash
source .venv/bin/activate
PYTHONPATH=src python -m backend --port 18081 --reload
```

### Step 3 — Start the frontend

In a second terminal:

```bash
cd app
npm run dev
```

Open the printed address — usually **http://localhost:5173** — in Safari or
Chrome. You lose the native desktop niceties (system notifications, dock
badge), but the app itself works fully.

You can also reach the backend directly:

| Endpoint | URL |
|----------|-----|
| Health check | http://localhost:18081/api/health |
| REST API | http://localhost:18081/api/ |
| WebSocket | ws://localhost:18081/ws/{session_id} |

### Optional — native desktop app (needs Rust)

Want the native Tauri window instead of the browser? Install Rust and the
Xcode Command Line Tools (the ~1.5 GB CLT package, *not* the full Xcode IDE),
then let `./start_app.sh` handle everything:

```bash
# Rust (required by Tauri)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Xcode Command Line Tools (macOS — required for the Tauri linker)
xcode-select --install

./start_app.sh              # first run (installs deps + launches backend & app)
./start_app.sh --no-install # subsequent runs — skip dependency install
```

Press `Ctrl-C` to stop both services cleanly.

---

## Configure a model

On first launch, open **Settings** and either:

- add a cloud API key (e.g. `ANTHROPIC_API_KEY`), or
- switch to **On-Device (MLX)** and download a model from the catalog.

Keys are stored in the OS keychain — never in config files — so there's no
`.env` to edit. See [`settings.md`](./settings.md) for the full walkthrough, or
use the command-line route below to pre-download an MLX model.

---

## Download MLX models via command line

The On-Device tab in the app downloads models for you, but if you'd rather
skip the UI, `hf` (installed with `huggingface_hub`, already a project
dependency) downloads straight into the **same cache OTTO reads from**
(`~/.cache/huggingface/hub` by default), so a model you pull via the CLI
shows up already-downloaded in the app.

```bash
# One-time setup (skip if huggingface_hub is already installed, e.g. via
# this project's .venv)
uv pip install -U "huggingface_hub[cli]" hf_transfer

# Fastest transfer — enables the Rust-based accelerated downloader
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download a model (this repo_id is from the built-in catalog)
hf download mlx-community/Qwen3-4B-4bit
```

**Skip duplicate weight formats** (`.bin`, `.gguf`, `original/*`, etc.) the
way the app's own downloader does, to save bandwidth and disk:

```bash
hf download mlx-community/Qwen3-4B-4bit \
  --include "*.safetensors" "*.json" "tokenizer*" "*.txt" "*.model"
```

**Gated models** (a few repos require accepting terms on Hugging Face first):

```bash
hf auth login          # or: export HF_TOKEN=hf_...
```

**Other model IDs worth trying** (see `backend/mlx_catalog.py` for the full
curated list, or search [huggingface.co/mlx-community](https://huggingface.co/mlx-community)):

| Size class | `repo_id` |
|---|---|
| Small / fast | `mlx-community/Qwen3-1.7B-4bit` |
| Balanced | `mlx-community/Qwen3-8B-4bit` |
| Vision-language | `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` |
| Power (32 GB+ Macs) | `mlx-community/Qwen3-30B-A3B-4bit` |

Once downloaded, open OTTO's On-Device tab and select the model — no
re-download needed.

---

## Optional — Browser automation

The `playwright_mcp` tool requires a running Playwright MCP server. Start it in
a separate terminal before launching the app:

```bash
npx -y @playwright/mcp@latest --port 8931
```

The defaults (`localhost:8931`, non-headless) match the command above. To point
at a different host/port or run headless, export the corresponding variables
before starting the backend:

```bash
export PLAYWRIGHT_MCP_HOST=localhost
export PLAYWRIGHT_MCP_PORT=8931
export PLAYWRIGHT_MCP_HEADLESS=false
```

---

## Troubleshooting

**Port already in use**

The backend uses port 18081 and Vite uses 5173. To free a port:

```bash
lsof -ti :18081 | xargs kill   # backend port
lsof -ti :5173  | xargs kill   # Vite frontend port
```

**Backend crashes on startup**

The backend starts fine without any keys — you configure a model afterward in
**Settings**. If it crashes, check the terminal output; the most common causes
are a port already in use or a missing Python dependency (re-run
`uv sync --frozen`).

**`uv` not found**

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Or re-run the installer: `curl -LsSf https://astral.sh/uv/install.sh | sh`

**Activate the venv manually**

```bash
source .venv/bin/activate
```

---

## Run tests

```bash
source .venv/bin/activate
PYTHONPATH=src pytest -v tests/
```
