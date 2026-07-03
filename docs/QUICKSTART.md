# Quickstart

Get OTTO running fast: install the pre-built app, or run it from source with
or without Rust — then grab MLX models from Hugging Face straight from the
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
4. On first launch, open **Settings** and either:
   - add a cloud API key (`ANTHROPIC_API_KEY` or similar), or
   - switch to **On-Device (MLX)** and download a model from the catalog —
     see [`settings.md`](./settings.md), or use the command-line route in
     [Download MLX models via command line](#download-mlx-models-via-command-line) below.

Skip to [Troubleshooting](#troubleshooting) or [Run tests](#run-tests) if
that's all you need — everything below is for running from source instead.

---

## Option 2 — Run from source

### Prerequisites

| Tool | Version | Check | Needed for |
|------|---------|-------|-------------|
| Python | 3.12 | `python3 --version` | both |
| Node.js | 18+ | `node -v` | both |
| `uv` | any | `uv --version` | both |
| Rust | stable | `rustc --version` | native desktop app only (not needed for the browser workflow) |

These are installed automatically by `./install.sh` and `./start_app.sh` if
missing. Missing something manually?

```bash
# uv (Python package/venv manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Rust (required by Tauri — native desktop app only)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Node.js — via Homebrew (macOS)
brew install node
```

**macOS, native desktop app only**: also install the Xcode Command Line
Tools (the ~1.5 GB CLT package, *not* the full Xcode IDE) — required for the
Rust/Tauri linker:

```bash
xcode-select --install
```

If you're short on disk space or don't want the Command Line Tools, use the
browser workflow below instead — it needs neither Rust nor Xcode.

### Step 1 — Add your API key

Copy the environment template and set at least one LLM key:

```bash
cp .env.template .env
```

Open `.env` and fill in one of:

```bash
ANTHROPIC_API_KEY=sk-ant-...
# or
COHERE_API_KEY=...
```

> Running fully on-device with MLX? Set `LLM_PROVIDER=mlx` instead — no API key needed.

### Step 2 — Start everything (native desktop app, with Rust)

```bash
./start_app.sh              # first run (installs deps too)
./start_app.sh --no-install # subsequent runs — skip dependency install
```

This will:
1. Create a Python 3.12 virtual environment at `.venv/` (first run only)
2. Install Python dependencies (`uv pip install -e .`)
3. Install frontend dependencies (`npm install`)
4. Start the FastAPI backend on **port 18081**
5. Start the Tauri desktop app (Vite dev server on **port 5173**)

Press `Ctrl-C` to stop both services cleanly.

If you'd rather run the pieces yourself, or don't have Rust installed, pick
one of the two manual workflows below.

#### Manual — With Rust (native desktop app)

```bash
# Terminal 1 — FastAPI backend
source .venv/bin/activate
PYTHONPATH=src python -m backend --port 18081 --reload

# Terminal 2 — Tauri desktop app
cd app
npm install        # first run only
npm run tauri dev
```

#### Manual — Without Rust (browser only)

Best for VMs, headless machines, or anywhere you don't want to install the
Command Line Tools. Runs the same UI in your web browser via the Vite dev
server.

```bash
# Terminal 1 — FastAPI backend
source .venv/bin/activate
PYTHONPATH=src python -m backend --port 18081 --reload

# Terminal 2 — frontend dev server
cd app
npm install        # first run only
npm run dev
```

Open the printed address — usually **http://localhost:5173** — in Safari or
Chrome. No Rust, no Command Line Tools, no Xcode required. You lose the
native desktop niceties (system notifications, dock badge), but the app
itself works fully.

### Step 3 — Use the app

The Tauri desktop window opens automatically (or your browser, for the
manual/no-Rust workflow). You can also reach the backend directly:

| Endpoint | URL |
|----------|-----|
| Health check | http://localhost:18081/api/health |
| REST API | http://localhost:18081/api/ |
| WebSocket | ws://localhost:18081/ws/{session_id} |

### Optional — Browser automation

The `playwright_mcp` tool requires a running Playwright MCP server. Start it in a separate terminal before launching the app:

```bash
npx -y @playwright/mcp@latest --port 8931
```

Configure via `.env`:

```bash
PLAYWRIGHT_MCP_HOST=localhost
PLAYWRIGHT_MCP_PORT=8931
PLAYWRIGHT_MCP_HEADLESS=false
```

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

Once downloaded, open OTTO's On-Device tab (or set `LLM_PROVIDER=mlx` and
`HF_LLM_MODEL_ID=<repo_id>` in `.env`) — no re-download needed.

---

## Troubleshooting

**Port already in use**

`start_app.sh` checks ports 18081 and 5173 before starting and will print which process holds them. To free a port manually:

```bash
lsof -ti :18081 | xargs kill   # backend port
lsof -ti :5173  | xargs kill   # Vite frontend port
```

**Backend crashes on startup**

Check that `.env` exists and has a valid API key set. The most common cause is a missing or malformed key.

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
