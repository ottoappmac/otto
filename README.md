# OTTO

[![OTTO intro video](https://img.youtube.com/vi/G0XRYghuWNc/maxresdefault.jpg)](https://youtu.be/G0XRYghuWNc)

A macOS AI agent desktop app. FastAPI backend, Tauri + React frontend, LangGraph orchestration. Runs entirely on your machine — no cloud relay, no telemetry.

Supports cloud LLMs (Anthropic, OpenAI) and fully local inference via MLX on [Apple Silicon](https://mlx-framework.org/#features), [oMLX](https://github.com/jundot/omlx), or an [exo](https://github.com/exo-explore/exo) cluster of Apple Silicon nodes. The agent can browse the web, automate your Mac desktop, read documents, query SEC filings, and build new MCP-backed tools for itself at runtime.

OTTO is self-managing: through conversation alone it creates and maintains its own **Agents**, **Skills**, **Tools**, **Schedules**, **Triggers**, and **Settings** — no config files or UI required.

## Documentation

Per-page guides for the desktop UI (each with screenshots). The pages below map to the items in OTTO's right-hand navigation:

| Page | Guide | What it covers |
|------|-------|----------------|
| Dashboard | [`dashboard.md`](./docs/dashboard.md) | Live overview — KPIs, running activity, charts, and breakdowns |
| Runs | [`runs.md`](./docs/runs.md) | Run history, filters, and the per-run detail tabs (Timeline, Graph, Results, Files, Metrics, Evaluation) |
| Chat | [`chat.md`](./docs/chat.md) | Talking to the agent — composer, model picker, live runs, steering |
| Suggestions | [`suggestions.md`](./docs/suggestions.md) | The ambient suggestions inbox |
| Agents | [`agents.md`](./docs/agents.md) | Managing agents, skills, and tools |
| Schedules | [`schedules.md`](./docs/schedules.md) | Cron-based automated runs |
| Triggers | [`triggers.md`](./docs/triggers.md) | Event-driven runs (file changes, AppleScript, macOS events) |
| Activity | [`activity.md`](./docs/activity.md) | The on-device macOS activity timeline |
| Settings | [`settings.md`](./docs/settings.md) | LLM, Agent Memory, Voice, Privacy, and all other settings |

Also in [`docs/`](./docs/): [`QUICKSTART.md`](./docs/QUICKSTART.md) (install the `.dmg`, run from source with/without Rust, download MLX models via the CLI) and [`features.md`](./docs/features.md) (a longer, screenshot-heavy feature tour).

## Services

| Service | Port | Description |
|---------|------|-------------|
| `backend/` (FastAPI + WebSocket) | 18081 | Local backend the Tauri app and any HTTP/WS client talks to |
| `app/` (Tauri + React) | dev: Vite | Desktop UI; production build via `npm run tauri build` |
| Playwright MCP | 8931 | Browser automation subprocess (started separately) |

## Features

### DeepAgent orchestrator

The `DeepAgent` (`src/deep_agent/`) is a LangGraph ReAct graph that wires up a configurable set of tools and subagents. It runs inside the backend as the engine behind every chat session.

**Built-in tools**

| Tool | Description |
|------|-------------|
| `wikipedia` | Wikipedia summary lookup |
| `duckduckgo` | DuckDuckGo web search — no API key needed |
| `web_researcher` | Search + full-page extraction, returns structured markdown |
| `doc_researcher` | BM25 keyword ranking over uploaded documents — no embeddings |
| `doc_reader` | LLM-based document summarisation and Q&A |
| `playwright_mcp` | Browser automation via Playwright MCP (accessibility snapshots, not screenshots) |
| `view_image` | Let the agent see uploaded images (PNG, JPEG, GIF, WebP) |
| `execute` | Run shell commands in a per-session sandbox with `SESSION_FILES` anchoring |
| `ask_user` | Interrupt execution and ask the user a free-text or multiple-choice question |
| `spawn_followup_session` | Hand off to a fresh session after building new tools mid-turn |
| `get_settings` / `set_*` | Read and update app config from inside an agent turn |

**Subagents** (delegated via `task` tool)

| Subagent | Description |
|----------|-------------|
| `web-voyager` | Autonomous web navigation agent — plans, navigates, extracts |
| `computer-voyager` | macOS desktop computer-use agent via the Accessibility API |

Subagent tool calls and results stream back to the WebSocket in real time so you see progress, not a spinner.

### Agent self-administration

The agent has a dedicated toolset for managing the application itself. You can ask it to set up automation, adjust settings, or wire new integrations entirely through chat — no config file or UI required.

**Settings**

| Tool | What it does |
|------|-------------|
| `get_settings` | Read the current config (credential-free view) |
| `set_llm_provider` | Switch the LLM provider and model family |
| `set_memory_config` | Adjust memory consolidation settings |
| `update_activity_settings` | Enable / disable the activity tracker, change poll interval, retention, or excluded apps |
| `toggle_mcp_server` | Connect or disconnect an MCP server for the current session |

Changes take effect immediately for subsequent sessions; `spawn_followup_session` can be used to pick them up in the same conversation.

**Schedules** — the agent can manage cron-based automated runs:

| Tool | What it does |
|------|-------------|
| `list_schedules` | Show all schedules with cron, agent, last-run status |
| `create_schedule` | Create a new cron schedule with a prompt and agent |
| `update_schedule` | Change the prompt, cron expression, or agent |
| `toggle_schedule` | Enable or pause a schedule |
| `delete_schedule` | Remove a schedule |
| `run_schedule_now` | Fire a schedule immediately (outside its cron window) |

Example: *"Create a schedule that runs every weekday at 9am, uses the web-researcher agent, and summarises the top AI news into my memory."*

**Triggers** — the agent can set up event-driven runs that fire when something changes:

| Tool | What it does |
|------|-------------|
| `list_triggers` | Show all triggers with type, poll interval, and last-fire status |
| `create_trigger` | Create a filesystem or macOS osascript trigger |
| `update_trigger` | Change the watched path, prompt, or poll interval |
| `toggle_trigger` | Enable or pause a trigger |
| `delete_trigger` | Remove a trigger |
| `run_trigger_now` | Fire a trigger immediately for testing |

Trigger types: `fileos` (watch a path for `mtime` / `size` / `exists` / `new_files` changes) and `macostool` (poll an AppleScript snippet and react when output changes). Privileged types (`http`, `git`, `shell`) are restricted to the dedicated `trigger-builder-agent`.

Example: *"Watch ~/Downloads for new PDFs and, when one appears, extract its title and first paragraph into my notes."*

**Agents & Skills** — the agent can author new agents and skills that persist in the library:

| Tool | What it does |
|------|-------------|
| `list_agents` / `list_skills` | Browse the current agent and skill library |
| `create_agent` / `update_agent` | Save a new agent definition (name, system prompt, tools, LLM) |
| `create_skill` / `update_skill` | Save a reusable skill (a named system-prompt fragment) |
| `delete_agent` / `delete_skill` | Remove a definition |
| `discover_available_tools` | Inspect all connected MCP tools and their parameter schemas |

Newly saved agents become available in the UI immediately — active sessions pick up the change without a restart.

**MCP servers** — the agent can author, provision, and connect new tool servers at runtime:

| Tool | What it does |
|------|-------------|
| `list_allowed_mcp_imports` | Show which third-party packages generated servers may use |
| `list_my_mcp_servers` | List agent-authored servers with their tools and credential names |
| `create_mcp_server` | Generate a new FastMCP server, provision its venv, register it |
| `delete_mcp_server` | Remove a generated server and its venv |
| `connect_mcp_server` | Start the subprocess for a registered server |
| `request_credential` | Prompt the user to supply a vault credential via the UI dialog |
| `is_credential_set` | Check whether a named credential has been stored |

Generated code is audited before provisioning: `ast` rejects forbidden imports and literal credential strings in the source. All secrets are stored in the OS keychain and only injected into the subprocess environment at spawn time — the LLM never sees the values.

Example: *"Build me a Stripe MCP that can list recent charges and create payment links."* The agent writes the server code, provisions a venv, registers it, then asks you to supply the API key through the vault dialog before connecting.

### MCP ecosystem

Any MCP server — stdio or SSE — can be added through the Settings UI or REST API. The backend:

- spawns stdio servers as supervised child processes with automatic restart on crash
- hydrates environment variables from the OS keychain at spawn time so secrets never appear in config files
- wraps every tool call with a loop-guard that detects identical-args failure loops and injects a recovery hint
- scrubs MCP tool results for known credential shapes (Stripe keys, Slack tokens, Discord bot tokens, GitHub PATs, AWS, OpenAI, Anthropic, and more) before they reach the LLM context

**MCP authentication** — four flows are supported per-server: static API key (vault-stored), OAuth device flow, OAuth auth-code flow, and browser-capture (for services that don't expose a machine-to-machine flow).

**Runtime MCP generation** — covered under [Agent self-administration](#agent-self-administration) above. The agent can author, provision, and register a new MCP server mid-conversation using the `create_mcp_server` tool.

**Built-in MCPs** — shipped with the app, provisioned automatically on startup:

| MCP | Tools |
|-----|-------|
| `edgar-sec` | `search_filings`, `get_company_submissions`, `search_company_by_ticker`, `get_company_facts`, `get_filing_document`, `get_xbrl_frames` — full read access to 18M+ SEC EDGAR filings |
| `macos-osascript` | Execute AppleScript snippets for macOS automation |
| `slack` | `list_channels`, `get_channel_history`, `get_thread_replies`, `send_message`, `add_reaction`, `list_users`, and more — read/write access to a Slack workspace via a bot token |
| `discord` | `list_guilds`, `list_channels`, `get_channel_messages`, `send_message`, `add_reaction`, `list_guild_members`, and more — read/write access to a Discord server via a bot token |
| `microsoft-teams` | `list_teams`, `list_channels`, `list_channel_members`, `get_channel_messages`, `list_users`, and more — **read-only** access to Microsoft Teams via Graph app-only auth (sending messages requires delegated auth, which this MCP doesn't implement) |

### Agent & Skill library

Reusable agent and skill definitions are stored as JSON under app-data. The library ships built-in definitions (including `macos-desktop-agent` and `macos-applescript-agent` on macOS) and lets you create, edit, and delete your own. Agents reference a skill, an LLM provider, and a tool/subagent selection — switching between them in the UI rebuilds the graph for the next session.

### Memory

A background consolidation pipeline distils session transcripts into durable topic files under `<app-data>/memory/`:

1. **Orient** — read the `MEMORY.md` index and topic frontmatter
2. **Gather** — collect candidate transcript JSONL files
3. **Consolidate** — LLM call produces structured memory diffs
4. **Prune** — enforce per-topic and index size limits

Run status is exposed via `GET /api/memory/status` so the UI can show a live progress indicator.

### Schedules

Cron-based scheduled agent runs backed by APScheduler with JSON file persistence (no extra database). Create up to five schedules, each with a cron expression, an agent, and a prompt. Run history is kept per-schedule. Schedules survive backend restarts.

### Triggers

Event-driven agent runs that fire when something on your machine changes:

| Trigger type | Watches |
|---|---|
| `fileos` | Filesystem path — modes: `mtime`, `size`, `exists`, `new_files` |
| `macostool` | Runs an AppleScript snippet on a timer, fires when output changes (optional regex gate) |

Built-in triggers (opt-in, all disabled by default): new download, new screenshot, Mail unread count change. Per-trigger watermark state persists across restarts so events don't re-fire.

### Activity tracker (macOS)

An opt-in, screenshot-free local activity timeline. A background loop polls the foreground macOS application every N seconds and records `(app, window title, browser URL, active document)` into a local SQLite database with FTS5 full-text indexing. Nothing leaves the device. Deduplicates consecutive identical rows into a single span with a running duration. Accessible via the Activity page in the UI and queryable via `GET /api/activity`.

### On-device inference

**MLX (Apple Silicon)** — run quantised language models and vision-language models locally using the MLX framework. The backend ships a curated model catalog with hardware-aware fit scoring: each model's weight footprint, KV-cache estimate, and architecture metadata are compared against your actual RAM and free disk, and labelled comfortable / tight / over. Download and switch models from the On-Device tab without restarting the backend. Optional turbo levels (`basic` → `cache` → `ssd`) layer oMLX-derived KV-cache optimisations on top of the base MLX path.

> **Note:** on-device prompting and tool-calling are currently tuned and tested primarily against **Qwen** models — you'll get the most reliable agent behaviour with **Qwen 3.5** or **Qwen 3.6**. Other families in the catalog (Llama, Gemma, Phi-4, Mistral, DeepSeek-R1) work but are less battle-tested for tool-heavy agent runs.

**oMLX** — an external OpenAI-compatible local inference server (installed separately via Homebrew or `.dmg`). Set `LLM_PROVIDER=omlx` and point `OMLX_BASE_URL` at the running server. Shares the same guard stack as other providers (RepetitionGuard, ToolCallBudget, etc.) and applies `LLM_FREQUENCY_PENALTY` to discourage repetition at the API level.

**exo cluster** — pool multiple Apple Silicon machines into a single inference cluster. The backend can provision remote nodes over SSH, start and stop the local exo process, and score the model catalog against the live cluster topology. Configure the cluster from the Exo tab in the UI.

### Safety

Multiple runtime guards run on every agent turn, independent of the prompt. They are layered so each catches what the others miss:

**Execution guards** (applied at the tool layer):

- **Execute path safety** — rewrites bare `/output/...` paths in `execute` calls to `$SESSION_FILES/...`
- **Subagent-as-tool guard** — rewrites stray direct tool calls to subagent names into the correct `task(...)` shape, catching the most common dispatch error on smaller models
- **High-risk command flagging** — screens `execute` calls for known-dangerous patterns (`rm -rf /`, `git push --force`, raw block-device writes, etc.) and adds a `high_risk=True` marker so the frontend renders a high-prominence approval badge before the human-in-the-loop interrupt fires
- **Tool loop guard** — per-tool ring-buffer that detects identical-args failure or no-progress loops and injects a recovery hint; on local MLX models it also requests a one-shot temperature bump to break out of greedy-decoding loops

**Runaway-run guards** (applied at the model-call layer, provider-agnostic):

- **Repeated-thought guard** — catches a model that re-emits the same thought + action signature across consecutive turns; nudges with a corrective message and temperature bump, then aborts the run gracefully if the loop continues (configurable via `REPEAT_GUARD_NUDGE_AT` / `REPEAT_GUARD_ABORT_AT` / `REPEAT_GUARD_MAX_PERIOD`)
- **Repetition guard** — catches a single model generation that collapses into repeating the same sentence over and over until the token cap (observed on oMLX / exo / OpenAI-compatible clients); replaces the degenerate blob with a compact recovery message so the turn ends cleanly
- **Tool-call budget** — per-run ceiling on the total number of tool calls since the last user message: at the **soft budget** (default 80) the model receives a one-shot nudge to converge; at the **hard budget** (default 150) the run ends gracefully with whatever the agent has gathered, rather than burning the full recursion budget on non-identical churn (configurable via `TOOL_CALL_SOFT_BUDGET` / `TOOL_CALL_HARD_BUDGET`)

---

## Getting started

### Prerequisites

- Python 3.12
- Node.js 18+
- `uv`

These are installed automatically by `install.sh` (and by `start_app.sh`) if missing.

**For the Tauri desktop app only** (not needed for the browser workflow below):

- Rust + cargo — installed automatically by the scripts
- **macOS: Xcode Command Line Tools** — required for the Rust/Tauri linker. Install once with:
  ```bash
  xcode-select --install
  ```
  This is the small (~1.5 GB) Command Line Tools package, *not* the full Xcode IDE. If `cargo build` fails with a linker error, this is usually why. If you're short on disk space, use the browser workflow below instead — it needs neither Rust nor Xcode.

### Installation

**macOS / Linux** (Debian/Ubuntu, Fedora/RHEL, Arch):
```bash
./install.sh
```
Uses your system package manager (Homebrew / apt / dnf / pacman) when available, but never requires one — `uv` and Node.js both fall back to official, brew-free installers on a clean machine.

**Windows**: not natively supported yet — use WSL and run `./install.sh` from there, or follow the manual steps below.

**Manual** (any platform):
```bash
# Install uv first: https://docs.astral.sh/uv/getting-started/installation/
uv venv .venv --python 3.12
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
uv pip install -e .
cp .env.template .env
```

Edit `.env` and set at least one of `ANTHROPIC_API_KEY` or `COHERE_API_KEY`.

**Optional system dep** (LangGraph PNG graph output — `install.sh` installs this automatically when a supported package manager is present):
```bash
brew install graphviz     # macOS
sudo apt install graphviz # Debian/Ubuntu
```

### Playwright MCP

Required for browser automation tools. Start it before launching the app or using any `playwright_mcp` tool:

```bash
npx -y @playwright/mcp@latest --port 8931
```

Configure via `.env`: `PLAYWRIGHT_MCP_HOST`, `PLAYWRIGHT_MCP_PORT`, `PLAYWRIGHT_MCP_HEADLESS`.

---

## Running locally

**Quickest way** — one command bootstraps the venv, installs dependencies, and starts both the backend and frontend:

```bash
./start_app.sh              # first run (installs deps too)
./start_app.sh --no-install # subsequent runs — skip dependency install
```

Press `Ctrl-C` to stop both services.

If you'd rather run the pieces yourself, pick one of the two workflows below.

### Option A — Browser (no Rust or Xcode needed)

Best for VMs, headless machines, or anywhere you don't want to install the Command Line Tools. Runs the same UI in your web browser via the Vite dev server.

```bash
# Terminal 1 — FastAPI backend (port 18081, auto-reload on code changes)
source .venv/bin/activate
PYTHONPATH=src python -m backend --port 18081 --reload

# Terminal 2 — frontend dev server (in a browser)
cd app
npm install        # first run only
npm run dev
```

Then open the address it prints — usually **http://localhost:5173** — in Safari/Chrome. No Rust, no Command Line Tools, no Xcode required. You lose the native desktop niceties (system notifications, dock badge), but the app itself works fully.

### Option B — Tauri desktop app

The full native shell. Requires Rust + cargo and, on macOS, the Xcode Command Line Tools (see [Prerequisites](#prerequisites)).

```bash
# Terminal 1 — FastAPI backend
source .venv/bin/activate
PYTHONPATH=src python -m backend --port 18081 --reload

# Terminal 2 — Tauri desktop app (Vite dev server + native shell)
cd app
npm install        # first run only
npm run tauri dev
```

Or use VS Code / Cursor Run & Debug (`Cmd+Shift+D`):

- **Run Backend** — starts the FastAPI server on port 18081
- **Run Tests** — runs the pytest suite

The backend is also accessible directly as a REST + WebSocket API — useful for scripting or calling from external tools without the UI.

---

## Building locally

### Python backend (PyInstaller)

```bash
# Install build deps into the active venv
uv pip install pyinstaller
uv pip install -e ".[app]"

# Bundle the backend into a self-contained onedir executable
python scripts/build_backend.py
```

Output lands in `app/src-tauri/resources/backend/`. Tauri bundles it into the `.app` at build time.

### Desktop app (Tauri)

```bash
cd app
npm ci
npm run tauri build
```

The `.app` and `.dmg` are written to `app/src-tauri/target/release/bundle/`.

For Apple code-signing and notarisation, set the environment variables listed in `scripts/sign_and_notarize.sh` and run that script after the Tauri build, or let the GitHub Actions workflow handle it (see below).

---

## Building in GitHub Actions

Three workflows live in `.github/workflows/`:

### `ci.yml` — runs on every PR and push to `main`

- **Python lint** (`flake8`) on the core source directories
- **Frontend type-check + build** (`tsc` + `vite build`) on the React app

### `build-app.yml` — builds the signed macOS app

Triggered on version tags (`v*`), pull requests that touch app/backend/src code, or manually via `workflow_dispatch`.

Steps:
1. Stamp the version from the git tag into `tauri.conf.json`, `package.json`, and `pyproject.toml`
2. Install Python 3.12 + `uv`, install all Python deps
3. Run `scripts/build_backend.py` to produce the PyInstaller bundle
4. Install Node 20 and `npm ci`
5. Install Rust stable with `sccache` for incremental compilation
6. Import the Apple Developer certificate into a temporary keychain
7. Build the Tauri app (unsigned — avoids Python.framework signature invalidation during bundling)
8. Sign all Mach-O binaries inside the bundled `.app`, with special handling to fix PyInstaller's non-standard `Python.framework` symlink layout before `codesign` sees it
9. Notarise the `.app` with `xcrun notarytool`, then staple the ticket
10. Recreate the `.dmg` from the stapled `.app` with `create-dmg`, sign it, and notarise it
11. Upload build artifacts (`.dmg`, `.app.tar.gz`) with a 7-day retention window

Required GitHub secrets: `APPLE_CERTIFICATE`, `APPLE_CERTIFICATE_PASSWORD`, `APPLE_SIGNING_IDENTITY`, `APPLE_ID`, `APPLE_PASSWORD`, `APPLE_TEAM_ID`.

### `release.yml` — semantic versioning and release triggering

Runs on pushes to `main`. Uses `semantic-release` to derive the next version from conventional commits, create a GitHub release, and then dispatch `build-app.yml` at the new tag. The `publish` job in `build-app.yml` attaches the `.dmg` and `.app.tar.gz` to the release.

---

## Environment variables

See `.env.template` for the full list. Key variables:

```bash
# LLM provider
LLM_PROVIDER=anthropic             # anthropic | cohere | openai | mlx | omlx | exo

# API keys — set at least one cloud provider
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL_NAME=claude-sonnet-4-6
COHERE_API_KEY=your_key_here

# OpenAI (native API or Azure OpenAI)
OPENAI_API_KEY=your_key_here
OPENAI_MODEL_NAME=gpt-4o
OPENAI_MODEL_PROVIDER=openai       # openai | azure
OPENAI_AZURE_ENDPOINT=             # https://<resource>.openai.azure.com (Azure only)
OPENAI_AZURE_API_VERSION=2024-12-01-preview
OPENAI_AZURE_DEPLOYMENT=           # deployment name (Azure only; defaults to model name)

# Application
ENVIRONMENT_TYPE=local             # local skips auth middleware
LOG_LEVEL=INFO

# MLX local inference (Apple Silicon, in-process)
HF_LLM_MODEL_ID=mlx-community/quantized-gemma-2b-it
HF_VLM_MODEL_ID=mlx-community/Qwen2.5-VL-7B-Instruct-4bit
HF_DRAFT_LLM_MODEL_ID=            # optional draft model for speculative decoding
MLX_MAX_TOKENS=8192
MLX_TEMP=0.0
MLX_PROMPT_CACHE=false             # enable KV prefix cache across turns
MLX_TURBO_LEVEL=off                # off | basic | cache | ssd
HF_TOKEN=                          # optional, for gated HuggingFace models

# oMLX local inference (external OpenAI-compatible server, Homebrew/.dmg install)
OMLX_BASE_URL=http://127.0.0.1:8000
OMLX_MODEL_NAME=                   # model served by the local oMLX process

# exo cluster inference (OpenAI-compatible API across Apple Silicon nodes)
EXO_BASE_URL=http://127.0.0.1:52415
EXO_MODEL_NAME=                    # e.g. mlx-community/Qwen3.5-9B-4bit

# Repetition discouragement for OpenAI-compatible clients (oMLX / exo)
LLM_FREQUENCY_PENALTY=0.3          # 0.0 disables
LLM_PRESENCE_PENALTY=0.0

# DeepAgent orchestrator
DEEP_AGENT_LLM_PROVIDER=anthropic  # leave blank to inherit LLM_PROVIDER
LOCAL_PROMPT_MODE=auto             # auto | full | lite (orchestrator prompt length)

# Per-run tool-call budgets (ToolCallBudgetMiddleware)
TOOL_CALL_SOFT_BUDGET=80           # nudge to converge at this count; 0 disables
TOOL_CALL_HARD_BUDGET=150          # end the run gracefully at this count; 0 disables

# Tool loop guard (identical-args / no-progress detection)
LOOP_GUARD_WINDOW=8
LOOP_GUARD_MAX_NO_PROGRESS=4
LOOP_GUARD_MAX_SUCCESS=3
LOOP_GUARD_MAX_ESCALATIONS=6       # 0 = only emit corrective messages, never hard-stop

# Repeated-thought guard (cross-turn identical thought + action)
REPEAT_GUARD_NUDGE_AT=3
REPEAT_GUARD_ABORT_AT=10
REPEAT_GUARD_MAX_PERIOD=4          # longest repeating cycle to scan for (1 = strict consecutive)

# Playwright MCP (browser automation)
PLAYWRIGHT_MCP_HOST=localhost
PLAYWRIGHT_MCP_PORT=8931
PLAYWRIGHT_MCP_HEADLESS=false

# LangSmith tracing (optional)
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=your_key_here
```

---

## Tests

```bash
PYTHONPATH=src pytest -v tests/
```

---

## Project structure

```
agents/
├── install.sh                 # macOS + Linux installer (brew-free fallbacks)
├── pyproject.toml             # Project metadata and dependencies (Hatchling)
├── .env.template              # All supported environment variables
├── app/                       # Tauri desktop app
│   ├── src/
│   │   ├── pages/             # Chat, History, Agents, Memory, Schedules,
│   │   │                      #   Triggers, Activity, Tools, Settings,
│   │   │                      #   MLX (on-device), Exo (cluster)
│   │   └── components/        # Layout, Sidebar, chat/, exo/, mlx/
│   └── src-tauri/             # Rust shell + Tauri config
├── backend/                   # FastAPI + WebSocket backend (port 18081)
│   ├── routes/                # REST endpoints: sessions, agents, mcp,
│   │   │                      #   memory, schedules, triggers, hooks,
│   │   │                      #   vault, activity, mlx, exo, settings
│   ├── auth/                  # static, OAuth device, OAuth auth-code,
│   │                          #   browser-capture auth flows
│   ├── builtin_mcps/          # edgar_sec, macos_osascript, slack, discord, microsoft_teams
│   ├── agent_library.py       # Agent + Skill CRUD
│   ├── mcp_manager.py         # Config-driven MCP connection manager
│   ├── mcp_builder.py         # Runtime MCP generation + venv provisioning
│   ├── credential_vault.py    # OS keychain wrapper (never logs values)
│   ├── output_redactor.py     # Credential scrubber for MCP tool results
│   ├── safety_middleware.py   # Execute path, subagent-as-tool, high-risk guards
│   ├── memory.py              # Background memory consolidation pipeline
│   ├── scheduler.py           # APScheduler cron-based scheduled runs
│   ├── trigger_manager.py     # Filesystem + osascript event triggers
│   ├── activity_tracker.py    # macOS local SQLite activity timeline
│   ├── session_manager.py     # LangGraph session lifecycle + checkpointing
│   ├── streaming_subagent.py  # Real-time subagent event relay to WebSocket
│   ├── spawn_tools.py         # spawn_followup_session tool
│   ├── ask_user_tools.py      # ask_user interrupt tool
│   ├── file_tools.py          # Image upload/view tools
│   ├── settings_tools.py      # Agent-facing config read/write tools
│   ├── mlx_catalog.py         # Curated MLX models + hardware fit scoring
│   ├── exo_catalog.py         # exo cluster model catalog + fit scoring
│   └── exo_provisioner.py     # exo cluster provisioning / management
├── scripts/
│   ├── build_backend.py       # PyInstaller bundle builder
│   ├── build_mac.sh           # macOS build helper
│   ├── sign_and_notarize.sh   # Apple code-signing + notarisation
│   ├── exo_cli.py             # Manage local exo cluster (stdlib-only)
│   └── bench_*.py             # Performance benchmarks
├── src/
│   ├── agents/                # web_voyager, computer_voyager subagents
│   ├── deep_agent/            # DeepAgent graph, options, tool/subagent factories
│   ├── chat_models/           # MLX text + VLM chat model wrappers
│   ├── callbacks/             # LangChain callbacks
│   ├── loaders/               # Document loaders
│   ├── middleware/            # Playwright pruning, MLX ReAct middleware,
│   │                          #   repetition_guard, tool_call_budget,
│   │                          #   repeated_thought_guard, context_truncation
│   └── tools/
│       ├── research/          # wikipedia, duckduckgo, web_researcher,
│       │                      #   doc_researcher, doc_reader
│       ├── navigation/
│       │   ├── web/           # Playwright MCP client + browser navigator
│       │   └── computer/      # macOS Accessibility navigator
│       ├── anthropic/         # Computer-use schema helpers
│       ├── evaluation/        # deepeval helpers
│       └── transcripts/       # Remote transcript fetchers + hook event buffer
├── tests/
├── doc/examples/              # Jupyter notebook examples
│   ├── deep_agent_orchestration/  # Streaming, browser use, config verification
│   ├── desktop_navigation/        # macOS accessibility examples
│   ├── web_voyager_agent.ipynb
│   └── test_mlx_max_tokens.ipynb
└── .github/workflows/
    ├── ci.yml                 # Lint + frontend type-check (every PR)
    ├── build-app.yml          # macOS build, sign, notarise, publish
    └── release.yml            # Semantic release → triggers build
```

---

## Acknowledgements

This project is built on the shoulders of many excellent open-source libraries. We are grateful to every maintainer and contributor.

Full copyright notices and license texts are in [`THIRD_PARTY_NOTICES`](./THIRD_PARTY_NOTICES).

### AI / Agent Frameworks

| Library | Description | License |
|---------|-------------|---------|
| [LangChain](https://github.com/langchain-ai/langchain) | Core LLM orchestration primitives (`langchain`, `langchain-core`, `langchain-community`, `langchain-text-splitters`) | MIT |
| [LangGraph](https://github.com/langchain-ai/langgraph) | Graph-based stateful agent runtime (`langgraph`, `langgraph-checkpoint-sqlite`) | MIT |
| [LangChain integrations](https://github.com/langchain-ai/langchain) | Provider adapters used: `langchain-openai`, `langchain-anthropic`, `langchain-cohere`, `langchain-aws`, `langchain-classic`, `langchain-mcp-adapters` | MIT |
| [Anthropic SDK](https://github.com/anthropic-ai/anthropic-sdk-python) | Official Python client for Claude models | MIT |
| [DeepAgents](https://pypi.org/project/deepagents/) | DeepAgent core utilities | Apache-2.0 |

### Evaluation

| Library | Description | License |
|---------|-------------|---------|
| [DeepEval](https://github.com/confident-ai/deepeval) | LLM evaluation framework used for agent benchmarking | Apache-2.0 |

### Local Inference (Apple Silicon)

| Library | Description | License |
|---------|-------------|---------|
| [MLX](https://github.com/ml-explore/mlx) | Apple's array framework for Apple Silicon | MIT |
| [MLX-LM](https://github.com/ml-explore/mlx-examples/tree/main/llms) | LLM inference on MLX | MIT |
| [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) | Vision-language model inference on MLX | MIT |
| [Transformers](https://github.com/huggingface/transformers) | HuggingFace model hub & tokenizers | Apache-2.0 |
| [huggingface-hub](https://github.com/huggingface/huggingface_hub) | Model download & Hub API client | Apache-2.0 |

### Web & Browser Automation

| Library | Description | License |
|---------|-------------|---------|
| [Playwright](https://github.com/microsoft/playwright-python) | Cross-browser automation used by the WebVoyager agent | Apache-2.0 |
| [aiohttp](https://github.com/aio-libs/aiohttp) | Async HTTP client/server | Apache-2.0 |

### Backend Framework

| Library | Description | License |
|---------|-------------|---------|
| [FastAPI](https://github.com/tiangolo/fastapi) | Async REST API backend | MIT |
| [Uvicorn](https://github.com/encode/uvicorn) | ASGI server | BSD-3-Clause |
| [Pydantic](https://github.com/pydantic/pydantic) | Data validation & settings management | MIT |
| [websockets](https://github.com/python-websockets/websockets) | WebSocket server for streaming agent events | BSD-3-Clause |
| [APScheduler](https://github.com/agronholm/apscheduler) | Background cron scheduler for scheduled agent runs | MIT |

### Research & Retrieval Tools

| Library | Description | License |
|---------|-------------|---------|
| [ddgs](https://github.com/deedy5/duckduckgo_search) (DuckDuckGo Search) | Web search without API keys | MIT |
| [markdownify](https://github.com/matthewwithanm/python-markdownify) | HTML → Markdown conversion for web pages | MIT |
| [rank-bm25](https://github.com/dorianbrown/rank_bm25) | BM25 retrieval used by doc_researcher | Apache-2.0 |
| [wikipedia](https://github.com/goldsmith/Wikipedia) | Wikipedia article fetcher | MIT |
| [tiktoken](https://github.com/openai/tiktoken) | Fast BPE tokeniser for token-budget tools | MIT |

### MCP (Model Context Protocol)

| Library | Description | License |
|---------|-------------|---------|
| [mcp](https://github.com/modelcontextprotocol/python-sdk) | Anthropic's Model Context Protocol Python SDK | MIT |

### Observability & Tracing

| Library | Description | License |
|---------|-------------|---------|
| [Traceloop SDK](https://github.com/traceloop/openllmetry) | OpenTelemetry-based LLM tracing | Apache-2.0 |

### Utilities

| Library | Description | License |
|---------|-------------|---------|
| [python-dotenv](https://github.com/theskumar/python-dotenv) | `.env` file loader | BSD-3-Clause |
| [httpx](https://github.com/encode/httpx) | Modern async HTTP client | BSD-3-Clause |
| [Pillow](https://github.com/python-pillow/Pillow) | Image processing for VLM screenshots | HPND |
| [aiosqlite](https://github.com/omnilib/aiosqlite) | Async SQLite for activity tracking & checkpointing | MIT |
| [nest-asyncio](https://github.com/erdewit/nest_asyncio) | Nested event-loop support (Jupyter & web loader) | BSD-2-Clause |
| [asyncssh](https://github.com/ronf/asyncssh) | Async SSH client for remote transcript access | Eclipse-2.0 |
| [PyJWT](https://github.com/jpadilla/pyjwt) | JWT encode/decode | MIT |
| [keyring](https://github.com/jaraco/keyring) | OS credential-vault wrapper (Keychain / Secret Service) | MIT |
| [grandalf](https://github.com/bdcht/grandalf) | Graph layout for LangGraph ASCII diagrams | Apache-2.0 |
| [strenum](https://github.com/irgeek/StrEnum) | `str`-based `Enum` backport | MIT |
| [certifi](https://github.com/certifi/python-certifi) | Mozilla CA bundle | MPL-2.0 |
| [pyautogui](https://github.com/asweigart/pyautogui) | Cross-platform GUI automation | BSD-3-Clause |
| [pyobjc](https://github.com/ronaldoussoren/pyobjc) | macOS Accessibility & AppKit bindings | MIT |

### Frontend

| Library | Description | License |
|---------|-------------|---------|
| [React](https://github.com/facebook/react) | UI component library | MIT |
| [Vite](https://github.com/vitejs/vite) | Fast frontend build tool | MIT |
| [Tailwind CSS](https://github.com/tailwindlabs/tailwindcss) | Utility-first CSS framework | MIT |
| [Tauri](https://github.com/tauri-apps/tauri) | Rust-based desktop app shell | MIT / Apache-2.0 |
| [react-router-dom](https://github.com/remix-run/react-router) | Client-side routing | MIT |
| [react-markdown](https://github.com/remarkjs/react-markdown) | Markdown rendering in React | MIT |
| [remark-gfm](https://github.com/remarkjs/remark-gfm) | GitHub Flavored Markdown plugin | MIT |
| [lucide-react](https://github.com/lucide-icons/lucide) | Icon library | ISC |
| [@dnd-kit](https://github.com/clauderic/dnd-kit) | Drag-and-drop primitives | MIT |

### Dev & Testing

| Library | Description | License |
|---------|-------------|---------|
| [pytest](https://github.com/pytest-dev/pytest) | Python test runner | MIT |
| [pytest-asyncio](https://github.com/pytest-dev/pytest-asyncio) | Async test support for pytest | Apache-2.0 |
| [flake8](https://github.com/PyCQA/flake8) | Python linter | MIT |
| [TypeScript](https://github.com/microsoft/TypeScript) | Typed JavaScript | Apache-2.0 |
