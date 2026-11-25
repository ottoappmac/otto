#!/usr/bin/env bash
# Dedicated Otto backend for MacArena runs: port 18093 (MACARENA_API_PORT), its own HOME, so your real sessions, memory and
# settings are untouched. Only the macarena-vm MCP is enabled; every other MCP server is off, so the agent
# has no tool that acts on this Mac. (Otto's file and shell tools are refused for macarena-agent by
# backend/macarena_guard_middleware.py.)
#
# Needs MACARENA_AGENT_TOKEN (the bridge's agent token) and optionally MACARENA_BRIDGE_URL; both are
# written to $HOME/.macarena-bridge.json of the benchmark profile for the tool server to read.
# Optional env:
#   MACARENA_PROVIDER / MACARENA_MODEL / MACARENA_API_KEY   model for the run (default: your Otto config)
#   MACARENA_RUNS         where runs live (default: ../otto-macarena-runs next to the repo)
#   MACARENA_STATE        this backend's own folder (benchmark HOME, desktop lock); default MACARENA_RUNS.
#                         bench.sh gives each slot its own, so two backends never share a profile or lock
#   MACARENA_API_PORT     port to listen on (default 18093; bench.sh's slot 2 uses 18193)
#   MACARENA_TOOL_BUDGET  Otto's own tool-call cap per task, a backstop behind the bridge's step cap (default 100)
#   MACARENA_TRACK        standard (default) or native. native also enables Otto's macos-native desktop tools and
#                         starts the backend through native_backend.py, which makes those tools run in the VM.
#   MACARENA_MEMORY       off (default), live, or the path of a procedural-memory snapshot (scripts/procmem/build_snapshot.py),
#                         served read-only to the native agent's recall_procedure tool. live serves the run's own
#                         write store instead: every finished task (passed or failed) is recallable by later tasks. Either way the backend saves
#                         each task's run to the write store run.py names (out/<run>/memory/write.db), never to the
#                         snapshot and never to the app's own store.
#   MACARENA_MEMORY_SAME_TASK  1 lets a task recall the snapshot's runs of the same task (normally left out). For
#                         smoke tests of recall only: results are marked memory_same_task and are not clean scores.
set -euo pipefail
: "${MACARENA_AGENT_TOKEN:?set MACARENA_AGENT_TOKEN (bench.sh does this)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS="${MACARENA_RUNS:-$(dirname "$ROOT")/otto-macarena-runs}"
STATE="${MACARENA_STATE:-$RUNS}"
API_PORT="${MACARENA_API_PORT:-18093}"
BENCH_HOME="$STATE/home"
REAL_HOME="$HOME"
SRC="$HOME/Library/Application Support/Otto/config.json"
DST="$BENCH_HOME/Library/Application Support/Otto"
mkdir -p "$DST" "$RUNS/out"
[[ -f "$SRC" ]] && SRC_ARG="$SRC" || SRC_ARG=""

TRACK="${MACARENA_TRACK:-standard}"
[[ "$TRACK" == standard || "$TRACK" == native ]] || { echo "MACARENA_TRACK must be standard or native" >&2; exit 2; }
BENCH_ROOT="$ROOT" REAL_HOME="$REAL_HOME" BENCH_HOME="$BENCH_HOME" TRACK="$TRACK" "$ROOT/.venv/bin/python" - "$SRC_ARG" "$DST/config.json" <<'PY'
import json, os, sys
root = os.environ["BENCH_ROOT"]
sys.path.insert(0, root)
sys.path.insert(0, root + "/src")
src, dst = sys.argv[1], sys.argv[2]
if src:
    cfg = json.load(open(src))
else:
    from backend.config import AppConfig
    cfg = AppConfig().model_dump()
o = cfg["orchestrator"]
budget = int(os.environ.get("MACARENA_TOOL_BUDGET", "100"))
# MACARENA_TOOL_BUDGET: Otto's cap on tool calls per run (reads included); 0 turns it and its "converge" nudge off.
o.update(default_run_depth="deep", tool_call_soft_budget=max(budget * 3 // 5, 1) if budget > 0 else 0,
         tool_call_hard_budget=max(budget, 0))
ap = cfg["action_policy"]
ap["categories"].update(desktop_control="allow")       # the only tools the agent has act on the VM
cfg["auto_approve_commands"] = True
cfg["decision"]["safety_guard_enabled"] = False        # isolated bench config only
# No memory: notes from one task would reach the next.
cfg["memory"].update(enabled=False, inject_enabled=False, inject_on_session_start=False, inject_realtime=False)

# Only macarena-vm runs. Everything else is off in this copy.
from backend.builtin_mcps.registry import BUILTIN_MCPS, builtin_mcp_config
servers = [s for s in (cfg.get("mcp_servers") or []) if s.get("id") != "macarena-vm"]
for s in servers:
    s["enabled"] = False
    s["auto_start"] = False
spec = next(m for m in BUILTIN_MCPS if m.id == "macarena-vm")
vm = builtin_mcp_config(spec).model_dump()
vm["enabled"] = True
# Otto resets a built-in's env from the registry on every config load, so the bridge address and token go in
# a file in this profile's own HOME (mode 600) that the tool server reads. It is not in config.json.
bridge_file = os.path.join(os.environ["BENCH_HOME"], ".macarena-bridge.json")
fd = os.open(bridge_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as f:
    json.dump({"url": os.environ.get("MACARENA_BRIDGE_URL", "http://127.0.0.1:18095"),
               "token": os.environ["MACARENA_AGENT_TOKEN"]}, f)
servers.append(vm)
if os.environ["TRACK"] == "native":
    # Otto's own desktop tools, with the user's settings for them (excluded tools etc.) when present.
    # native_backend.py makes them run in the VM; the guard refuses them if that swap is missing.
    from backend.config import MCPServerConfig
    mine = next((s for s in (cfg.get("mcp_servers") or []) if s.get("id") == "macos-native"), None)
    native = dict(mine) if mine else MCPServerConfig(
        id="macos-native", name="macOS Desktop (Accessibility)", transport="stdio", builtin=True,
        enabled=True, auto_start=False, requires_os="macos").model_dump()
    native["enabled"] = True
    # Otto's AppleScript tools. native_backend.py loads them without starting the server here: every call runs
    # in the VM (backend/macarena_native.py); the guard refuses them if that swap is missing.
    osa = builtin_mcp_config(next(m for m in BUILTIN_MCPS if m.id == "macos-osascript")).model_dump()
    osa.update(enabled=True, auto_start=False)
    servers = [s for s in servers if s.get("id") not in ("macos-native", "macos-osascript")] + [native, osa]
    ap["macos_automation_enabled"] = True     # this config's kill switch; these tools only reach the VM
    # Scripts act on the VM's apps only. run.py approves any card anyway; a per-session cap would instead deny
    # (a script that says "delete" counts as destructive), which no approval can undo.
    ap["categories"].update(applescript="allow", mail_read="allow")
    ap["rate_limit_per_session"] = {}
cfg["mcp_servers"] = servers

prov, model, key = (os.environ.get(k) for k in ("MACARENA_PROVIDER", "MACARENA_MODEL", "MACARENA_API_KEY"))
if prov:
    cfg["llm"]["provider"] = prov
provider = cfg["llm"]["provider"]
if provider == "mlx":
    mlx = cfg["llm"]["mlx"]
    if model:
        mlx["hf_llm_model_id"] = model
        mlx["mlx_model_type"] = "llm"
    mlx["hf_hub_cache"] = os.environ["REAL_HOME"] + "/.cache/huggingface/hub"
    shown = mlx.get("hf_llm_model_id")
else:
    target = cfg["omlx"] if provider == "omlx" else cfg["llm"].setdefault(provider, {})
    if model:
        target["model_name"] = model
    if key:
        target["api_key"] = key
    shown = target.get("model_name")
print("benchmark track:", os.environ["TRACK"])
print("benchmark model:", provider, shown)
print("NOTE: Track A needs a vision model; with a text-only model the agent only has the accessibility tree.")
json.dump(cfg, open(dst, "w"), indent=2)
PY

cd "$ROOT"
if [[ "$TRACK" == native ]]; then ENTRY=("$ROOT/scripts/macarena/native_backend.py"); else ENTRY=(-m backend); fi
MEMORY="${MACARENA_MEMORY:-off}"
PROC_ENV=(OTTO_PROCEDURE_MODE=off)
if [[ "$MEMORY" != off ]]; then
  [[ "$TRACK" == native ]] || { echo "MACARENA_MEMORY needs MACARENA_TRACK=native" >&2; exit 2; }
  if [[ "$MEMORY" == live ]]; then
    # Live: recall reads the run's own write store (out/<run>/memory/write.db), so each finished task is there
    # for the tasks after it. Starts empty for a new --out; never the app's store (run.py refuses that path).
    PROC_ENV=(OTTO_PROCEDURE_MODE=live)
  else
  [[ -f "$MEMORY" ]] || { echo "no procedure snapshot at $MEMORY" >&2; exit 2; }
  case "$MEMORY" in
    "$REAL_HOME/Library/Application Support/Otto"*)
      echo "refusing to serve the app's own procedure store to a benchmark; build a snapshot" >&2; exit 2 ;;
  esac
  PROC_ENV=(OTTO_PROCEDURE_MODE=serve "OTTO_PROCEDURE_SERVE_DB=$MEMORY")
  fi
  # Optional: the similarity floor (default from config, tuned for MiniLM) and the query embedder (default: the
  # snapshot's own). Each changes the arm, so run.py records both in every result row.
  [[ -n "${MACARENA_MEMORY_FLOOR:-}" ]] && PROC_ENV+=("OTTO_PROCEDURE_FLOOR=$MACARENA_MEMORY_FLOOR")
  [[ -n "${MACARENA_MEMORY_EMBEDDER:-}" ]] && PROC_ENV+=("OTTO_PROCEDURE_EMBEDDER=$MACARENA_MEMORY_EMBEDDER")
  [[ "${MACARENA_MEMORY_SAME_TASK:-}" == 1 ]] && PROC_ENV+=("OTTO_PROCEDURE_SAME_TASK=1")
fi
echo "procedural memory: $MEMORY"
HOME="$BENCH_HOME" OTTO_API_TOKEN_REQUIRED=0 PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring \
  HF_HOME="$REAL_HOME/.cache/huggingface" HF_HUB_CACHE="$REAL_HOME/.cache/huggingface/hub" \
  OTTO_DESKTOP_LOCK_PATH="$STATE/desktop.lock" \
  exec env "${PROC_ENV[@]}" "$ROOT/.venv/bin/python" "${ENTRY[@]}" --host 127.0.0.1 --port "$API_PORT"
