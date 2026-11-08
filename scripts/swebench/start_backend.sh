#!/usr/bin/env bash
# Start a dedicated Otto backend for benchmarking on port 18091, with its own
# HOME so your real sessions, memory and settings are untouched. The model and
# keys come from a copy of your current Otto config, patched for headless runs.
#
# Model selection (all optional; defaults to your Otto config):
#   SWEBENCH_PROVIDER = anthropic | openai | google | omlx | mlx
#   SWEBENCH_MODEL    = model name for that provider (HF repo id for mlx)
#   SWEBENCH_API_KEY  = API key for the provider (written only to the isolated config)
#   SWEBENCH_PROMPT_CACHE = 1 (default) | 0   prompt caching for the in-process mlx provider
#   SWEBENCH_TOOL_BUDGET  = max tool calls per instance (default 200)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS="${SWEBENCH_RUNS:-$(dirname "$ROOT")/otto-swebench-runs}"
BENCH_HOME="$RUNS/home"
REAL_HOME="$HOME"
SRC="$HOME/Library/Application Support/Otto/config.json"
DST="$BENCH_HOME/Library/Application Support/Otto"
mkdir -p "$DST" "$RUNS/work/pythons"

if [[ -f "$SRC" ]]; then SRC_ARG="$SRC"; else SRC_ARG=""; fi

BENCH_ROOT="$ROOT" REAL_HOME="$REAL_HOME" "$ROOT/.venv/bin/python" - "$SRC_ARG" "$DST/config.json" <<'PY'
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]
if src:
    cfg = json.load(open(src))
else:
    # No Otto config yet: let the backend build its defaults, then patch those.
    sys.path.insert(0, os.environ["BENCH_ROOT"] + "/src")
    from backend.config import AppConfig
    cfg = AppConfig().model_dump()
o = cfg["orchestrator"]
budget = int(os.environ.get("SWEBENCH_TOOL_BUDGET", "200"))
o.update(default_run_depth="deep", tool_call_soft_budget=max(budget * 3 // 5, 1), tool_call_hard_budget=budget)
ap = cfg["action_policy"]
ap["categories"].update(shell="allow", workspace_write="allow", destructive="allow")
ap["execute_sandbox"]["network"] = False
cfg["auto_approve_commands"] = True
cfg["decision"]["safety_guard_enabled"] = False  # isolated bench config only; real config untouched
# No memory: consolidation would write earlier instances' fixes into MEMORY.md, inject them into later
# instances, and spend model time on the shared server while agents run.
cfg["memory"].update(enabled=False, inject_enabled=False, inject_on_session_start=False, inject_realtime=False)
# The agent only needs file and shell tools. Tool servers (Playwright on port 8931, Slack, Mail...) would
# only cost startup time, memory and port clashes with the real app, so none start in the benchmark copy.
for server in cfg.get("mcp_servers") or []:
    server["enabled"] = False
    server["auto_start"] = False

prov = os.environ.get("SWEBENCH_PROVIDER")
model = os.environ.get("SWEBENCH_MODEL")
key = os.environ.get("SWEBENCH_API_KEY")
if prov:
    cfg["llm"]["provider"] = prov
provider = cfg["llm"]["provider"]
if provider == "mlx":
    mlx = cfg["llm"]["mlx"]
    if model:
        mlx["hf_llm_model_id"] = model
        mlx["mlx_model_type"] = "llm"
    # Share the weights already downloaded for the real Otto instead of re-downloading.
    mlx["hf_hub_cache"] = os.environ["REAL_HOME"] + "/.cache/huggingface/hub"
    mlx["mlx_max_tokens"] = max(int(mlx.get("mlx_max_tokens") or 0), 16384)
    # In-process MLX only; an oMLX server does its own prefix caching (see its settings).
    cache_on = os.environ.get("SWEBENCH_PROMPT_CACHE", "1") != "0"
    mlx["mlx_prompt_cache"] = cache_on
    mlx["mlx_system_prompt_cache"] = cache_on
    shown = mlx.get("hf_llm_model_id")
else:
    target = cfg["omlx"] if provider == "omlx" else cfg["llm"].setdefault(provider, {})
    if model:
        target["model_name"] = model
    if key:
        target["api_key"] = key
    shown = target.get("model_name")
print("benchmark model:", provider, shown)
json.dump(cfg, open(dst, "w"), indent=2)
PY

cd "$ROOT"
BENCH_ROOT="$ROOT" HOME="$BENCH_HOME" OTTO_API_TOKEN_REQUIRED=0 \
  UV_PYTHON_INSTALL_DIR="$RUNS/work/pythons" \
  HF_HOME="$REAL_HOME/.cache/huggingface" HF_HUB_CACHE="$REAL_HOME/.cache/huggingface/hub" \
  exec "$ROOT/.venv/bin/python" -m backend --host 127.0.0.1 --port 18091
