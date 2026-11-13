#!/usr/bin/env bash
# Dedicated Otto backend for BenchCAD dev runs: port 18092, its own HOME, so your real sessions,
# memory and settings are untouched. Model and keys come from a copy of your Otto config.
#
# Optional env:
#   BENCHCAD_PROVIDER / BENCHCAD_MODEL / BENCHCAD_API_KEY   model for the run (default: your Otto config; on oMLX, Qwen3.6-35B-A3B-8bit)
#   BENCHCAD_RUNS      where runs live (default: ../otto-benchcad-runs next to the repo)
#   BENCHCAD_TOOL_BUDGET  max tool calls per case (default 300)
#   CAD_EXEC              where the agent's programs run: auto (Seatbelt on macOS) | docker | local
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS="${BENCHCAD_RUNS:-$(dirname "$ROOT")/otto-benchcad-runs}"
BENCH_HOME="$RUNS/home"
REAL_HOME="$HOME"
SRC="$HOME/Library/Application Support/Otto/config.json"
DST="$BENCH_HOME/Library/Application Support/Otto"
mkdir -p "$DST" "$RUNS/out"
[[ -f "$SRC" ]] && SRC_ARG="$SRC" || SRC_ARG=""

BENCH_ROOT="$ROOT" REAL_HOME="$REAL_HOME" CAD_RUNS="$RUNS" "$ROOT/.venv/bin/python" - "$SRC_ARG" "$DST/config.json" <<'PY'
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
budget = int(os.environ.get("BENCHCAD_TOOL_BUDGET", "300"))
o.update(default_run_depth="deep", tool_call_soft_budget=max(budget * 3 // 5, 1), tool_call_hard_budget=budget)
ap = cfg["action_policy"]
ap["categories"].update(shell="allow", workspace_write="allow", destructive="allow")
ap["execute_sandbox"]["network"] = False
cfg["auto_approve_commands"] = True
cfg["decision"]["safety_guard_enabled"] = False      # isolated bench config only
# No memory: notes from one case would reach the next.
cfg["memory"].update(enabled=False, inject_enabled=False, inject_on_session_start=False, inject_realtime=False)

# Only cad-tools runs. Everything else is off in this copy.
from backend.builtin_mcps.registry import BUILTIN_MCPS, builtin_mcp_config
servers = [s for s in (cfg.get("mcp_servers") or []) if s.get("id") != "cad-tools"]
for s in servers:
    s["enabled"] = False
    s["auto_start"] = False
spec = next(m for m in BUILTIN_MCPS if m.id == "cad-tools")
cad = builtin_mcp_config(spec).model_dump()
cad["enabled"] = True
cad["env"] = {k: v for k, v in {
    "CAD_WORK_ROOT": os.environ["CAD_RUNS"],
    "CAD_EXEC": os.environ.get("CAD_EXEC", ""),
    "CADENV_DOCKER_IMAGE": os.environ.get("CADENV_DOCKER_IMAGE", ""),
    "CAD_TIME_LIMIT": os.environ.get("BENCHCAD_TIME_LIMIT", "3600"),
    "CAD_DENY_READ": os.environ.get("BENCHCAD_DIR") or os.path.join(os.path.dirname(root), "benchcad-2-main"),
}.items() if v}
servers.append(cad)
cfg["mcp_servers"] = servers

prov, model, key = (os.environ.get(k) for k in ("BENCHCAD_PROVIDER", "BENCHCAD_MODEL", "BENCHCAD_API_KEY"))
if prov:
    cfg["llm"]["provider"] = prov
provider = cfg["llm"]["provider"]
if provider == "omlx" and not model:       # default benchmark model: Qwen3.6 35B MoE (3B active), 8-bit
    model = "mlx-community--Qwen3.6-35B-A3B-8bit"
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
print("benchmark model:", provider, shown)
json.dump(cfg, open(dst, "w"), indent=2)
PY

cd "$ROOT"
HOME="$BENCH_HOME" OTTO_API_TOKEN_REQUIRED=0 PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring \
  HF_HOME="$REAL_HOME/.cache/huggingface" HF_HUB_CACHE="$REAL_HOME/.cache/huggingface/hub" \
  exec "$ROOT/.venv/bin/python" -m backend --host 127.0.0.1 --port 18092
