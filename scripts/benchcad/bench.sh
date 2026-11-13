#!/usr/bin/env bash
# BenchCAD dev runs with Otto's benchcad-agent.
#
#   scripts/benchcad/bench.sh check                 prerequisites
#   scripts/benchcad/bench.sh run [run.py args]     start backend if needed, run cases, score
#   scripts/benchcad/bench.sh stop                  stop the benchmark backend
#   scripts/benchcad/bench.sh harness [--cases DIR] [--ids task3/case1,...] [harness.py args]
#                                                   scripted single-part pipeline (no Otto backend needed);
#                                                   model via --base-url/--model/--api-key or BENCHCAD_* env
#
# BENCHCAD_DIR points at a BenchCAD checkout (default ../benchcad-2-main). Examples:
#   scripts/benchcad/bench.sh run --ids task1/case1          one sample case
#   scripts/benchcad/bench.sh run                            all 9 development samples
#   scripts/benchcad/bench.sh run --cases <public-set-dir> --workers 2
#   BENCHCAD_NO_SUBAGENTS=1 scripts/benchcad/bench.sh run --ids task4/case1   ablation: no subagents
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BENCH="${BENCHCAD_DIR:-$(dirname "$ROOT")/benchcad-2-main}"
PORT=18092
API="http://127.0.0.1:$PORT"
ARCH="$([[ "$(uname -m)" == arm64 ]] && echo arm64 || echo amd64)"
IMAGE="${CADENV_DOCKER_IMAGE:-benchcad-sandbox:$ARCH}"

check() {
  local ok=1
  command -v uv >/dev/null || { echo "missing: uv"; ok=0; }
  [[ -x "$ROOT/.venv/bin/python" ]] || { echo "missing: $ROOT/.venv (run ./start_web.sh once)"; ok=0; }
  [[ -d "$BENCH/envs" ]] || { echo "missing: BenchCAD checkout at $BENCH (git clone https://github.com/BenchCAD-org/benchcad-2-main; git checkout v2.1)"; ok=0; }
  if docker info >/dev/null 2>&1 && docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "scoring: in Docker ($IMAGE), as BenchCAD does"
  else
    echo "scoring: no Docker image $IMAGE, answers will be scored as a local subprocess (fine for dev runs;"
    echo "         for BenchCAD's own isolation: docker build -t $IMAGE $BENCH/sandbox/)"
  fi
  [[ -x "$BENCH/.venv/bin/python" ]] || { echo "missing: BenchCAD env. Run: (cd $BENCH && uv sync --group harness)"; ok=0; }
  [[ $ok == 1 ]] && echo "ok: uv, Otto venv, BenchCAD at $BENCH"
  [[ $ok == 1 ]]
}

backend_up() { curl -fsS "$API/api/health" >/dev/null 2>&1 || curl -fsS "$API/" >/dev/null 2>&1; }

MODE="subagents=${BENCHCAD_NO_SUBAGENTS:-0} time=${BENCHCAD_TIME_LIMIT:-3600} model=${BENCHCAD_MODEL:-default} key=${BENCHCAD_API_KEY:+set}"
MODEFILE="${BENCHCAD_RUNS:-$(dirname "$ROOT")/otto-benchcad-runs}/backend.mode"

start() {
  if backend_up; then
    # the backend reads these settings when it starts; a running one keeps its own
    if [[ "$(cat "$MODEFILE" 2>/dev/null)" != "$MODE" ]]; then
      echo "the running backend was started with: $(cat "$MODEFILE" 2>/dev/null || echo unknown); this run wants: $MODE" >&2
      echo "run: scripts/benchcad/bench.sh stop   then run again" >&2; exit 1
    fi
    return
  fi
  echo "starting benchmark backend on $PORT ($MODE) ..."
  echo "$MODE" >"$MODEFILE"
  nohup perl -e 'use POSIX; setsid(); exec @ARGV' "$HERE/start_backend.sh" \
    >"${BENCHCAD_RUNS:-$(dirname "$ROOT")/otto-benchcad-runs}/backend.log" 2>&1 &
  for _ in $(seq 1 90); do backend_up && return; sleep 1; done
  echo "backend did not come up; see backend.log" >&2; exit 1
}

case "${1:-}" in
  check) check ;;
  run) shift; check
       docker image inspect "$IMAGE" >/dev/null 2>&1 || export CADENV_LOCAL=1; mkdir -p "${BENCHCAD_RUNS:-$(dirname "$ROOT")/otto-benchcad-runs}"; start
       exec "$ROOT/.venv/bin/python" "$HERE/run.py" --bench "$BENCH" --api "$API" "$@" ;;
  harness) shift
       CASES="$BENCH/examples"; IDS=""; PASS=()
       while [[ $# -gt 0 ]]; do case "$1" in
         --cases) CASES="$2"; shift 2 ;; --ids) IDS="$2"; shift 2 ;; *) PASS+=("$1"); shift ;; esac; done
       CADPY="$HOME/Library/Application Support/Otto/mcp_server/cad-tools/.venv/bin/python"
       [[ -x "$CADPY" ]] || { echo "missing: the CAD Tools environment. In Otto, open Settings > MCP and Connect CAD Tools once." >&2; exit 1; }
       RUNS="${BENCHCAD_RUNS:-$(dirname "$ROOT")/otto-benchcad-runs}"; OUT="$RUNS/harness/$(date +%Y%m%d%H%M)"; mkdir -p "$OUT"
       CASELIST="$(uv run --project "$BENCH" python "$HERE/cases.py" list "$BENCH" "$CASES")"
       echo "$CASELIST" | python3 -c "
import sys, json
ids = [x for x in '$IDS'.split(',') if x]
for c in json.load(sys.stdin):
    if c['task'] in ('task1', 'task3') and (not ids or c['id'] in ids):
        print(c['id'], c['case'])" | while read -r id case; do
         echo "== $id"
         "$CADPY" "$HERE/harness.py" --bench "$BENCH" --case "$case" --work "$OUT/work/${id//\//__}" ${PASS[@]+"${PASS[@]}"} \
           2>"$OUT/${id//\//__}.log" | tail -1 | tee -a "$OUT/results.jsonl" | python3 -c "
import sys, json
r = json.loads(sys.stdin.read() or '{}'); print('  score', r.get('score'), ' overlap', r.get('best_overlap'), ' calls', r.get('model_calls'), ' error', r.get('error') or r.get('score_error'))"
       done
       echo "results: $OUT/results.jsonl" ;;
  stop) rm -f "$MODEFILE"; pkill -f "backend --host 127.0.0.1 --port $PORT" && echo stopped || echo "not running" ;;
  *) sed -n 2,14p "$0"; exit 2 ;;
esac
