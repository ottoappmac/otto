#!/usr/bin/env bash
# The CAD eval set: general prompts with fixed expected figures, scored by measuring the exported STEP.
#
#   scripts/cadeval/eval.sh selftest                      references pass, broken copies fail (no model, no backend)
#   scripts/cadeval/eval.sh run [--arm spec|control|both] [--ids a,b|part|assembly|open|hard] [--repeat N]
#   scripts/cadeval/eval.sh report OUTDIR
#   scripts/cadeval/eval.sh stop                          stop the backend this starts
#
# `run` uses the same dedicated backend as scripts/benchcad (port 18092, its own HOME, only cad-tools on);
# it starts it if needed, with the model from BENCHCAD_PROVIDER / BENCHCAD_MODEL / BENCHCAD_API_KEY or your Otto
# config. Stop a running BenchCAD backend first if it was started with other settings.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RUNS="${BENCHCAD_RUNS:-$(dirname "$ROOT")/otto-benchcad-runs}"
PORT=18092
API="http://127.0.0.1:$PORT"
backend_up() { curl -fsS "$API/api/health" >/dev/null 2>&1 || curl -fsS "$API/" >/dev/null 2>&1; }

case "${1:-}" in
  selftest) shift; exec "$ROOT/.venv/bin/python" "$HERE/evalrun.py" selftest "$@" ;;
  report) shift; exec "$ROOT/.venv/bin/python" "$HERE/evalrun.py" report "$@" ;;
  run) shift; mkdir -p "$RUNS/out"
       if ! backend_up; then
         echo "starting backend on $PORT ..."
         nohup perl -e 'use POSIX; setsid(); exec @ARGV' "$ROOT/scripts/benchcad/start_backend.sh" >"$RUNS/backend.log" 2>&1 &
         for _ in $(seq 1 90); do backend_up && break; sleep 1; done
         backend_up || { echo "backend did not come up; see $RUNS/backend.log" >&2; exit 1; }
       fi
       exec "$ROOT/.venv/bin/python" "$HERE/evalrun.py" run --api "$API" "$@" ;;
  stop) pkill -f "backend --host 127.0.0.1 --port $PORT" && echo stopped || echo "not running" ;;
  *) sed -n 2,11p "$0"; exit 2 ;;
esac
