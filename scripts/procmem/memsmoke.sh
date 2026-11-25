#!/usr/bin/env bash
# Live smoke test of procedural memory on macOSWorld: is each task's run stored, does a later task call
# recall_procedure, and does it get the earlier run back? Uses live memory (`bench.sh up --memory live`): every
# finished task, passed or failed, is recallable by the tasks after it in the same run. Run from Terminal.app
# (not the Claude app's terminal), with OTTO.app quit (it shares the model server).
#
#   scripts/procmem/memsmoke.sh            # pair: task A, then the similar task B; B should recall A
#   scripts/procmem/memsmoke.sh same       # the same task twice (--runs 2); run 2 should recall run 1
#   scripts/procmem/memsmoke.sh up         # only bring up the VM + backend with live memory
#
# Env: MEMSMOKE_A, MEMSMOKE_B (task ids; default 7e4615cb "keep only the largest image" then 77e03b5e "delete only
# the largest text file"), MEMSMOKE_MAX_STEPS (default 100).
# `same` turns on same-task recall (MACARENA_MEMORY_SAME_TASK=1), which a normal live run leaves out; the report
# refuses to score such a run.
set -euo pipefail
cd "$(dirname "$0")/../.."

RUNS=../otto-macarena-runs
MODE="${1:-pair}"
A="${MEMSMOKE_A:-7e4615cb-9916-7215-cef9-42b13aed738c}"
B="${MEMSMOKE_B:-77e03b5e-e9b1-840a-550e-035c8e3f0134}"
STEPS="${MEMSMOKE_MAX_STEPS:-100}"
API=http://127.0.0.1:18093
case "$MODE" in
  pair|up) export MACARENA_MEMORY_SAME_TASK=0; IDS="$A,$B"; RUNS_PER=1 ;;
  same)    export MACARENA_MEMORY_SAME_TASK=1; IDS="$A";    RUNS_PER=2 ;;
  *) echo "usage: $0 [pair|same|up]" >&2; exit 2 ;;
esac

if pgrep -f "/Applications/OTTO.app/Contents/Resources/backend/otto-backend" >/dev/null; then
  echo "WARNING: OTTO.app is running and shares the model server; turns will be slow. Quit it first." >&2
fi

echo "== restarting the VM and backend with live memory (same-task recall: $MACARENA_MEMORY_SAME_TASK)"
scripts/macarena/bench.sh stop || true
MACARENA_MAX_STEPS="$STEPS" MACARENA_STEP_UNIT=turns scripts/macarena/bench.sh up --native --memory live
curl -sf "$API/api/procedures/status" | python3 -c 'import json,sys; d=json.load(sys.stdin); print({k: d.get(k) for k in ("mode","live","serve_tool","same_task","floor","embedder")})'
[[ "$MODE" == up ]] && exit 0

OUT="$RUNS/out/$(date +%Y%m%d%H%M)-memsmoke-$MODE"
echo "== running $IDS (runs per task: $RUNS_PER) into $OUT"
scripts/macarena/bench.sh run --sources macosworld --ids "$IDS" --runs "$RUNS_PER" --max-steps "$STEPS" \
  --step-unit turns --out "$OUT"

echo "== what memory did"
.venv/bin/python scripts/procmem/show_memory.py "$OUT" --text
python3 - "$OUT" "$A" <<'EOF'
import json, sqlite3, sys
from pathlib import Path
out, a = Path(sys.argv[1]), sys.argv[2]
rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines() if l.strip()]
for r in rows:
    print(f"{r['id'][:8]} run{r['run']}  score {r['score']}  memory {r.get('memory_mode')}  calls "
          f"{r.get('memory_calls', 0)}  used {[u.get('item') for u in r.get('memory_used', [])]}  "
          f"saved {r.get('memory_saved')}{'  SAVE ERROR ' + r['memory_save_error'] if r.get('memory_save_error') else ''}")
db = out / "memory" / "write.db"
if db.exists():
    for task, run_label, status, intent in sqlite3.connect(db).execute(
            "select task_id, label, status, intent from episodes order by created_at"):
        print(f"  stored  {str(task)[:8]}  {run_label:<8} {status:<9} {intent}")
second = rows[1] if len(rows) > 1 else None
first_ok = bool(rows) and rows[0].get("memory_mode") == "live" and bool(rows[0].get("memory_saved"))
got = [u for u in (second or {}).get("memory_used", []) if u.get("source_task") == a]
if not first_ok:
    verdict = "FAIL: the first task was not saved to live memory"
elif second is None:
    verdict = "FAIL: the second task did not run"
elif not second.get("memory_calls"):
    verdict = "FAIL: the second task never called recall_procedure"
elif not got:
    verdict = "FAIL: the second task called memory but got nothing from the first (below the floor, or another app)"
else:
    verdict = f"PASS: the second task recalled the first task's run ({[u.get('item') for u in got]})"
print("SMOKE", verdict)
EOF
