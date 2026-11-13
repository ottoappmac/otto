# BenchCAD harness for Otto

Runs Otto's `benchcad-agent` on [BenchCAD 2.x](https://benchcad.com/setup.html) cases and scores them with BenchCAD's own scorer.

**This is a dev harness. Its scores are not reportable.** BenchCAD only accepts agent runs on its
*Customized harness* track (`own_agent/` in the BenchCAD repo): Harbor, a Docker container, 500k output
tokens per case counted by a meter at the model API, and only the Anthropic / OpenAI / Gemini APIs
reachable. This harness lets you build and tune the agent first, with any model Otto supports,
including local ones. The reportable adapter is listed under "Not done".

## What it uses from Otto

| Piece | Where |
|---|---|
| CAD tools (`cad_run`, `cad_render`, `cad_compare`, `cad_autoplace`, `assembly_status`, `step_inspect`, `answer_checkpoint`, `budget_status`, `netlist_validate`, `cad_export`) | built-in MCP `cad-tools`, [`backend/builtin_mcps/cad_tools/`](../../backend/builtin_mcps/cad_tools) |
| `benchcad-agent` (headless, BenchCAD's answer contract) and its one subagent `drawing-reader` (dense drawings only). Assembly and PCB guidance is in its own prompt. The general, interactive `cad-agent` (subagents `drawing-reader`, `cad-critic`) is separate and not used here | [`backend/agent_library.py`](../../backend/agent_library.py) |
| Subagent wiring, tool allowlists, memory off | `_AGENT_SUBAGENTS`, `_AGENT_TOOL_ALLOWLIST`, `_AGENTS_WITHOUT_MEMORY` in [`backend/session_manager.py`](../../backend/session_manager.py) |

The agent's programs run with the same CadQuery pins as BenchCAD's sandbox, under Seatbelt on macOS (no Docker).
Scoring uses BenchCAD's own code: in Docker when the `benchcad-sandbox` image exists, otherwise as a local
subprocess (`CADENV_LOCAL=1`), which is fine for dev runs.

## Setup

```bash
git clone https://github.com/BenchCAD-org/benchcad-2-main ../benchcad-2-main
git -C ../benchcad-2-main checkout v2.1                    # use a tag for anything you compare
docker build -t benchcad-sandbox:arm64 ../benchcad-2-main/sandbox/      # optional: score in Docker like BenchCAD
(cd ../benchcad-2-main && uv sync --group harness)         # staging and scoring run in BenchCAD's env
scripts/benchcad/bench.sh check
```

## Run

```bash
scripts/benchcad/bench.sh run --ids task1/case1            # one development sample
scripts/benchcad/bench.sh run                              # the 9 samples in examples/
scripts/benchcad/bench.sh run --cases <public-set-dir> --workers 2
scripts/benchcad/bench.sh stop
```

Subagents on/off (an ablation; stop the backend between the two, `bench.sh run` refuses to reuse one started
with other settings, and each row in `results.jsonl` records `"subagents"`):

```bash
scripts/benchcad/bench.sh stop; scripts/benchcad/bench.sh run --ids task4/case1
scripts/benchcad/bench.sh stop; BENCHCAD_NO_SUBAGENTS=1 scripts/benchcad/bench.sh run --ids task4/case1
```

Pick the model with `BENCHCAD_PROVIDER`, `BENCHCAD_MODEL`, `BENCHCAD_API_KEY` (default: your Otto config).
`bench.sh run` starts an isolated backend on port 18092 with its own HOME, memory off, and only `cad-tools` enabled.

Output goes to `../otto-benchcad-runs/out/<run>/`: `results.jsonl` (resumable), `summary.txt`,
`trajectories/` (events, messages, score detail per case) and `work/` (each case's folder).

## How a case runs

1. `cases.py stage` copies the inputs the way BenchCAD's sandbox does (PDFs become sheets and tiles) and writes `TASK.md` from BenchCAD's own task text.
2. The agent works in that folder and leaves `answer.py`.
3. `cases.py score` stages a fresh copy of the inputs, copies in only the agent's own `.py` files, runs `answer.py` the way BenchCAD's harness does (Docker, no network, 600 s) and scores it.

## Status

Tested: the tool server end to end with real CadQuery 2.3.0, and the unit tests (`tests/test_cad_tools.py`,
`tests/test_cad_autoplace.py`). Scores from this harness are dev scores only.

**Not tested**: other cases and tasks, models other than the local one, and the Docker scoring path.

## Not done

- **Reportable track.** Needs headless Otto on Linux inside BenchCAD's Harbor task image, an `otto` CLI entry point,
  all model calls routed to the meter (`ANTHROPIC_BASE_URL=http://meter:8080/anthropic`), and memory / Auto depth / Jev / title
  generation off so nothing else calls a model. The `cad-tools` server already reads `METER_URL`, and the agents are written
  to run without a user.
- **Token budget in the agent.** Otto's budgets count tool calls, not output tokens. `budget_status` only reads the meter.
- **Context compaction and image handling** tuned to BenchCAD's long, image-heavy runs.

## Harness for single-part cases (`harness.py`)

A scripted pipeline instead of a free-form agent. The harness stages the case, asks the model for several
candidate programs in parallel, runs and scores each (`cad_compare`: outline overlap with the input
`views.png`), repairs the best ones for a few rounds with the red/blue difference picture, always keeps the best
valid program as `answer.py`, then scores it with BenchCAD's scorer. Any OpenAI-compatible endpoint works.

```bash
CADPY="$HOME/Library/Application Support/Otto/mcp_server/cad-tools/.venv/bin/python"   # has CadQuery, numpy, httpx
"$CADPY" scripts/benchcad/harness.py --bench ../benchcad-2-main \
  --case ../benchcad-2-main/examples/task3/cases/case1 --work ../otto-benchcad-runs/h/task3-case1 \
  --base-url https://api.openai.com/v1 --model <model> --api-key "$KEY"
```

Tested with a mock model only (the pipeline, scoring hand-off and `cad_compare` run). Not tested with a real model. Single parts only; the overlap score needs a
2x2 `views.png` (task 3), so drawing cases (task 1) are repaired on errors only. The scorer's scale handling is unchecked.
