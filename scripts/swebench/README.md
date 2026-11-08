# SWE-bench harness for Otto

Runs Otto's `swebench-coding-agent` on SWE-bench instances through the local
backend API, then scores the patches with SWE-bench's own Docker harness.

## One command

```bash
scripts/swebench/bench.sh                  # smoke test: 2 Verified instances, then score
scripts/swebench/bench.sh check            # only check / set up prerequisites
scripts/swebench/bench.sh run --sample 50  # 50 repo-stratified Verified instances
scripts/swebench/bench.sh score --out DIR  # score an earlier run folder
```

`bench.sh --help` lists every option (`--ids`, `--repo`, `--dataset`, `--workers`,
`--timeout`, `--provider/--model/--api-key`, `--runs-dir`, `--out`, `--method`,
`--no-score`, `--yes`, `--keep-backend`).

## How it works

```
 bench.sh (one command)
 │
 ├─ 1. SET UP ───────────────────────────────────────────────────────────────
 │     uv + Otto .venv ─► pick model ─► Docker running? ─► start benchmark backend
 │                         (Otto's own, else a local MLX     (own HOME, port 18091,
 │                          model, downloaded if missing)     patched config copy)
 │
 ├─ 2. RUN  (run.py, one pipeline per instance, several in parallel) ──────────
 │
 │   SWE-bench Verified ─► for each instance:
 │   (500 issues)
 │        │
 │        ├─ prepare ──► git archive of repo at base_commit ──► fresh 1-commit repo
 │        │              + uv venv (right Python) + pip install   (no upstream history,
 │        │              (runs AHEAD of the agent, bounded)        so no fix to peek at)
 │        │
 │        ├─ agent  ───► allow folder (network off) ─► new session ─► send the issue
 │        │              swebench-coding-agent: find files ► reproduce ► minimal fix
 │        │              ► re-run tests. Approvals and questions are answered by run.py.
 │        │
 │        └─ collect ──► git diff vs baseline, minus tests/scratch ──► preds.jsonl
 │                       + full message log ──► trajectories/<id>.json
 │
 └─ 3. SCORE  (score.py) ──────────────────────────────────────────────────────
       preds.jsonl ─► SWE-bench harness builds Docker image per repo/version
                      ─► applies each patch ─► runs the hidden tests
                      ─► results.json  (resolved %, which ids, empty patches, errors)
```

## Offline use, and less disk and download

Pulling a ~3 GB official image per instance, several times over, is the biggest cost of a run.
`warm` replaces that with a one-time setup, and `--offline` then runs with no network at all.

```bash
scripts/swebench/bench.sh warm                       # one time, needs the internet
scripts/swebench/bench.sh run --sample 10 --offline  # no network from here on
```

`warm` does four things:

| Step | What it leaves behind |
|---|---|
| Exports the dataset | `<runs-dir>/dataset/swe-bench_verified.jsonl` (8 MB). Used automatically afterwards, online or off |
| Records the scorer's GitHub lookups | `<runs-dir>/http-cache/` (a few hundred KB). The scorer reads requirements files from GitHub even with prebuilt images; offline runs replay them |
| Caches the benchmark's Python packages | in uv's cache |
| Loads the image bundle | all 500 Verified images in Docker: a 5.4 GB download, about 34 GB on disk, about 10 minutes. Skipped if already loaded |

The bundle is [LogicStar's compressed set](https://github.com/logic-star-ai/swe-bench-compressed)
(`logicstar/sweb.eval.x86_64.<repo>_1776_<id>:latest`, Verified only, x86_64). It was checked against the
official images on 23 instances (all 21 sample gold patches, plus two with known results: `django-10097`
fails the same 7 tests, `sphinx-10323` passes). The other 477 have not been compared, and it is a small
unlicensed project, so **use the official images (`--image-source official`) for any score you quote**.

`--offline` also: sets the Hugging Face and uv offline switches, never pulls an image (an instance whose
image is not local is skipped or recorded as an error), forces Docker environments (host virtualenvs need
PyPI and compilers), and checks everything before the first agent starts. Images already in Docker are
used as they are and never removed; only images a run pulled itself are deleted afterwards.

Tested by running the scripts under a macOS sandbox that blocks all outside network (localhost and the
Docker socket allowed): `check`, gold scoring, agent-patch scoring and a full agent run all pass.
That sandbox cannot be used around the agent itself (it stops Otto's own shell sandbox from starting),
so the agent run was repeated with web libraries pointed at a dead proxy instead.

## What it sets up

| Need | Handled by |
|---|---|
| `uv`, Otto's `.venv`, Python deps | installed with Otto's own bootstrap helpers if missing |
| A model | uses your current Otto model if usable (oMLX reachable with the model, MLX model cached, or a cloud provider). Otherwise downloads a local MLX model sized to your RAM (30B coder MoE at 32 GB+, 8B below) |
| Docker (scoring only) | started if installed; installed with Homebrew only when you pass `--yes`; otherwise scoring is skipped and the command to run later is printed |
| Benchmark Python packages | `uv run --with-requirements scripts/swebench/requirements.txt` |
| Repos, per-instance Python, dataset | downloaded on demand |

The benchmark backend runs on port 18091 with its own `HOME`, a patched copy of your
Otto config, and the Jev guardrail off in that copy only. Your real Otto data and
settings are not touched.

## Where files go

Everything is written outside the repo, to `otto-swebench-runs/` **next to the otto
repo** by default. Change it with `--runs-dir DIR` or `SWEBENCH_RUNS=DIR`.

```
<runs-dir>/out/<run>/preds.jsonl        predictions (resumable)
<runs-dir>/out/<run>/trajectories/      per-instance message and tool-call logs
<runs-dir>/out/<run>/results.json       score summary
<runs-dir>/work/                        repo mirrors, checkouts, per-instance venvs, Pythons
<runs-dir>/home/                        isolated HOME for the benchmark backend
<runs-dir>/backend.log
```

Budget tens of GB for a large run. Docker images and model weights stay in their
normal global caches.

## What the agent sees

Each repo is exported at `base_commit` into a fresh one-commit git repo, so the upstream
fix cannot be read from history. Shell network is off. Test files are excluded from the
saved patch. The hidden tests are only used by the scorer.

## Results: full SWE-bench Verified run (500 instances)

Run `202610271735`, Oct 27 to Nov 8 2026, on a MacBook Pro (Apple Silicon).

| Setting | Value |
|---|---|
| Model | `mlx-community/Qwen3.6-35B-A3B-8bit` through oMLX (local) |
| Agent | `swebench-coding-agent`, no subagents, no memory, narrow toolbox |
| Environment | Docker (`--env-mode docker`), LogicStar prebuilt images, offline |
| Limits | 5400 s and 200 tool calls per instance, 4 agent workers |
| Scoring | `--method local` with the LogicStar bundle images |

**Resolved: 327 of 500 (65.4%).**

| Outcome | Instances |
|---|---|
| Resolved | 327 |
| Unresolved (patch applied, hidden tests failed) | 169 |
| Empty patch (agent made no source edit) | 3 |
| Scoring error (tests ran past the harness's 1800 s limit) | 1 |

| Repo | Resolved / scored |
|---|---|
| scikit-learn | 28 / 32 (88%) |
| pydata (xarray) | 16 / 22 (73%) |
| pytest-dev | 13 / 19 (68%) |
| django | 161 / 231 (70%) |
| sympy | 51 / 75 (68%) |
| astropy | 12 / 22 (55%) |
| matplotlib | 18 / 34 (53%) |
| sphinx-doc | 23 / 44 (52%) |
| mwaskom (seaborn) | 1 / 2 |
| pallets (flask) | 1 / 1 |
| pylint-dev | 2 / 10 (20%) |
| psf (requests) | 1 / 8 (13%) |

Per instance the agent used 21 minutes and 65 tool calls on average (174 agent hours in
total); 494 runs ended normally and 6 hit the time limit.

How to read the number:

- **Not a leaderboard number.** Scoring used the LogicStar compressed images, which were only
  compared with the official ones on 23 instances (see above). Re-score with
  `--image-source official` before quoting it.
- **Empty patches were re-run once.** The first pass resolved 319 (63.8%) with 19 empty
  patches. Those 19 were run again after fixing two harness problems (below); 16 then produced a
  patch and 8 of the 327 resolved instances come from that retry. Retrying only the instances
  that produced nothing favours the score, so treat 63.8% as the single-attempt figure.
- **`django__django-15128`** is counted as unresolved. Its patch adds a `while` loop in
  `Query.combine` that probably never ends, so the test run was killed by the harness timeout
  both times it was tried.
- **Why runs came back empty:** the tool budget or the repeated-thought loop detector ended the
  run before any source edit; the 5400 s limit; or the agent concluded the bug was already fixed
  (`django__django-13513`). In an earlier re-run of five empty instances, agents also lost time
  mixing `/links/...` and `/testbed/...` paths, which the task prompt now spells out.

## Re-running instances

The runner skips every instance that already has a line in `preds.jsonl`, and `--retry-failed`
only redoes instances that were skipped or crashed, not ones the agent finished with an empty
patch. To redo some, remove their lines and move their trajectories aside, then run the same
command again:

```bash
cd <runs-dir>/out/<run>
cp preds.jsonl preds.jsonl.bak && mkdir -p trajectories/old
grep -v -e '"<instance_id>"' preds.jsonl.bak > preds.jsonl
mv trajectories/<instance_id>.* trajectories/old/
```

Event logs are append-only, which is why the old files must be moved. Scoring is resumable: it
keeps `score_ledger.jsonl` and scores only instances that are new, pending or whose patch
changed. `score --retry-errors` also retries instances that errored, and `score --status`
shows the counts without starting Docker. Pass `--score-batch N` to record progress after
every N instances.

To continue a full run, repeat the original command with the same `--out`
(`bench.sh run --workers 4 --offline --yes --out DIR`, no `--sample`).

## Troubleshooting

- **Image builds exit with code 137.** Docker ran out of memory; local scoring builds one image
  per worker. Use the prebuilt images (`--image-source bundle`) or `--score-workers 2`.
- **`409 ... container name is already in use`.** A second scorer is running, or a scorer was
  killed and left its container. Check `pgrep -fl score.py` and `docker ps`, then
  `docker rm -f <container>` and run `score` once.
- **`Test timed out after 1800 seconds`.** The patch makes the hidden tests hang. Look at the
  patch in `logs/run_evaluation/<run_id>/otto/<instance>/patch.diff`. It is recorded as an
  error and counts as unresolved.
- **`Not ready to run offline` / `Docker is not running`.** Start Docker Desktop and run
  `bench.sh warm` if the bundle was removed.
- **Disk.** Agent checkouts and the backend's home are not needed once a run is scored.
  `<runs-dir>/work` and `<runs-dir>/home` can be deleted, as can the `logs/`, `score_batches/` and
  `*.bak*` files in the run folder. Keep `preds.jsonl`, `results.json`, `score_ledger.jsonl`, the
  `otto.*.json` reports and, if you want them, `trajectories/`. `docker rmi` the `logicstar/*`
  and `sweb.*` images to free about 80 GB (`warm` reloads the bundle).

## Known limits

- Instances that need an old Python (for example 3.6) cannot be built with `uv` on
  Apple Silicon. They are recorded as `skipped_env`, count as unresolved, and the agent
  does not run on them.
- C-extension repos (for example astropy) can fail to build natively; the failure is
  listed in `env_notes`.
- Scoring builds x86 Docker images under emulation on Apple Silicon. The first run is
  slow; later runs reuse the cache.
- The scorer's `--method sbcli` uploads patches to SWE-bench's cloud service and needs
  `sb login`. It is never the default.
- Scores depend on the model. A local model gives a local-model score.
- Public leaderboard submissions need reasoning traces, and Verified/Lite/Multilingual
  are restricted to academic or research teams. See the `SWE-bench/experiments` repo.
