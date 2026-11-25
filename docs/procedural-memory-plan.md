# Procedural memory: plan

A memory of what worked and what failed when Otto's agents drove macOS and the browser: tool calls, AppleScript /
JXA scripts, Playwright code. Saved after each run with a success label, retrieved by similarity, and offered to
`macos-desktop-agent`, `browser-agent` and the benchmark's `macarena-desktop-agent` through new tools. Repeated
app-side tasks can also replay a saved script directly.

Covers both the MacArena harness (`scripts/macarena/`) and the Otto app. Phases 0-2 and the saving half of Phase 3
are built (off by default); see [Implementation](#implementation-2026-11-23) for what exists and how to run it.

Revised 2026-11-23 after a survey of research and product practice:
[`reports/Agent procedural memory practice.md`](../reports/Agent%20procedural%20memory%20practice.md). Most of its
numbers are from 2025-2026 preprints; check the paper tables before quoting them.

## Short answer: is it feasible?

**The plumbing, yes, and most of it exists.** The value, unproven. The literature says the hard part is not
retrieval but **admission**: whether a run stored as a "success" really succeeded.

What the evidence says (details and sources in the report):

- Published gains are about +4 to +12 points: AWM on WebArena (GPT-4) 23.5 → 35.5; Agent S on OSWorld 11.2 → 20.6
  for the whole system, memory alone about 5-8 points; ReasoningBank on WebArena +4.6 to +8.3.
- A 2026 budget-matched study (arXiv 2606.15017) found a plain agent given 15 steps beat AWM, ASI and ReasoningBank at
  10 steps on WebArena, and that 50-60% of their stored "successes" came from runs that had failed. So part of the
  published gain is extra steps and tokens, and self-judged labels poison memory.
- Prompted judges top out around 0.70 precision on "success" (AgentRewardBench). Agents' own "done" is worse: in one
  OSWorld study 90% of failed runs ended with a success claim. Our own data agrees: 59 of 174 benchmark runs passed,
  so if every run claims success, self-report precision is at most 0.34.
- Retrieving fewer items is better: ReasoningBank scored 49.7 at k=1, 46.0 at k=2, 44.4 at k=4. No paper ablates the
  embedding model.
- Failures help as short lessons (ReasoningBank 46.5 → 49.7) and hurt when mixed in as procedures (AWM 44.4 → 42.2).
- Products (Skyvern, Stagehand, Browser Use, Simular) replay a saved script exactly on a precise key and fall back to
  the agent when it breaks. Consumer agents mostly store what the user wrote or showed them. Only Browser Use learns
  procedures automatically, gated by a personal-data filter and score-based retirement.

Otto has one real advantage: Accessibility reads and AppleScript queries make post-condition checks exact and cheap,
which screenshot-based agents cannot do. That is the route to high-precision labels.

What already exists:

| Need | Already in the repo |
|---|---|
| Vector store + embedder | `backend/embedding_index.py`: sqlite-vec, `sentence-transformers`, `chunks.source_type` column |
| Full tool-call trace per run | `sessions/{id}.messages.json` (JSONL rows: `user`, `agent`, `tool_call`, `tool_result`), written as the run streams, unaffected by compaction |
| App-side success label | `backend/eval_runner.py`: LLM judge (DeepEval, `g_eval`), `eval_overall_score` stamped on session meta |
| Benchmark success label | MacArena evaluator score in `out/*/results.jsonl`, with `trajectories/{id}.run1.messages.json` |
| Runnable browser scripts | `src/tools/navigation/web/playwright_script_extractor.py` splices `### Ran Playwright code` blocks |
| Hand-written app knowledge | `macos_app_cards.py` + the `app_guide` tool (pattern to copy for the new tool) |

Labeled data on disk today: **174 benchmark runs** (59 passed, 115 failed; osworld 159, macosworld 15), median 37
steps, every task id seen once.

Open questions that decide whether this is worth it:

1. **Can the app-side judge reach high precision on "success"?** The bar is absolute (Phase 0 gate 1), not "better
   than self-report".
2. **Are neighbours useful?** With 174 unique tasks, the nearest run is always a different task. It helps only when
   it shares an app and a sub-skill.
3. **Does memory beat a plain agent given the same extra steps and tokens?** Only a budget-matched A/B answers this.
   A null result there is consistent with the 2026 literature; the app-side replay and lesson paths are then judged
   on their own metrics.

## What "complete" means (labels)

Each saved run gets a label (`success` / `failure` / `unknown`), a `label_source` and a confidence. Sources, from
most to least trusted:

| Source | Where | Trust |
|---|---|---|
| `benchmark` | MacArena evaluator score (1.0 / 0.0) | High, but some checkers can be satisfied by a stub: an upper bound on real success |
| `user` | Thumbs up / down or "this didn't work" on the run (Phase 3) | High; overrides everything else |
| `verified` | A **read-only post-condition probe** the host runs after the agent stops: a file read, `get_control_value`, `read_table` or an AppleScript property query, derived from the task's key points. A read-after-write already in the trace counts as weaker evidence. App side only: in benchmark mode it would compromise the evaluator's independence | Medium-high |
| `judge` | `eval_runner` with a trajectory-aware rubric: task key points checked against tool results and key frames; the agent's final message is **excluded**; 3-4 samples that must agree unanimously | Medium, calibrated in Phase 0 |
| `self_report` | Agent's final message / `vm_done` | Low. Never enough on its own to save a run as `success` |

Rules:

- A run is served as a **procedure** only if `success` from a source whose measured success precision is at least
  0.90 (Phase 0). Below that it is `unknown` and not served as a procedure.
- **Lessons** (below) can come from successes and failures. From a judged failure only if judged-failure precision
  also clears 0.90; judges lean toward false *success*, so a judged failure is likely more reliable than a judged
  success, but this is measured, not assumed.
- Failures are never served as step sequences.
- Infra errors (`status != ok`, transient error codes from `classify_error_code`) are never stored.
- A label can change later (user feedback); the record keeps its history.

## What is stored

Three units. The episode is the record; segments and lessons are what retrieval returns.

```
episode
  id, session_id, agent, track (app | benchmark), task_id (benchmark only)
  task_text            the user's request / task instruction
  intent               LLM-normalised, values stripped: "Calc: sort a table by one column, descending"
  apps, app_versions   apps touched (launch_app / activate_app / tell application / page origins)
  start_state          frontmost app and window title at the start (optional retrieval signal)
  label, label_source, confidence, created_at, otto_version, model
  steps[]              compacted calls (below)
  segments[]           step ranges per sub-goal, each with its own intent and vector
  artifacts[]          final osascript scripts, extracted Playwright spec, with parameter slots
  lessons[]            up to 3 per run (below)
  usage                times retrieved, outcomes of the runs that used it
```

Compaction of tool calls (deterministic, no LLM):

- Drop pure reads that did not change the plan (repeated `get_screen_controls`, screenshots); keep the last read
  before each write so the step shows what was targeted.
- Drop calls that errored and were retried; keep the retry that worked. Collapse the loop guard's repeats.
- Keep arguments that transfer (AX labels, menu paths, roles, app names, script text). Mark arguments that do not
  (`click_at` coordinates, element ids that change per launch) as `nontransferable`; show them as "clicked
  <label from the read before>" or drop them.
- Truncate each result to a status line ("ok", "error: …", the value read).

For a median 37-step run this leaves roughly 8-15 steps.

Scripts are the most reusable thing in the trace. A final `run_osascript` that succeeded and a Playwright spec from
the extractor are stored verbatim as artifacts (capped, for example 4 KB each), with the task's values replaced by
named parameter slots where they can be found.

**Lessons** follow ReasoningBank's format: a title, one sentence, and short content, at most 3 per run, from
successes and failures alike. They are phrased as **checks**, not prohibitions ("after File > Export To > CSV,
confirm the file exists"), so a mislabelled lesson costs one verification step instead of blocking a correct path.

**On "just store the tool calls":** the compacted tool calls plus the script artifacts are the source of record. An
LLM summary as the only copy would lose the exact menu path or AX label, which is the useful part. The LLM writes
only the `intent`, the segment intents and the lessons.

## Retrieval and context budget

1. **Hard filters first:** same agent family (desktop vs browser), at least one shared app (or the same site origin
   for the browser), label rules above. The app is the strongest signal of a reusable procedure.
2. **Score:** embedding cosine on `intent` (raw task text is dominated by file names and values), blended with BM25
   (sqlite FTS5) on task text for exact menu and feature names. Optionally add `start_state`, as Agent S does.
3. **Return one item by default:** the best segment or episode as a recipe of at most ~12 one-line steps, with its
   lessons in the same item. About **600-800 tokens**. **Nothing** below the similarity floor set in Phase 0.
4. A second item, or the full steps or script of one item, only through an explicit follow-up call by id.
5. Never re-inject on every step: per-step injection was the main token cost in the budget study.

Embedding model: start with the configured `sentence-transformers/all-MiniLM-L6-v2` (384-d, already loaded for
memory). Intents are short single sentences. No published ablation shows the model matters much; Phase 0 compares
it with one stronger small model (for example `BAAI/bge-small-en-v1.5`) and switches only on a clear difference.
Procedures get their own vector table, so a model change re-embeds only procedures.

## The tools

`recall_procedure(task: str, app: str | None = None, id: str | None = None, detail: str = "recipe")`

- Added to `macos-desktop-agent`, `browser-agent` and `macarena-desktop-agent` (and to
  `macarena_native.AGENT_TOOLS`, since `macarena_guard_middleware` refuses anything not listed). It runs on the
  host and reads only the procedure store, so it does not break the benchmark rule that the agent changes VM state
  only through its desktop tools.
- **Counts against the step budget** like any other call. It is not a free read; otherwise the comparison with a
  no-memory agent is not fair.
- Prompt guidance (generic, no task nouns, same rule the v9 prompt test enforces): "Before a multi-step task in an
  app, you may call `recall_procedure`. Examples are from other tasks: confirm each step against what is on screen;
  never copy values from them."

`replay_procedure(id: str, params: dict)` (app side only, Phase 3)

- Offered only when a request matches a stored episode on an **exact key**: normalised intent + app + app version +
  parameter slots. Benchmark tasks never repeat, so this path is judged in the app, not by the benchmark score.
- Checks a precondition first (frontmost app, expected window or control present), runs the saved `run_osascript` /
  JXA or Playwright artifact with the parameters filled in, then runs the post-condition probe.
- On any error or failed check, falls back to the normal agent loop and records the fallback.
- Never rewrites a stored script by itself. A fixed version from a fallback run is stored as a candidate and
  promoted after a second verified success.
- Asks the user before running a script that writes, sends or deletes. Goes through the existing action policy.

## Benchmark contamination (designed in, not added later)

Memory built from MacArena runs and served back to MacArena runs is train-on-test unless guarded. `app_guide` cards
already follow a rule that nothing from a task or its checker goes in; this needs the equivalent:

- Exclude from what a task can retrieve: the same `task_id`, any task whose instruction is near-identical (cosine
  above a threshold), and any task sharing a checker function.
- Serve from a **frozen snapshot**, not a store that grows during the run, so results do not depend on task order.
  Build snapshots by **group k-fold over near-duplicate instruction clusters** (same template, other values), not
  over apps: sharing an app is what memory is for, sharing a task is leakage. The osworld/macosworld split is too
  lopsided (159 vs 15 runs) to use as the folds.
- If memory that grows during a run is ever tested, run at least 3 shuffled task orders and report the spread.
- Audit each snapshot for stored successes the evaluator scored as failures (zero by construction for
  benchmark-labelled snapshots; the check guards against bugs).
- Never ingest public OSWorld trajectories (for example the published OSWorld-Verified runs).
- Report memory-on scores separately from memory-off, with the snapshot id, its size and the fold.
- Benchmark episodes never flow into the app's store by default. The benchmark backend runs with its own HOME, so
  the stores are already separate; add explicit `export` / `import` commands for when they should be shared.

## Phases

### Phase 0: offline feasibility (about a day, no product code)

Script under `scripts/procmem/`, reading existing `out/*/results.jsonl` and trajectories.

1. **Judge calibration.** Run the judge as configured above (final message excluded, key points, 3-4-sample
   unanimous vote) on the 174 benchmark runs and compare with the evaluator. Report success precision with a Wilson
   lower bound (with about 59 positives the interval is roughly ±0.08), recall, and failure precision. Measure
   `self_report` alone on the same set.
   *Gate:* success precision at least 0.90 to admit judged successes; the same bar on failure precision to admit
   lessons from judged failures. Otherwise app-side memory admits only `user` and `verified` labels.
2. **Retrieval check.** For each task, retrieve the top 3 from the other runs, with the contamination exclusions
   applied. Label (LLM, then spot-check) whether each shares an app and a sub-skill that would help. Compare MiniLM
   vs one stronger model, and `intent` vs raw task text. Set the similarity floor here.
   *Gate:* if fewer than about a third of tasks get one useful neighbour, the pool is too small; collect more runs
   before Phase 2, or expect a small effect.
3. **Compaction check.** Compact 20 runs; read them. Confirm a recipe plus lessons fits 600-800 tokens and keeps the
   menu paths, labels and scripts.

### Phase 1: store and ingest (about 3-4 days)

- `backend/procedure_store.py`: tables `episodes`, `segments`, `lessons`, `artifacts`, label history, usage; vectors
  in a sibling sqlite-vec table using the existing `_Embedder`; FTS5 on task text.
- `backend/procedure_ingest.py`: `messages.json` → episode (compaction, app detection, segmenting, script
  extraction with parameter slots), one LLM call for intents and lessons.
- Hooks: app, after `eval_runner` finishes (`_stamp_session_meta`); benchmark, `scripts/macarena/run.py` after
  `/ctl/evaluate`, labelled `benchmark`. Backfill command for existing sessions and benchmark runs.
- Tests: compaction on fixture transcripts, label and admission rules, the contamination exclusions, infra errors
  not stored.

### Phase 2: retrieval tool and benchmark A/B (about 3-4 days, plus run time)

- `recall_procedure`, prompt lines, allowlist entries. Tests that it respects the exclusions and the token cap.
- Arms, all with the same model and a frozen snapshot, **3 runs per task per arm**:
  1. memory off, standard budget;
  2. memory off with **extra steps and tokens equal to the memory arms' measured mean overhead**;
  3. memory on, tool only;
  4. memory on, one item injected once at task start (only above the floor);
  5. memory on, hybrid: a one-line pointer (title and id) injected, details through the tool.
- Report mean ± std, Any-of-3 / All-of-3, steps and tokens per task, tool call rate, and results per app and task
  class.
  *Gate:* ship retrieval to the app only if a memory arm beats arm 2 beyond noise and no task class gets worse.

### Phase 3: Otto app (about 1-2 weeks)

- `verified` post-condition probe after app runs.
- `replay_procedure` with precondition, post-condition, fallback and candidate-revision handling. Measure steps
  saved, fallback rate and user-confirmed success. This path does not wait on the Phase 2 gate.
- Thumbs up / down on a run in the Runs detail view, writing a `user` label (and an optional "what went wrong").
- A Procedures list (search, label, source, delete, "don't learn from this run"), in Settings or under Memory.
- Settings switch, off by default at first: "Learn from completed runs". Respect the existing memory on/off switch.
- Privacy: strip typed text from `type_text` / `type_into_control` arguments by default (keep the target label),
  never store anything from a field the action policy marks as secret, and run a personal-data filter on lessons
  and intents before saving.

### Lifecycle (from Phase 1, enforced from Phase 3)

- Track per item how often it is retrieved and how the runs that used it ended. Retire items whose use correlates
  with failure.
- Stamp `otto_version` and the app version. Re-check or retire items when the app version changes, and drop items
  that call tools that no longer exist.

### Later, only if the gates pass

- Promote repeated, verified segments into reviewed `app_guide` cards.
- Use the same labelled episodes as Distill training data (the Dataset tab already filters by eval failure).

## Implementation (2026-11-23)

Package `backend/procedural_memory/`; tests in `tests/test_procedural_memory.py`.

| Module | Does |
| --- | --- |
| `compact.py` | Splits a transcript into runs (a delegated `task` call's tagged subagent rows, or a direct agent's whole session), compacts each into steps, segments, app facts and artifacts. Coordinates dropped, indices turned into labels, typed text and field values slotted. |
| `store.py` | SQLite per store: episodes, items (recipe / fact / lesson), artifacts, labels, usage, float32 vectors, FTS5. |
| `policy.py` | Label ranks and admission (`self_report` never admits; judge only when calibrated). |
| `enrich.py` | LLM intents and up to 3 lessons phrased as checks; deterministic fallback. |
| `retrieve.py` | Hybrid dense 0.75 + BM25 0.25, app and agent-family filter, floor abstain, k=1 recipe + up to 3 notes, 800-token cap, near-duplicate exclusion. |
| `judge.py` | Trajectory judge (final message hidden, unanimous over samples), Wilson gate. |
| `service.py` | Runtime (app or benchmark), save → pending, label → admit/retire, outcome recording. |
| `tool.py` | `recall_procedure` for `macos-desktop-agent`, `browser-agent`, `macarena-desktop-agent`. |

**Saving.** Every learning-agent run is saved as a `pending` episode with no vectors (`session_manager._maybe_save_procedures`,
its own hook, since `auto_evaluate` is off by default). A label admits it: the evaluator's pass/fail
(`eval_runner._label_procedures`, source `judge`, so pending until calibrated), the user or a verified check
(`POST /api/procedures/label`), or the benchmark score (`/api/procedures/benchmark/ingest`). Relabelling rebuilds
the items; excluded or deleted episodes drop theirs.

**Learning from failures.** A failed run gives (1) the segments its trace confirmed, never the last one; (2) no-op
facts ("'search' changed nothing (3 tries)"); (3) lessons phrased as checks; (4) usage outcomes, so items whose
recalls keep failing get a small negative prior. Never its whole step sequence as a recipe.

**Separate stores.** App: `<app data>/procedures.db`, enabled by `procedural_memory.enabled`. Benchmark: a
read-only snapshot (`OTTO_PROCEDURE_SERVE_DB`) plus a fresh `out/<run>/memory/write.db` per run; the backend refuses
any benchmark path inside the app data dir, and `start_backend.sh` refuses a snapshot under the real
`~/Library/Application Support/Otto`.

**Benchmark results.** Each row carries `memory_mode` (`off` or `snapshot:<id>`), `snapshot_id`, `snapshot_size`,
`memory_calls`, `memory_turns`, `memory_used` (item id, episode, source task, step, query), `memory_abstained`,
`memory_tokens`, and `memory_leak` if an item came from the same task. Every call is also in
`memory_calls.jsonl` in the task's run folder. A `recall_procedure` call costs a step (free if it shares a
reply with an action in turns mode) and is refused past the budget. `report.py` refuses to mix memory arms or
rows with a leak, and prints pass rates for runs that did and did not get a recipe.

**Commands.**

```bash
# Phase 0
.venv/bin/python scripts/procmem/calibrate.py --runs ../otto-macarena-runs/out/* --out procmem-phase0 [--judge --samples 3]
# a fold's snapshot (writes <db>.test-ids: run only those tasks against it)
.venv/bin/python scripts/procmem/build_snapshot.py --runs ../otto-macarena-runs/out/* --folds 3 --fold 1 --out snap-f1.db --llm
# serve it
scripts/macarena/bench.sh up --native --memory snap-f1.db
# snapshots and queries use MiniLM by default (needs sentence-transformers, now in pyproject/requirements);
# leave MACARENA_MEMORY_FLOOR unset for MiniLM (0.45 default). Only a hash-256 snapshot needs ~0.3
# queries use the snapshot's own embedder unless MACARENA_MEMORY_EMBEDDER is set; rows record memory_floor and
# memory_embedder, and the report refuses to pool different floors
```

**Phase 0 smoke result** (hash embedder, deterministic intents, 203 runs). `self_report` success precision 0.364
(95% CI 0.294-0.439) and the trace's last segment 0.42: neither is close to trusted, as the research predicted.
Top-1 neighbour shared the app in 50% of tasks. Rendered recipes: median 215 tokens, max 394 (under the 800 cap).
Deterministic segment intents are weak ("Google Chrome: Search"); rerun with MiniLM and `--llm` intents, and with
`--judge`, before setting the floor and before Phase 2.

**Getting the agent to call it (2026-11-23).** In live smoke runs the benchmark model (local Qwen3.6-35B-A3B,
temperature 0) never called `recall_procedure`, with the tool description alone or with a section at the end of the
prompt. Forcing it is not available: the local oMLX server ignores `tool_choice` (named or `required`). Injecting a
recall at the start was tried and removed: without the agent naming the app it matched across apps (a Script Editor
task got a Safari recipe). What works is prompting. Offline, on the real system prompt and 12 real first messages:

| Prompt | First reply calls `recall_procedure` |
| --- | --- |
| Step 0 of the Workflow in the system prompt | 7 / 12 |
| Step 0 + one line after the task in the first message (`tool.FIRST_MESSAGE_NUDGE`) | 12 / 12 |

Both are added only when the tool is bound, so a memory-off run gets exactly the prompt it had; the line goes into the
model's copy of the first message, never the transcript. The model names the app itself ("Chrome",
"LibreOffice Calc"); `tool.resolve_app` maps that to the stored name ("Google Chrome", "LibreOffice"). Against the
smoke snapshot those 12 queries returned memory for 3 (LibreOffice x2, GIMP), all from other tasks; the rest were apps
the snapshot has nothing for (Keynote, QuickTime, Safari) or below the floor. `scripts/procmem/show_memory.py <run>
--text` shows per task what was asked, what came back and from which task.

**Live memory (2026-11-24).** `bench.sh up --native --memory live` (flag: `off` is the default, a `.db` path serves
a frozen snapshot). Recall then reads the run's own write store (`out/<run>/memory/write.db`, empty for a new
`--out`), so every finished task is recallable by the tasks after it in the same run, as in the app:

- Passed and failed runs both count. A failed run gives its lessons (the LLM is asked what to check and what to try
  instead next time) and any segments that worked before it went wrong; its failed step sequence is never offered as
  a recipe.
- Similar tasks (same template, other values) are recalled; only stored runs with the very same task text are left
  out. `MACARENA_MEMORY_SAME_TASK=1` allows those too, for testing a repeat (`--runs 2`).
- `--ids` run in the order given. The runner waits for each task's save (up to 15 minutes: lessons are an LLM call)
  before the next one starts.
- Rows record `memory_mode: live` (and `memory_same_task`). The scores are order-dependent, a learning curve rather
  than a frozen-memory score, so the report never pools them with snapshot or memory-off rows, and refuses rows that
  recalled their own task.

Smoke test: `scripts/procmem/memsmoke.sh` (task A, then the similar task B, which should recall A) or
`scripts/procmem/memsmoke.sh same` (one task twice). It prints `SMOKE PASS` or what failed.
`scripts/procmem/query_memory.py <store.db> "<query>" --app <App>` asks a store what recall would return, offline.

**Not built yet:** `replay_procedure`, the app's post-condition probe (`verified` labels), UI (thumbs, a Procedures
list), hindsight relabelling, contrastive pass/fail lessons, batching changes.

## Risks

- **Wrong labels teach wrong habits.** The main documented failure of this kind of system. Mitigated by the 0.90
  precision gate, excluding the final message from the judge, the post-condition probe, and lessons phrased as checks.
- **Gains that are only extra budget.** Mitigated by the budget-matched arm and by counting `recall_procedure`
  against the budget.
- **Negative transfer.** An example from a similar-looking task leads the agent down the wrong path. Mitigated by
  k=1, the similarity floor, the "confirm against the screen" prompt line, and results per task class.
- **A wrong replay is worse than a slow run.** Mitigated by exact keys, pre- and post-condition checks, fallback,
  no automatic script rewrites, and confirmation for scripts that write or send.
- **Staleness.** App versions and Otto's tools change. Mitigated by version stamps and outcome-based retirement.
- **Benchmark leakage.** Covered above; a memory-on score is not reportable without the snapshot, fold and
  exclusions.
