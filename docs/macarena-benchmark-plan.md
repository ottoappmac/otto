# Plan: running MacArena with Otto

Status: **Track A is built** in [`scripts/macarena/`](../scripts/macarena/README.md) and tested against a stand-in VM only;
it has never run on a real MacArena VM. Written 2026-11-14 from MacArena `main` (last push 2026-06-08) and the
paper ([arXiv 2606.06560](https://arxiv.org/abs/2606.06560)). Track B is not built.

## Where the build differs from this plan

- **An MCP server, not a `ComputerNavigator`.** The VM tools are the built-in `macarena-vm` MCP
  (`backend/builtin_mcps/macarena_vm/`), like `cad-tools`. It needed no change to `computer_voyager.py`, and the
  agent is a separate headless `macarena-agent`.
- **A guard middleware was needed.** Otto gives every agent file and shell tools the allowlist does not remove, which
  on the host would expose MacArena's evaluator scripts. `backend/macarena_guard_middleware.py` refuses everything
  but the `vm_*` tools.
- **`steps_turns` was dropped.** The MCP cannot see model turns, so only the strict count (every action is a step)
  is enforced and logged.
- **A fresh VM for every run by default.** MacArena's runner does not reset between run 1 and run 2; `--paper-protocol`
  copies that.
- **The bridge address and token travel in a file** (`~/.macarena-bridge.json` in the benchmark profile's HOME), because
  Otto resets a built-in MCP's `env` from the registry on every config load.
- **Typing is `pyautogui.typewrite('...')`**, the exact shape MacArena's env rewrites `<` for.
- Parallel VMs, the Otto-native track and the 4 checks as scripts are not built; `--noop` covers the do-nothing check.

## What MacArena is, in the parts that matter here

- A fork of OSWorld. macOS 15.5 runs in a **UTM VM on Apple Silicon** (at most two VMs per host).
  The VM image is on Hugging Face (`macpaw-research/MacArena`).
- 421 tasks: 221 OSWorld tasks moved to macOS, 151 from macOSWorld, 49 new MacArena tasks.
  Each task has an instruction, setup steps (`config`) and an evaluator that scores the final VM state from 0 to 1.
- An HTTP server inside the VM (`vm_files/server`) does everything: `/screenshot`, `/accessibility`
  (macapptree), `/execute` (runs pyautogui), `/run_bash_script`, `/start_recording`, and so on.
- The loop is `runners/lib_run_single.py`. Each step, `agent.predict(instruction, obs)` returns a list of actions,
  and `env.step()` runs them. Observations are a screenshot, plus an optional accessibility tree.
  Actions are raw mouse and keyboard input (`CLICK`, `TYPING`, `HOTKEY`, `DRAG_TO`, `SCROLL`, …, `WAIT`, `DONE`, `FAIL`).
  Most baselines send them as `pyautogui` command strings.
- Protocol used in the paper: **`max_steps 15`, 2 runs per task, `sleep_after_execution 3.0`**.

Three details in the runner affect how scores compare:

1. **A "step" is one `predict()` call, not one action.** A baseline can send several actions in one step.
2. **The task score is the max of the two runs** (`run_single_example_n_times`). That is closer to pass@2 than an average.
3. **Run 2 does not reset the VM.** It only quits visible apps (`CLOSE_ALL_APPS_COMMAND`). Files left by run 1 are still there.

## The main design problem

Otto's desktop agent (`computer-voyager`, `MacOSToolkit` in
[`src/tools/navigation/computer/macos_tools.py`](../src/tools/navigation/computer/macos_tools.py)) acts on
**the Mac Otto runs on**, through the Accessibility API, CGEvent, OCR and AppleScript. The task is inside the VM.
If nothing changes, Otto would click around on your own desktop.

So every action has to go into the VM, and nothing may reach the host. There are three ways to do that:

| Option | How | Verdict |
|---|---|---|
| A. Otto as a MacArena `mm_agents` agent | `predict()` sends one turn to Otto and returns actions | Not recommended. Otto runs its own tool loop and can't be cut into single `predict` calls without rewriting it. |
| B. Install Otto inside the VM | Run Otto in the guest | Not recommended. It is heavy, changes the VM state the evaluators read, and needs a model inside the guest. |
| **C. Remote navigator plus env bridge** | Otto runs on the host. Its desktop tools call a bridge in the harness process, and the bridge owns MacArena's `DesktopEnv` and calls `env.step()`. | **Recommended.** |

Option C keeps MacArena's own code doing reset, setup, actions, recording and evaluation.
Every action goes through one place (`env.step`), so step counting, the 15-step cap and `traj.jsonl` all live in
MacArena's own format. Otto already has the extension point it needs: `ComputerNavigator` in
[`src/tools/navigation/computer/base.py`](../src/tools/navigation/computer/base.py) was written so another
navigator can be swapped in without changing the agent.

```
 bench.sh run
  ├─ start isolated Otto backend (port 18093, own HOME, only VM tools enabled)
  └─ run.py  (runs in MacArena's Python env, imports desktop_env)
       for each task, for run in 1..2:
         env.reset(task)            ← MacArena: copy VM image / setup steps
         bridge.arm(task, budget=15)
         POST /sessions + instruction to Otto ──► macarena-agent
                                                    └─ VM navigator tools ──HTTP──► bridge ──► env.step()
                                                                                        └─► VM server
         wait for DONE / FAIL / budget / timeout
         env.evaluate()             ← MacArena's evaluator, score 0..1
         write results.jsonl, trajectories/
```

## Two tracks, reported separately

**Track A: standard (comparable).** This is the headline number. It uses MacArena's action space only:
screenshot, an optional accessibility tree, and the mouse, keyboard, `WAIT`, `DONE` and `FAIL` actions,
all sent through `env.step` as pyautogui commands. The cap is 15 steps and each task runs twice.
It has no shell, AppleScript, file access or `/run_bash_script`. Those would let the agent edit the final state
the evaluator checks without using the GUI.

**Track B: Otto-native (tool-augmented).** This uses Otto's real strengths remotely: AX controls through
macapptree, OCR, AppleScript through `osascript` in the guest, and `batch_actions`. These scores are not
comparable with the paper's baselines, so label them like MacAgentBench's "framework-augmented" evaluation.
Build Track B after Track A works.

### What counts as a step (Track A)

MacArena counts one `predict()` call as one step, however many actions it returns. Otto has no `predict()`.
The bridge counts:

- **`steps_strict`**: every action is one step. **This is the cap that's enforced** (15). It is stricter than the
  baselines, so it never overstates Otto.
- **`steps_turns`**: actions grouped by model turn (the bridge groups calls that share a tool-call batch id).
  This is logged so the step use can be compared with the baselines.

Observation calls (`screenshot`, `accessibility_tree`) don't count as steps, because MacArena hands the agent a
free observation every step. They are capped at 3 × the step budget, so a model can't loop on them.
Otto's own tool-call budgets stay on as a backstop.

## Code to add

Keep the same layout as [`scripts/swebench/`](../scripts/swebench) and [`scripts/benchcad/`](../scripts/benchcad).
MacArena lives in a checkout next to Otto (`../MacArena`) at a pinned commit SHA, with its own Python env.
`run.py` runs in that env and talks to Otto over HTTP, the same way the BenchCAD scorer is split.

### Harness: `scripts/macarena/`

| File | What it does |
|---|---|
| `README.md` | Setup (UTM, VM import, MacArena env, commit pin), how to run, what the numbers mean, status |
| `bench.sh` | `check` (UTM, `utmctl`, VM present, MacArena env, Otto venv, vision model set), `run`, `stop`, `report`. Refuses to reuse a backend started with other settings (copy BenchCAD's `backend.mode` file). |
| `start_backend.sh` | Isolated backend on port **18093**. Own HOME, memory off, Auto depth / Jev / title generation off. **Disables every host-acting tool**: the local `MacOSToolkit`, `macos_*` MCPs, Playwright, shell. |
| `run.py` | Loads a task list (`--source osworld,macosworld,macarena`, `--ids`, `--domain`, `--sample`), then for each task: `env.reset`, arm the bridge, create an Otto session, send the instruction, answer approvals the way swebench `run.py` does, wait, `env.evaluate`, start and end recording. Writes resumable `results.jsonl`, with **both runs' scores** per task. |
| `bridge.py` | Small HTTP server inside `run.py` that holds `DesktopEnv`. It has `GET /obs` (screenshot plus optional tree), `POST /act` (one action → pyautogui string → `env.step`), `POST /done`, `POST /fail` and `GET /budget`. It enforces the caps, writes `traj.jsonl` and step PNGs in MacArena's layout, and returns `budget exhausted` once the budget is spent. |
| `report.py` | Success rate by source (OSWorld / macOSWorld / MacArena) and by domain, for **max-of-2 (MacArena protocol)**, **mean of runs (pass@1)** and **run 1 only**. Also step usage, timeouts and infra errors (counted apart from failures). |
| `tasks.py` | Reads MacArena's task JSONs and meta files and checks that each task's VM (`osworld` image vs. `macarena` image) is present |

### Otto changes

| File | Change |
|---|---|
| `src/tools/navigation/computer/vm_navigator.py` (new) | `RemoteVMNavigator(ComputerNavigator)`. Its tools are `screenshot`, `accessibility_tree`, `click(x, y, button)`, `double_click`, `right_click`, `move_to`, `drag_to`, `scroll`, `type_text`, `press`, `hotkey`, `key_down` / `key_up`, `wait`, `done` and `fail`. Each one calls the bridge (`OTTO_VM_BRIDGE_URL`). Coordinates are in VM screen space, and every action returns the new screenshot. The system prompt explains the VM coordinates and the DONE/FAIL contract. |
| [`src/agents/computer_voyager.py`](../src/agents/computer_voyager.py) | Choose the navigator from config (`OTTO_COMPUTER_NAVIGATOR=macos|macarena_vm`, default `macos`) instead of always building `MacOSNavigator`. |
| [`backend/agent_library.py`](../backend/agent_library.py) | A headless `macarena-agent`, like `benchcad-agent` and `swebench-coding-agent`. It never asks the user, always ends with `done` or `fail`, and is told the step budget. No subagents in Track A. |
| [`backend/session_manager.py`](../backend/session_manager.py) | Add `macarena-agent` to `_AGENT_TOOL_ALLOWLIST` (VM tools only) and `_AGENTS_WITHOUT_MEMORY` |
| [`backend/action_policy/catalog.py`](../backend/action_policy/catalog.py) | List the VM tools as their own category, so the benchmark backend can auto-approve them while host desktop control stays gated |
| `tests/test_vm_navigator.py` (new) | Uses a mock bridge to test the action → pyautogui mapping (including MacArena's `<` typing fix and hotkey names), the step and observation caps, the budget-exhausted reply, and that the agent's allowlist has no host tools |

## Checks before trusting a number

1. **Environment check without Otto.** Run MacArena's own runner with one baseline (for example UI-TARS-1.5-7B)
   on a few tasks. If that fails, the problem is UTM or the VM, not Otto.
2. **Do-nothing agent.** Send `DONE` straight away on every task. Any task that scores above 0 passes by default.
   List those and report them.
3. **Host isolation test.** During a run, check that the host's frontmost app and mouse position never change,
   and that there are no `osascript` or Playwright processes on the host.
4. **Repeatability.** Run the 49 MacArena tasks twice and report the difference between the runs before quoting anything.

## Phases

1. **Setup (half a day):** install UTM, import the VM, set up the MacArena env, run the environment check above.
2. **Track A, one task (2–3 days):** `vm_navigator.py`, `bridge.py`, `run.py` for a single task, and the tests.
3. **Subset:** the 49 MacArena-native tasks. They are macOS-native, have their own evaluators and are the cheapest signal.
   Then the do-nothing agent and repeatability checks.
4. **Full 421**, two VMs in parallel at most (`--workers 2`, each with its own VM copy and bridge port).
5. **Track B** (Otto-native tools in the guest), reported apart from Track A.

## Things to decide or know

- **Licence.** The macOSWorld tasks (151 of 421) are **CC BY-NC 4.0, non-commercial only**. If Otto is
  commercial, check this before running or publishing results on them. OSWorld tasks are Apache 2.0, and
  MacArena's own code and tasks are MIT.
- **A vision model is needed for Track A.** With a text-only model, use the accessibility tree only.
  Report that variant separately.
- **Memory.** Each macOS VM takes a lot of RAM, and it competes with a local MLX model. With a local model,
  start with `--workers 1`.
- **Two VM images.** OSWorld and macOSWorld tasks use the hand-configured VM (`osworld`). MacArena tasks use the
  scripted one (`macarena`). `tasks.py` should pick the image for each task.
- **Report both protocols.** Report max-of-2 to match the paper, and pass@1 because it is the more honest number.
