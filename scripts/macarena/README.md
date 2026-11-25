# MacArena harness for Otto

Runs Otto on [MacArena](https://github.com/MacPaw/MacArena) tasks (macOS 15 in a UTM VM,
[paper](https://arxiv.org/abs/2606.06560)) and scores them with MacArena's own evaluators. Two tracks:

| Track | Agent | What it measures |
|---|---|---|
| standard (default) | `macarena-agent`: screenshot + raw mouse/keyboard (`vm_*`) | MacArena's own action space; comparable with the paper |
| native (`up --native`) | `macarena-desktop-agent`: Otto's macOS desktop agent prompt and its Accessibility tools | Otto's desktop agent as the app runs it; **not** comparable with the paper's baselines |

**Status: built and tested against a stand-in VM only. It has never run on a real MacArena VM, so there is
no Otto score yet.** See [Status](#status) for exactly what is and is not verified.

## How it works

```
 bench.sh up ─┬─ bridge.py        (MacArena's Python env) owns the UTM VM through MacArena's DesktopEnv
              └─ Otto backend     (port 18093, own HOME, only the macarena-vm MCP on, memory off)

 run.py, for each task and run:
   bridge /ctl/reset ──► fresh VM copy + the task's setup steps
   Otto session (macarena-agent) ──► vm_* tools ──► macarena-vm MCP ──► bridge /agent/act ──► env.step ──► VM
   bridge /ctl/evaluate ──► MacArena's evaluator on the final VM state ──► score 0..1
```

The agent has only the `vm_*` tools: screenshot, click, double/right click, move, drag, scroll, type, press,
hotkey, wait, done, fail. That is MacArena's own action space. It has **no shell, file or AppleScript tool**,
because those would let it change the state the evaluators read without using the GUI. Two things enforce it:
the MCP exposes only those tools, and `backend/macarena_guard_middleware.py` refuses every other tool the agent
tries (Otto gives all agents file and shell tools that the per-agent allowlist does not remove; on the host those
would expose MacArena's task files and evaluator scripts).

The bridge never accepts code: the agent sends a small dict, `actions.py` validates it (ranges, key whitelist, text
quoted with `repr`), and only a command built there reaches the VM. Two random tokens keep the agent away from
`/ctl/*` (reset, evaluate).

## Otto-native track

The agent is Otto's own `macos-desktop-agent`: the same prompt (plus a short benchmark section: step budget,
`vm_done` / `vm_fail`, no user) and the same `macos-native` tools the app builds (`get_screen_controls`,
`press_control`, `type_into_control`, `launch_app`, `batch_actions`, ...), with Otto's loop guard and wrappers. The
agent and its model run on this Mac. Each tool call runs in the VM, because that is where the apps being tested are:

```
 Otto backend (native_backend.py) ── macos-native tool ── RemoteMacOSToolkit ──► bridge /agent/native
     ──► guest/otto_tools_server.py in the VM ──► Otto's MacOSToolkit, unmodified ──► the VM's apps
```

- `bench.sh up --native` builds a bundle (Otto's toolkit files copied verbatim, the guest server, and pydantic /
  langchain-core / python-dotenv / aiohttp pinned to this repo's venv, built for the VM's Python 3.12), and the
  bridge uploads and starts it in every fresh VM through MacArena's own in-guest server. Nothing is installed into
  the VM's own Python environment and the VM needs no network.
- Steps: Otto's action policy already files these tools as reads or `desktop_control`. Reads are free (capped at
  3x the step budget); every acting call is one step; `batch_actions` is one per action inside it. `vm_done` /
  `vm_fail` are a step, as in the standard track. No 3 s pause after actions (Otto's own delays apply).
- Safety: only `scripts/macarena/native_backend.py` installs the remote toolkit, and only in the isolated benchmark
  backend. `backend/macarena_guard_middleware.py` refuses this agent's desktop tools unless they are the remote
  ones, so they cannot act on this Mac, and refuses file tools, this host's shell (`execute`) and the standard
  track's `vm_*` mouse tools. Commands in the VM go through `run_osascript` with `do shell script`.
- Records: `native_calls.jsonl` (every call, reads included, with cost and result) next to `traj.jsonl` (each step,
  with a screenshot of the VM after it).

```bash
scripts/macarena/bench.sh up --vm macarena --native
scripts/macarena/bench.sh run --sources macarena --ids <id> --runs 1 --timeout 1800 --out ../otto-macarena-runs/out/native-<name>
```

Keep native and standard results in different `--out` folders; `run.py` and `report.py` refuse to mix them.

### Harness settings (Otto-native track)

These came out of reading 22 runs (calls, failures, timing) and the review of a 155-task OSWorld run
(`docs/reports/osworld-native-100step-failure-review.md`). Each run records the settings it used and
`report.py` refuses to pool runs with different settings, so results from before and after stay apart.

| Setting | What it does | Why | Off switch |
|---|---|---|---|
| Observe after acting | After a click, press, typing or hotkey, the result also lists the frontmost app's controls (Otto's own `get_screen_controls`, one free read, cut at 8000 characters). | The call right after an action was a read 73% of the time: one model turn per action spent only on looking. | bridge `--no-observe-after` |
| Failure notice | When 4 acting calls in a row fail (an error, or an AppleScript that ran and failed), the 4th result says so and asks for a different kind of approach; again at 8, 12, ... | Runs had up to 6 failed AppleScript UI-scripting variations in a row, each with different arguments, which Otto's identical-call loop guard cannot see. | bridge `--fail-notice-after 0` |
| Repeat notice | When the same acting call (same tool, same arguments) is made 3 times in a row, the 3rd result says the screen did not change and asks for a different way; again at 6, 9, ... Arrow keys, Tab, Return and scrolling do not count. | In the 155-task OSWorld run one agent made about 25 identical `click_at(42,152)` calls; the failure notice missed them because each call succeeded. | bridge `--repeat-notice-after 0` |
| Finish nudge | When the agent's turn ends without `vm_done` / `vm_fail` while the run is still open (steps and at least 60 s left), the runner sends one message telling it to check the result and call `vm_done` or `vm_fail`. | 19 of the 155 runs ended with a text reply and no `vm_done` / `vm_fail`; some had stopped halfway. | `MACARENA_FINISH_NUDGES=0` or run `--finish-nudges 0` |
| Update nags closed | At reset the bridge closes Chrome's "Can't update Chrome" window through the VM tool server, before the starting state is read. | It was in front at the start of 12 tasks; agents spent steps on it or typed into it. | bridge `--keep-nags` |
| Task setup waited for and checked | The bridge runs each task's `pre_command` itself and waits for it (after Finder answers AppleScript), keeps its output and exit code in the run's `setup.json`, checks that the folders it opens exist, and runs it once more when that is safe (its AppleScript only closes or quits). A setup that still fails is recorded as `setup_status: failed`; the run goes on and is scored as usual. Both tracks. | MacArena only starts `pre_command` and never looks at the result; the bridge then waited 10 s. In the 2026-11-24 macOSWorld runs the 29 file tasks whose setup starts with `osascript ... Finder to close windows &&` never had their folder (29 of 29 runs), and one task's 15 s setup was still filling a spreadsheet when the agent started. | `MACARENA_SETUP_MODE=launch` (bridge `--setup-mode launch`) |
| Harness folders refused | Native calls whose arguments name `~/vm_files`, `~/Benchmark_Backup` or `/tmp/otto-bench` are refused (counted as invalid, nothing run). A search that does not name them can still list them. | With a setup broken, agents read MacArena's server code for ~30 steps and deleted a file from the backup copy every file task restores from. No instruction names these folders. | none |
| No tools after finish | Once `vm_done` / `vm_fail` is accepted the model is offered no tools, so it answers instead of calling more. | Runs kept calling tools after the run was over. | none (no effect on the score) |
| Host prompt sections removed | The app's session sections about this computer (`Session Context` with this host's folder, `/output/`, subagents, scheduling, mapped projects) are dropped from the benchmark agents' prompt. | The agent typed this host's home folder into the VM, where the user is someone else. | none |
| Mac kept awake | `bench.sh run` runs under `caffeinate -i -s`. A run that timed out while the host slept over a minute is an infra error and is retried on resume. | Idle sleep froze one run for 16 minutes while its time limit kept counting. Closing the lid still sleeps the Mac. | `MACARENA_ALLOW_SLEEP=1` |

Tried and removed: a VM shell tool (`vm_shell`). In the one check that used it (2 tasks x 3 runs), one run spent 36
shell calls poking at system plists and databases, created a stray file, deleted an existing login item and then
claimed success; it did not help the other task. Commands go through `run_osascript` with `do shell script`.

What these do not do: they make no task pass by themselves, and the 22 runs behind them are too few to show a
pass-rate change. Judge them on the next runs: turns per task, failure streaks, and whether final answers match
the graded result.

## Setup (once)

```bash
scripts/macarena/bench.sh setup --dry-run          # shows every step and download, changes nothing
scripts/macarena/bench.sh setup                    # UTM, the pinned MacArena checkout, its Python env, the macarena VM
scripts/macarena/bench.sh setup --vm osworld       # later, only if you want the OSWorld / macOSWorld tasks
```

`setup` asks before each download or install (`--yes` skips the questions; a non-interactive shell without it refuses):

**Then, once, grant the VM its screen permissions:** `scripts/macarena/bench.sh grant-perms`. MacArena starts everything
in the VM over SSH, and macOS blames those programs' permission requests on `/usr/libexec/sshd-keygen-wrapper`. Without
Accessibility and Screen Recording for it, MacArena's accessibility tree fails (HTTP 500) and screenshots show only the
wallpaper, **no app windows**, so a vision agent sees nothing. The grant is a macOS security setting that needs the
VM's admin password, so you make it once in the base image (the command boots it, opens the pane, tells you the clicks,
checks the result and shuts the VM down); every fresh copy then inherits it. It also changes the base image, so it is
kept in `macarena-backup` (a copy-on-write clone made before the change).

| Step | What | Size |
|---|---|---|
| UTM | `brew install --cask utm` | small |
| MacArena | `git clone` of github.com/MacPaw/MacArena, checked out at the commit pinned in `bench.sh` | small |
| Python env | `uv venv` (Python 3.11) in `../MacArena/.venv` plus MacArena's `requirements.txt` | a few GB (torch, transformers) |
| VM image | `hf buckets sync` from huggingface.co/buckets/macpaw-research/MacArena, then imported into UTM | `MacArena.utm` 67 GB, `osworld.utm` 54 GB |

It checks free disk space first. `setup` moves the downloaded image into UTM's own Documents folder (UTM opens a
`.utm` in place and MacArena reads it from there), so the image is not a spare copy: do not delete it. The download is resumable. `bench.sh up --setup --vm macarena` runs `setup`
first for whatever is missing. `bench.sh check` shows what is still missing. The VM names are `macarena` (MacArena's own
tasks) and `osworld` (OSWorld and macOSWorld tasks).

## Base image prerequisites (a person does these once per image)

macOS only grants these on a person's click, so they cannot be scripted. Do them in the **base** VM (boot it with
`utmctl start <name>`, or use the helper commands), shut it down so they are saved, then run
`scripts/macarena/bench.sh verify-image --vm <name>`. It boots a throwaway copy, checks every line below, prints
PASS/FAIL, and deletes the copy. Do not run it during a benchmark run.

| # | Needed for | How | Check |
|---|---|---|---|
| 1 | Accessibility and Screen Recording for `/usr/libexec/sshd-keygen-wrapper` (a vision agent sees windows; the accessibility tree works) | `bench.sh grant-perms` | verify-image: both `PASS` |
| 2 | Full Disk Access for the grader server's Python (`/opt/homebrew/Caskroom/miniconda/base/envs/server/bin/python3.12`; pick `python3.12`, not the `python3` link). Without it, checkers cannot read the content of files that apps such as Obsidian write in `~/Documents`, and those tasks fail whatever the agent does. | System Settings > Privacy & Security > Full Disk Access > `+`, Command-Shift-G, paste the path | verify-image: `Full Disk Access` PASS |
| 3 | Automation for each app a task scripts (`sshd-keygen-wrapper` controlling Contacts, Reminders, Calendar, Notes, System Events and others), plus the data grants for Contacts, Reminders and Calendar | `bench.sh grant-apps "App" ...` and click Allow on each dialog | verify-image: one `PASS` per app |
| 4 | Screen Recording for apps a task itself uses (for example OBS) | click Allow when the app asks, in the base VM | the app's task, or by hand |
| 5 | Apps installed. iWork (Keynote, Numbers, Pages) comes only from the App Store (needs an Apple ID); LibreOffice can be installed over SSH | by hand | verify-image: `not installed` lines |

verify-image asks `osascript` the same questions the task scripts ask, so a PASS means a task's setup and grader
can run. It cannot see inside an app: an app that opens a first-launch screen may still need one manual start-up.
The base image then differs from MacArena's stock image (items 1 to 5): report that with any score. Each image
(`macarena`, `osworld`) is separate, so repeat the list for each.

## Run

```bash
scripts/macarena/bench.sh up --vm macarena                                   # starts the bridge and Otto's backend
scripts/macarena/bench.sh run --sources macarena --noop                      # 1. do-nothing baseline, no model
scripts/macarena/bench.sh run --sources macarena --ids <id>                  # 2. one task, watch it
scripts/macarena/bench.sh run --sources macarena                             # 3. the 49 MacArena-native tasks
scripts/macarena/bench.sh report ../otto-macarena-runs/out/<run>
scripts/macarena/bench.sh stop
```

One bridge drives one VM, so a run takes one source family at a time (`--sources macarena` with `--vm macarena`;
`osworld` or `macosworld` with `--vm osworld`). `--sample N`, `--domain`, `--ids`, `--runs`, `--timeout`,
`--dry-run` are in `run.py --help`. Runs are resumable; infrastructure failures are retried on resume.
`--a11y` (on `up`) gives the agent the accessibility tree too; the default is screenshot only.
`--memory off|live|<snapshot.db>` (on `up`, native track) sets procedural memory: `off` (default) has no memory tool;
`live` lets each task recall the earlier tasks of the same run, passed or failed (run them in order with `--ids a,b`;
`scripts/procmem/memsmoke.sh` is the smoke test); a snapshot is frozen memory built from earlier runs
(`docs/procedural-memory-plan.md`).
Pick the model with `MACARENA_PROVIDER` / `MACARENA_MODEL` / `MACARENA_API_KEY`. Track A needs a vision model.

Output is in `../otto-macarena-runs/out/<run>/`: `results.jsonl`, `summary.txt` / `summary.json`,
`tasks/<source>/<domain>/<id>/run_N/` (MacArena's own layout: `traj.jsonl`, `native_calls.jsonl`; the per-step screenshots and `recording.mp4` are written only with `MACARENA_RECORD=1`) and
`trajectories/` (Otto's events and messages).

## What the numbers mean

| | MacArena paper | This harness | Effect on comparison |
|---|---|---|---|
| Step cap | 15 | none by default (`MACARENA_MAX_STEPS=15` restores 15) | The paper's 15 only with `MACARENA_MAX_STEPS=15` |
| What counts as a step | one `predict()` call, which may return several actions | **every action** | Stricter: Otto gets fewer actions than a baseline that batches. Scores here cannot be inflated by the cap. |
| Free observations | a screenshot each step | `vm_screenshot` is free, capped at 45 per task | Roughly the same |
| `done` / `fail` | an action in a step | an action, uses a step | Same |
| Runs per task | 2, task score = **best of the two** | 2; reports **best-of, mean-of and run-1** | Compare the paper only with best-of-runs. Mean-of-runs is the per-attempt number. |
| VM between runs | run 2 is **not** reset (only apps are quit) | **fresh VM every run** (`--paper-protocol` copies the paper) | Cleaner than the paper; run 2 cannot reuse run 1's files |
| `sleep_after_execution` | 3.0 s | 3.0 s | Same |
| Observation | screenshot, optional accessibility tree | screenshot (tree with `--a11y`) | Same options |

Not comparable with the paper: any run with a different step cap, observation type or number of runs, and any run
that gives the agent tools beyond the `vm_*` set (a future "Otto-native" track; not built).

`report.py` excludes infrastructure failures (VM would not reset, bridge down) from every rate and says so; an
infrastructure error is not an agent failure. "Solved" means score 1.0; some tasks give partial credit.

Licence: the 151 macOSWorld tasks are **CC BY-NC 4.0, non-commercial**. OSWorld tasks are Apache 2.0, MacArena's own
code and tasks are MIT. Check this before running or publishing results on the macOSWorld tasks.

## Status

**Verified** (`tests/test_macarena.py`, 54 tests, no VM or model needed; mutation-checked: 10 deliberate breakages
of the cap, validation, auth, guard, infra accounting and typing were each caught):
action validation and mapping, step / observation / invalid-action caps, actions refused after the budget is spent
or after done / fail, token separation, trajectory layout, the report's three scoring protocols and infrastructure
accounting, task discovery, the tool guard, MCP registration, and the MCP server over real stdio (which caught a
real bug: image results failed to serialise).

**Run end to end with a stand-in VM** (`bridge.py --fake-env`: blank screen, always scores 0): the real Otto
backend with the local Qwen3.6 model drove one task through `bench.sh run`. It used the `vm_*` tools, hit the
15-step cap exactly, every action was logged, and the result was recorded. That checks the plumbing, not the
agent's ability. The same run showed the model trying `execute`, `write_file`, `ask_user` and others; all refused.

**Not verified. These need a real VM:**
- **`setup` itself.** Its `--dry-run`, the refusal without `--yes`, and the dependency resolution (`uv pip install
  --dry-run` on Python 3.11) were checked; the real installs, the 67 GB download and the UTM import were not.
  The VM is called `MacArena` inside UTM but the MacArena README says to pass `macarena`; whether `utmctl` matches
  the name case-insensitively is unchecked.
- That MacArena's `DesktopEnv` starts under UTM on this machine, and that tasks reset and evaluate.
- **Coordinates.** The bridge scales screenshot pixels to pyautogui coordinates using `get_vm_screen_size()`. If the
  VM screenshot is larger than its logical screen (Retina), this is what keeps clicks on target, and it has never
  been checked on a real VM. Use `--coord-scale SX SY` on the bridge to override. **Click a known button first.**
- `pyautogui` key names and `typewrite` on the guest, and whether typing is reliable at the default speed.
- Whether `vm_scroll`'s sign (positive = up) matches what the guest does.
- Whether a model gets useful results from these tools: the prompt and the 15-step budget are untuned.
- The `--a11y` path against a real accessibility tree.

**Otto-native track.** Verified by `tests/test_macarena_native.py` (22 tests; 16 of 17 deliberate breakages of the
guard, step costs, caps, routing and guest server caught, the 17th was an equivalent change), by a run with a stand-in
VM (the real backend and model used Otto's tools, every call reached the stand-in, 15 steps, `execute` refused), and
on the real VM: the toolkit deploys and runs with Accessibility permission, `get_screen_controls` returns Otto's
indexed tree and `launch_app` opens an app. **Screen permissions (fixed, verified once).** The VM's SSH-launched programs had neither Accessibility nor Screen
Recording, so MacArena's accessibility tree failed and screenshots showed no app windows. After `bench.sh grant-perms`
a fresh copy reports both granted on the first attempt, Otto's `read_screen` OCR returns text, and MacArena's
`/accessibility` returns 200. Verified on one fresh copy only; the earlier failures were intermittent (two of four
boots worked before the grant), so watch for a recurrence.

**Not built:** Otto's AppleScript tools in the VM, and a fixed wall-clock timeout tuned to a real VM (the 600 s
default is a guess). Two VMs at once: see below.

### Two VMs at once

macOS runs at most two macOS VMs at a time, so `bench.sh` has two slots. Each slot is a bridge (with its own VM
copy) and a backend:

| | Slot 1 (default) | Slot 2 (`MACARENA_SLOT=2`) |
|---|---|---|
| Bridge / backend ports | 18095 / 18093 | 18195 / 18193 |
| Logs, pid file, desktop lock, benchmark home, VM bundle | `otto-macarena-runs/` | `otto-macarena-runs/slot2/` |

Tokens, MacArena's environment, the VM images and the built VM packages are shared. Start both slots before
either run: `bench.sh run` records how many slots are up (`parallel` in each result; override with
`MACARENA_PARALLEL`). Then give each slot the same selection with its own `--shard` and `--out`:

```bash
MACARENA_SLOT=1 scripts/macarena/bench.sh up --vm osworld --native
MACARENA_SLOT=2 scripts/macarena/bench.sh up --vm osworld --native
MACARENA_SLOT=1 scripts/macarena/bench.sh run --sources osworld --shard 1/2 --out ../otto-macarena-runs/out/X-1
MACARENA_SLOT=2 scripts/macarena/bench.sh run --sources osworld --shard 2/2 --out ../otto-macarena-runs/out/X-2
scripts/macarena/bench.sh report ../otto-macarena-runs/out/X-1 ../otto-macarena-runs/out/X-2
```

- **Shards:** `--shard K/N` takes every N-th task of the sorted selection, so each half gets a share of every
  domain, and the halves together are the selection exactly once.
- **Reporting:** `report` with several folders scores them together. It refuses a task's run that appears in two
  folders, and it prints the summary without writing into either folder.
- **Stopping:** `bench.sh stop` stops one slot (`MACARENA_SLOT=2 ... stop` for the second). `bench.sh slots` lists
  which slots are up.
- **Speed:** both agents share one model server. Reading long prompts barely speeds up with two requests at once,
  so expect less than twice the throughput. Each agent is also slower, so more runs may reach `--timeout`.
- **Pooling:** `report.py` will not pool results run with different numbers of VMs at once, so one-VM and two-VM
  runs stay apart.

**Checks to do before quoting any number:** the do-nothing run (`--noop`) to list tasks that pass untouched; one
task watched by eye; the same 49 tasks run twice to see how much the score moves between runs.

### Permission check before a run

Tasks are set up and graded with `osascript` ("tell application Contacts ..."). Until `sshd-keygen-wrapper` has
been allowed to talk to Contacts, Reminders, Calendar and Notes, the first query hangs on a dialog nobody clicks.
MacArena starts setup commands without waiting for them, so this fails silently: the contact the task should
edit is never created, and the grader says False whatever the agent does. `run` therefore probes every app the
chosen tasks script (about 20 s per app that is waiting on a dialog), skips the tasks that cannot work, lists
them in `<out>/excluded.json`, and writes the probe results to `<out>/permission_check.json`.

- `scripts/macarena/bench.sh grant-apps` answers the dialogs once in the base VM image (a person clicks Allow).
- `run.py --preflight-only` prints the check and stops; `--no-preflight` skips it.
- On the current base image the check finds Contacts, Reminders, Calendar and Notes waiting, which skips 15 of
  the 44 usable tasks until `grant-apps` is run.

### Step limit

The bench runs with **no step limit** by default (`MACARENA_MAX_STEPS=0`). A run then ends on `vm_done` or
`vm_fail`, on the runner's `--timeout`, or when the read cap (600 reads) is hit. Every acting call still counts
and is logged, so you can see the steps a run used. Scores with no limit are **not comparable with the paper's
15-step results**: the report says "step limit: none". To match the paper, start with
`MACARENA_MAX_STEPS=15 scripts/macarena/bench.sh up ...`; the runner refuses a `--max-steps` that differs from the
bridge's, and the report refuses to mix results taken at different limits.

**What one step is** (`MACARENA_STEP_UNIT`, native track only; start the bridge and the run with the same value):

- `actions` (default): every acting call is a step, and `batch_actions` costs one per action. The stricter rule.
- `turns`: one step per agent turn, meaning one model response that calls any tool, however many calls it makes.
  This is the closest match to the paper's cycle (macOSWorld allows 15 screenshots or 30 dialog turns, MacArena 15
  steps, and one `predict()` call is one step even when it returns several actions). A turn with only reads counts.
  `vm_done` / `vm_fail` in a turn that already used a step is free; in a new turn it uses one, and past the budget it
  is refused (macOSWorld's agent can only say DONE / FAIL inside its 15 calls). The backend sends each call's turn id (the issuing model response, from
  `MacarenaGuardMiddleware`); a call with no id is counted as an action. The new turn after the last allowed one is
  refused with "no turns left"; the calls of an allowed turn still run.

**macOSWorld's "15 screenshots or 30 dialog turns"** is one limit in its code (`utils/run_task.py`): a loop of 15
model calls, each running every tool call in that response. OpenAI-style agents get a screenshot each step (15
screenshots); the Anthropic agent takes screenshots as an action, so its 15 steps show as about 30 messages. `turns`
with `MACARENA_MAX_STEPS=15` is that loop. The backend also reports each model call and tool call to the bridge
(`/agent/tick`, benchmark agent only), so every result row records `model_calls`, `tool_messages` and
`dialog_messages`, and the report prints their maximum and mean per run.

Either way the agent still has text controls and AppleScript where the paper's agents have pixels, so neither rule
makes the number comparable with the paper's baselines. The report prints the unit and refuses to mix units.

### Skipping tasks that need apps the VM does not have

A few MacArena tasks name apps that are not in the public VM image (Keynote, Numbers, Pages, LibreOffice: 5 of
49 tasks). They cannot pass, so skip them rather than count them as failures:

```bash
scripts/macarena/bench.sh run --exclude-apps Keynote,Numbers,Pages,LibreOffice --sample 10 ...
# or for every run:  export MACARENA_EXCLUDE_APPS=Keynote,Numbers,Pages,LibreOffice
```

`--exclude-apps` matches each task's `related_apps` (case-insensitive); `--exclude-ids` skips single tasks. The
exclusion is applied before `--sample`, so a sample of 10 is still 10 tasks, and the skipped ones are listed in
`<out>/excluded.json`. Say that you excluded them when you report a score.

### Tasks whose checker is wrong (exclude by id, and say so)

`known_bad.json` lists them with the reason and the evidence. `report.py` prints the score as measured, then an
adjusted line without failed setups and without the `confirmed` entries; `suspected` ones are only listed. Quote
the adjusted number only beside the measured one, with what it left out. Move an entry to `confirmed` only after
checking it on a VM: every entry raises the adjusted score.

| Task id | Status | Why |
|---|---|---|
| `77be0051-e4d5-139b-04d4-a760344aa282` (macOSWorld, Automator Quick Action "Hello") | confirmed | The instruction asks for `Hello.workflow`; the grader runs `automator` on `~/Library/Services/Untitled.workflow`. |
| `8887a199-469d-c544-09d4-bb32d5b285b4` (macOSWorld, AppleScript dialog) | suspected | The grader greps the dialog's AXChildren for one exact string; the agent showed the requested dialog twice and scored 0. Not checked on a VM. |

| Task id | Why |
|---|---|
| `5649b5ef-5592-4c35-bed0-631809292b61` (install and enable the Excalidraw plugin) | The checker tests `~/Documents/ResearchVault/obsidian-excalidraw-plugin`. A real install through Obsidian lands in `~/Documents/ResearchVault/.obsidian/plugins/obsidian-excalidraw-plugin` (confirmed on a VM, 2026-11-17), so the task cannot pass. |

```bash
export MACARENA_EXCLUDE_IDS=5649b5ef-5592-4c35-bed0-631809292b61
```

Another candidate, not excluded yet: `d2b53cf2-aa18-44aa-a8d5-84194a7791d0` (Jupyter notebook). The instruction says
Desktop but the checker reads `~/Documents` (`suspected` in `known_bad.json`).

**Comparability.** Waiting for and checking the setup departs from MacArena's runner, which only starts it. It
makes the score a better measure of the agent here; it also means a number taken with `setup_mode: sync` is not
the same measurement as one taken MacArena's way (`report.py` will not pool the two). Fixing the setup can lower
some scores: with a failed setup, a grader that only checks a file is *gone* (`4c97119f`, delete the zip files)
passes whatever the agent does.
