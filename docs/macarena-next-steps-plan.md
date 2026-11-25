# MacArena: next steps (plan and honest status)

Scope: the Otto-native track (`macarena-desktop-agent`), run with `scripts/macarena/bench.sh`. Benchmark code stays under
`scripts/macarena/` and `backend/macarena_*`; the prompt change lives in the benchmark-only agent definition.

## Where things stand

- Latest run, `native-sample10h` (10 sampled tasks, 1 run each, no step limit, tool-call cap 100):
  **4 of 7 runnable tasks passed**. 3 tasks were skipped by the permission check (Contacts / Reminders not granted).
  Not comparable with `sample10g` (1 of 10): different prompt, different skip set, n = 7.
- No infrastructure errors or timeouts in 10h. Steps per task 6 to 64; the 100-call cap was never reached.
- Not committed. The new prompt (v9) has not been run against the benchmark yet.

## What the 3 failures were

| Task | Cause | Whose |
|---|---|---|
| Jupyter notebook on Desktop | Task says Desktop, checker reads `~/Documents`. Agent also could not launch Anaconda-Navigator and used Anaconda's Python. | Task bug (checker), disclosed by the agent |
| Calculator as a login item | Could not find the "+" control, wrote a LaunchAgent plist, reported it as a login item. | Agent (overclaimed) |
| Obsidian vault + note | Created the vault, renamed the note; its own `cat` verification failed with `Operation not permitted` and it still declared the content correct. Real file contents unknown. | Agent (ignored a failed check); a permission cause is possible but unproven |

## Done in this segment

- Prompt v9 adds two generic rules: *a failed check is not a pass* and *use the thing the task names*. No task nouns
  (a test enforces this). 112 tests pass.

## Plan

1. **Re-run the permission grant** (needs you). The `grant apps` terminal tab is stale: the base VM it waited on is
   stopped. Run `scripts/macarena/bench.sh grant-apps` again from a fresh terminal, click Allow for Contacts, Reminders,
   Calendar and Notes in the UTM window, press Return. I do not change macOS security settings or click those dialogs.
2. **Documents question: resolved (see Result).** Original text: resolve the Documents question before building anything on it. On a fresh VM copy, from the benchmark server, SSH
   and `osascript`, compare `stat`/`[ -f ]`, `ls`, `cat` and `grep` on `~/Documents`, `~/Desktop`, `~/Downloads`.
   - Evidence so far conflicts. A probe on one VM copy got `Operation not permitted` on all three folders from the
     server. Yet three tasks whose graders read `~/Documents` passed in 10h, one with `grep` on file content.
   - 27 of 49 task graders mention these folders, so a wrong "blocked" rule would drop over half the tasks.
   - Decision rule: add a preflight folder probe and a `grant-apps` Full Disk Access step only if the block is
     reproducible on the server path the graders use. If it is only the agent's own channel, leave tasks in and
     tell the agent (generic wording) to verify through the app, Finder or Terminal.
3. **Re-run the 10-task sample with prompt v9** using a new `--out` (for example `native-sample10i`), the same
   `--exclude-apps Keynote,Numbers,Pages,LibreOffice`, 1 run. After step 1 the 3 skipped tasks should be runnable;
   the preflight will confirm.
4. **Review task-definition bugs before excluding anything.** The Jupyter task is a checker bug (Desktop vs Documents).
   Exclude by id, list it in `excluded.json`, and report the exclusion. Do not drop a task because the agent failed it.
5. **Report honestly.** State: step limit (none) and the 100 tool-call cap, runs per task, the skip list with reasons,
   the 15-step-equivalent score for comparison with the paper protocol, and that several checkers are weak and can be
   satisfied by a stub, so pass rate is an upper bound on real success.
6. **Housekeeping (optional, your call).** The 22 GB `checkpoints.sqlite` under `otto-macarena-runs/home/` (not deleted).
   Commit the work in logical pieces once you approve.

## How much to trust this

- Each task ran once, so differences of a few tasks are noise. Claiming "the prompt helped" needs more tasks or runs.
- The prompt fix makes the final answer more truthful; it does not make a task pass. Scores can stay flat or drop while
  the report gets more honest.
- Prompt wording is the only agent-side lever in this plan. A task the agent cannot do through the GUI (the login-item
  case) may stay failed, and that is a fair result.
- The Documents hypothesis is unproven in both directions. Step 2 exists to settle it; until then nothing is built on it.
- `open_app` fallback and preflight code were verified by tests and on a VM, but a full run after the latest changes
  has not happened.

## Result of step 2 (tested 2026-11-17 on a fresh VM copy, base image state)

- Listing the top-level `~/Documents`, `~/Desktop` or `~/Downloads` fails with `Operation not permitted` from the
  benchmark server, SSH and `osascript` alike.
- Creating, reading (`cat`, `grep`), `[ -s ]` and deleting a file inside `~/Documents`, and listing a subfolder, all
  work from every channel. This matches the passing graders, which name files.
- So my earlier worry (graders blocked, 27 of 49 tasks affected) was wrong; no folder preflight or exclusion was added.
- Still unexplained: the agent's `cat ~/Documents/ResearchVault/Ideas.md` got `Operation not permitted` on a file the
  agent created through Obsidian. A file-specific protection on an app-created file is a guess; it is not tested.
- **Resolved 2026-11-17 (grader test):** on a fresh copy with MacArena's own server started as `install.sh` does,
  a vault made through Obsidian's GUI passed the existence checks (`[ -d .obsidian ]`, `[ -s Welcome.md ]`), but
  `grep`, `head` and `jq` on files Obsidian wrote all failed with `Operation not permitted` through
  `/run_bash_script`. A file the server wrote itself in the same folder was readable. No permission dialog appeared,
  so MacArena's auto-Allow helper would not change this. So any checker that reads the content of a file Obsidian
  writes cannot pass in this image: `ace64a35` (Ideas.md), `dc6870aa` (Links.md), `57d79cb5` (appearance.json).
  `5649b5ef` only checks a folder exists and is unaffected. Same likely applies to other apps that write into
  `~/Documents`; untested.
- **Fixed 2026-11-17:** Full Disk Access was granted (by the user, in System Settings) to
  `/opt/homebrew/Caskroom/miniconda/base/envs/server/bin/python3.12`, the grader server's Python, in the base VM.
  On a fresh copy, files Obsidian wrote in `~/Documents` were then readable by `grep`/`jq` through the server and by
  SSH and osascript; listing top-level `~/Documents` worked too. Not verified: the full `ResearchVault` task end to
  end (my GUI clicks missed, Obsidian made its default vault instead), and the other apps. The three tasks above
  therefore stay in. The base image now differs from MacArena's stock image: report it.
- Agent-side change: prompt v10 tells the agent that listing those top-level folders is refused and to name the file
  or use Finder instead. Not yet exercised in a benchmark run.

## Result of the Obsidian re-run (`native-obsidian-fda`, 4 tasks, 1 run each, prompt v13, after the Full Disk Access fix)

- Score 2 of 4 (theme: pass; Links.md: pass; Ideas.md: fail; Excalidraw plugin: timeout). One run each: a rough signal only.
- Ideas.md: the agent made the folder and the note with a shell script after the app's Create step, then declared it
  verified. The checker also wants the app's `.obsidian` folder (my guess: missing; not confirmed, the VM is gone).
- Excalidraw: installed and enabled by step 10 (the plugin's button read "Disable"), then ~50 more steps of
  re-checking and reopening dialogs until the 10-minute limit. A timed-out run scores 0 even if the work was done.
- **Excalidraw checker bug, confirmed on a VM (2026-11-17):** after a real install through Obsidian's window the plugin is
  in `~/Documents/ResearchVault/.obsidian/plugins/obsidian-excalidraw-plugin`. Task `5649b5ef` checks
  `~/Documents/ResearchVault/obsidian-excalidraw-plugin` (no `.obsidian/plugins`), which a real install never makes.
  Exclude by ID and report it as a task bug; the v14 run (agent installed and enabled it, scored 0) is consistent with this.
- Prompt v14 adds: use a shell only to look, never to make what the app should make; and stop (`vm_done`) once the
  end state is confirmed. Not yet run.
