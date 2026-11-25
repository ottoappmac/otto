# OSWorld (macOS, Otto-native track, 100 turns): failure review and fix plan

Run: `otto-macarena-runs/out/20261122125242-osworld-full-native-100step`. It was reviewed on 2026-11-23 while still
running, with 155 of 221 tasks scored. Task `7ff48d5b` was in progress and is excluded.

Per-task diagnoses are in `docs/reports/osworld-native-100step-findings.jsonl`, one JSON object per failed task with
tags, what happened, why it scored 0, evidence and a fix idea. They were made by reading every failed run's tool
calls against the task's evaluator in `MacArena/evaluation_examples/osworld/`. For calc and writer the reviewers
also diffed the saved files against the cached gold files.

## Where it stands

**Mean score 0.35 (54 of 155).** Passing runs took a median of 14 turns and 66 s; failing runs took a median of
54 turns and 330 s. Failures used 15.4 of the 19.7 machine-hours.

| Domain | Tasks | Score | Passed |
|---|---|---|---|
| chrome | 32 | 0.44 | 14 |
| gimp | 24 | 0.38 | 9 |
| libreoffice_calc | 45 | **0.18** | 8 |
| libreoffice_writer | 23 | 0.52 | 12 |
| multi_apps | 31 | 0.36 | 11 |

How the 101 failures ended:

| Ending | Count | Note |
|---|---|---|
| `vm_done`, but the result was wrong | 44 | the largest group; usually no check before finishing |
| used all 100 turns | 24 | 13 of them in calc |
| ended with a text reply, no `vm_done` / `vm_fail` | 19 | 4 were the repetition guard ending the turn (see H1) |
| `vm_fail` on a task that could be done | 7 | |
| task can't be done, but the agent made a stand-in | 7 | 5 `vm_done`, 1 budget, 1 chat answer |

Finishing without `vm_done` does not by itself cost the score: the evaluator runs anyway, and 5 such runs passed.
It matters for the tasks that can't be done, and when the agent stopped early.

## Implementation status (2026-11-23, uncommitted)

Everything in the plan below is implemented, with tests in `tests/test_osworld_review_fixes.py` (55) and
`tests/test_repetition_guard.py`. None of it has been run against the VM yet. The next step is to re-run the failed
calc and gimp ids.

| Item | Where |
|---|---|
| T1 Name Box: the group redirects to its field, buttons and bare groups refuse typing, and the Name Box is read back after Return | `macos_tools.py` `field_group_redirect`, `type_into_control`, `name_box_readback_note` |
| T2 replace-typing: a retry when the old text survives, `type_text(replace=True)`, and a warning when the old text is left after the typed text | `macos_tools.py` |
| T3 hotkey: names normalised (cmd, alt, enter), unknown keys refused with nothing sent | `parse_hotkey` |
| T4 fallback clicks: refused outside the app's windows; a note when the frontmost app changes after a click | `press_control`, `front_change_note` |
| T5 batch: quoted values stay text, values take the schema's types, and pointer tools are allowed | `_parse_batch_actions`, `coerce_args` |
| T6 OCR: clicks the matched word, not the line centre; off-screen matches dropped | `substring_point`, `find_text_on_screen` |
| T7 Chrome bookmark AppleScript refused before it runs | `_helpers.refused_script_reason` |
| H1 repetition guard retries once before it ends the turn | `src/middleware/repetition_guard.py` |
| H2 finish nudge (recorded harness setting) | `scripts/macarena/run.py` |
| H3 identical-call notice after 3 (recorded harness setting) | `scripts/macarena/bridge.py` |
| H4 old controls lists cut from the replayed history; image pruning extended to the desktop agents | `src/middleware/old_controls_pruning.py` |
| H5 Chrome's update window closed at reset (recorded harness setting); LibreOffice's update balloon is no longer read as the front window | `bridge.py`, `pick_app_windows` |
| H6 dates: investigated, not a harness bug (see H6); a prompt line on "next Monday" | `agent_library.py` |
| Group 3 tools: `write_cells`, `drag`, `set_text_at`, `select_text`, pop-up options in `set_popup` / `fill_dialog`, `fill_dialog(path=)` | `macos_structured_tools.py`, `macos_tools.py`, `macos_ax_struct.py` |
| Group 4 notes: Chrome, GIMP, Calc, Writer, General | `macos_app_cards.py`, `agent_library.py` |
| Group 5: `run_osascript(shell_command=...)` does the AppleScript quoting | `macos_osascript/server.py` |

The new harness settings (repeat notice, finish nudge, update nags closed) are recorded per run. `report.py`
will not pool new runs with this one.

## Root-cause groups, ranked by tasks they would likely recover

The counts are tasks where the group was the primary cause or a decisive one. A task can be in more than one group.

### 1. Tool bugs (fix in `src/tools/navigation/computer/`), about 30 tasks

**T1. Typing into the Name Box wrapper types into the active cell and reports success. About 19 calc tasks; the
biggest single fix.**
`get_screen_controls` lists `[1]G'Name Box'` (the group) and `[2]TA'Name Box'` (the field). The agent picks #1.
`type_into_control(index=1, "D2:D29", submit=True)` then returns `Typed 'D2:D29' into control #1 and pressed
Return`, but the text went into the current cell. The saved files hold stray cells such as `G13='C2'`,
`B27='A1:C1'`, `J32='A2:J14'` and `A1='A1:C11'`, header cells are overwritten, and the following select / copy /
freeze acts on one cell. `name_box_refusal` (`macos_tools.py:939`) only refuses non-reference text, and only by
label. A reference typed into the group gets through.
- Fix: when the target is the `Name Box` group (or any non-text role whose child is a text field), redirect to
  the child text field and say so in the result. After a Name Box submit, read the active-cell reference back and
  report it ("now at D2:D29"), or report that the selection did not change.
- Also: the LibreOffice app card says "Name Box (index 1)". Remove the index.
- Tasks: 4188d3a4 4172ea6e 1d17d234 347ef137 12382c62 1954cced 21df9241 0326d92d 01b269ae 4e6fcf72 51719eea
  eb03d19a 7e429b8d 7efeb4b1 a01fbce3 a9f325aa 7a4e4bc8 6e99a1ad.

**T2. Replace-typing does not replace. About 7 tasks.**
- The Calc Format Code combo box with `append=false` produced `0;0;;@General` (6054afcb, a01fbce3).
- The GTK Export filename produced `The_Lost_River_Of_Dreexport.jpgams.jpg` (77b8ab4d; the agent then `rm`'d the
  graded original).
- The Find & Replace field appended (236833a3).
- A Writer TextArea got a prefix, `aaaadGrammar test 2:` (1f18aa87).
- Newlines became `.` in Replace fields (88fe4b2d).
- Fix: for `append=false`, select all with Cmd+A plus AXSelectedTextRange, falling back to triple-click, then
  type. Read the value back and say clearly when it isn't the typed text (already done for some roles; extend it
  to combo boxes and GTK entries). Also accept `\n` in Find & Replace when regex is on.

**T3. `hotkey` accepts junk key strings.**
`'<parameter=command>\nz'` typed literal `sszzz` into cells (4f07fbe9). `'{return}'` was accepted and did
nothing (7a4e4bc8).
- Fix: check every key against the allowed set and refuse with the list. Never fall back to typing characters.

**T4. AX fallback clicks land in another app.**
`press_control` returns "AXPress err=-25204; clicked at (1082, 210) instead", and the click hits Finder
(2ad9387a, 7a5a7856) or the wrong sidebar item (5bc63fb9). `click_text` on LibreOffice's "update available /
Click the icon…" notice clicked at y≈1194, which is in the Dock area, and the next actions went to TV and
Messages (72b810ef 936321ce 0810415c 88fe4b2d). A Chrome calendar CheckBox reported a frame at x=12
(1704f00f).
- Fix: before any coordinate fallback, check that the point is inside one of the target app's windows. If it
  isn't, refuse with "the control's frame is outside the app's window".
- Fix: drop OCR hits inside the Dock or menu-bar band unless that app is the target.
- Fix: after a fallback click, say whether the frontmost app changed.

**T5. `batch_actions` argument parsing.**
Bare numbers fail with "Input should be a valid string input_value=2015" (04d9aeaf, 00fa164e), and
`click_at(x=480, y=620)` returns "No valid steps parsed" (e2dd0213).
- Fix: turn scalars into strings for text arguments, and accept keyword arguments in the step parser.

**T6. `find_text_on_screen` returns the centre of the OCR line, not the matched word.**
In 0b17a146 the agent subscripted the wrong character. In 734d6579 it returned an off-screen y of 1181.
- Fix: return the word or substring box, using Vision's per-character ranges.
- Fix: clip results to the screen.

**T7. `run_osascript`: Chrome bookmark AppleScript crashes Chrome.**
`make new bookmark folder` or `make new bookmark item` returned "Connection is invalid (-609)", and Chrome
quit (2ad9387a, 7a5a7856). It came back without the harness `--user-data-dir`, so the evaluator read the
untouched profile.
- Fix: refuse those AppleScript commands and point the agent to the Bookmark Manager UI
  (Cmd+Shift+O / Cmd+D).

### 2. Harness and agent-loop fixes (`scripts/macarena/`, `src/middleware/`), 5 to 10 tasks plus cleaner data

**H1. The repetition guard ends the turn, and in a run with no user nothing resumes it.**
`RepetitionGuardMiddleware` (`src/middleware/repetition_guard.py`) replaces a degenerate reply with a recovery
note that has no tool calls. The ReAct loop then ends the turn. 1954cced, 30e3e107, aa3a8974 and f4aec372 died
mid-task with 16 to 45 turns left.
- Fix, either in the guard or the harness: when the agent has no user (benchmark or unattended mode), feed the
  recovery note back as a message and call the model again, up to N times, instead of ending the turn.

**H2. Re-prompt once when a turn ends with no `vm_done` / `vm_fail` and turns are left.**
19 runs stopped this way. The 6 Chrome Q&A runs especially would get a chance to put the answer page in the
active tab, and 58d3eeeb (can't be done) would get a chance to call `vm_fail`. Change `run.py drive()`: on
`done` with `finished is None`, send one fixed follow-up that says the run is not finished: "call vm_done if the
screen now shows the finished task, or vm_fail".
- Record it as a harness setting (like `fail_notice_after`) so `report.py` keeps runs from before and after it
  apart.

**H3. Identical-action streaks.**
734d6579 made about 25 identical `click_at(42,152)` calls and about 15 `/` presses. The failure notice did not
fire because those calls "succeed".
- Fix: add an identical-call streak notice to `bridge._streak_notice`. Use the same tool and arguments with no
  change in the controls list, and fire after 3 calls.

**H4. Context growth.**
2a729ded died at turn 98 on the oMLX prefill memory guard. `src/middleware/old_image_pruning.py` is uncommitted.
- Fix: also prune old "Controls now" lists, which can run past 8000 characters per action, from tool results
  older than the last few turns.

**H5. VM base image noise.**
Chrome's "Can't update Chrome" dialog and LibreOffice's "update available" notice appear at task start and cost
turns or misdirect clicks (see T4). Turn off update checks in the base image, or dismiss them in the reset
steps.

**H6. Dates relative to today.**
"Next Monday" evaluators use the host clock. The run started on a Sunday (Nov 22), and 82bc8d6a and 1704f00f may
have been graded against a different date. Check the evaluator's timezone and the VM clock. Investigate only; it
may not be our bug.

*Investigated (2026-11-23): dates were not the cause.*
- `get_rule_relativeTime` (MacArena `desktop_env/evaluators/getters/misc.py`) takes "now" on the evaluator host. It
  uses the task's `timezone` if set, otherwise the IP-geolocated timezone, and falls back to UTC. Neither task sets
  a timezone.
- "next Monday" is `(6 - weekday) + 1` days ahead, so on a Sunday it means tomorrow.
- Both tasks were graded on Sunday Nov 22: 82bc8d6a at 14:45 AEDT and 1704f00f at 13:28 AEDT. That is still
  Sunday in UTC (03:45 and 02:28), so the fallback gives the same answer. Expected: Monday 2026-11-23, then
  Friday 2026-11-27.
- 82bc8d6a: the agent's `departing=2026-11-23` was the right date. The zero comes from another key, or from the
  URL the grader read out of the accessibility tree (`get_active_url_from_accessTree` plus `goto_prefix`), not
  from the date.
- 1704f00f: the agent picked Nov 30 to Dec 4 (a week late) and "Zurich Airport". Both are agent errors.
- The prompt now says that on a Sunday, "next Monday" is tomorrow.

### 3. Missing tools, about 15 tasks (mostly budget-outs)

| Tool | Why | Tasks |
|---|---|---|
| `write_cells(anchor, rows)` for Calc: Name Box, then type the block with Tab/Return, then read it back | Entering a 2-D block one cell at a time wastes the budget, and Tab through `type_text` merges cells | 04d9aeaf 26a8440e 0326d92d 8b1ce5f2 3e3fc409 |
| `drag(from, to)` with control or OCR targets | Pivot Table Layout fields, GIMP layer moves | 1954cced 535364ea e2dd0213 f4aec372 |
| Pop-up button and combo-box options in `read_dialog` / `fill_dialog` (open the menu, read its items, pick one) | LibreOffice Page Style "Scaling mode", Format Cells Language, Preferences tree | aa3a8974 a01fbce3 30e3e107 |
| `select_text(paragraph=n \| find="…")` in Writer, reporting the selected word count | The agent selects from mid-paragraph, and shift+`end` selects to the end of the document | 0810415c 0b17a146 72b810ef 936321ce 88fe4b2d |
| `fill_dialog(path=…)` for the macOS open/save sheet using Cmd+Shift+G | Sidebar clicks hit the wrong item, and the Where pop-up has no Desktop | 415ef462 5990457f 5bc63fb9 185f29bd |
| GTK spin-scale / slider setter for GIMP (click the field, Cmd+A, type the value, Return) | GIMP sliders, the colour swatch, tool options | 7767eef2 734d6579 2a729ded |

### 4. Prompt and `app_guide` notes, about 25 tasks (cheap; do first)

**Chrome** (6 answered in chat, 2 wrong refusals):
- The graded state is the active tab. Finish with the page that shows the answer open in the front tab, on the
  site the task starts on (or the site it names), and call `vm_done`. Answering in chat is not graded.
  The evaluator is `is_expected_active_tab`, for example `babycenter.com/baby-names/details/carl-853` and
  `nfl.com/scores/2019/post4`.
- Settings locations: Do Not Track is under `chrome://settings/cookies`, and "delete site data on close" is under
  `chrome://settings/content/siteData`. Check both on the VM's Chrome version first.
- Don't edit Chrome's profile files from the shell. The evaluator reads `~/user-data-dir` (44ee5668 deleted
  History rows in the wrong profile).

**GIMP** (3 overwrote the source, 4 missed can't-do tasks):
- Don't save, export or overwrite the image unless the task names an output file. Leave the edited image open
  and in front. The checker exports it itself with Cmd+Shift+E and compares the export with the original file.
  This is confirmed in `554785e9…json`.
- Remove "File > Overwrite" from the GIMP card's save advice for this case.
- Tasks that can't be done in GIMP: converting to SVG, opening RAW without a loader, batch processing across
  files, and describing an image. Call `vm_fail`; don't make a stand-in with `brew` or `cp`.

**Calc**:
- Keep the default sheet name (`Sheet2`) unless the task gives one. 5 checkers look sheets up by name: 035f41ba
  1273e544 12382c62 04d9aeaf 26a8440e.
- Before writing, read the header row and the data extent, and write under the matching header (abed40dc d681960f
  a9f325aa ecb0df7a). Include the corner totals (f9584479).
- Sort and Text to Columns must cover the whole range, and Text to Columns needs its destination set (3a7c8185,
  37608790).
- Note that `read_table` may start at column B (01b269ae).

**Writer**:
- Before applying formatting, check that the "Selected: N words" count matches the target text.
- Don't press Cmd+Z more than once without re-reading (0a0faba3 undid its own correct tab stop).

**General**:
- Before `vm_done`, re-read the exact thing the task asks for (cell, page, setting, file), not a nearby proxy.
- Don't satisfy a GUI task by hand-editing the document's file format (docx XML, sqlite, `cp`). These broke
  8472fece, 44ee5668 and dbbf4b99.

### 5. AppleScript and shell friction (cross-cutting)

828 `run_osascript` calls were made and 279 failed (about 34%): app-dictionary calls 48%, System Events UI
scripting 44%, `do shell script` 26%. The most common errors are quoting errors from nesting shell or Python
inside AppleScript strings ("A number can't go after this \"", "Expected \" but found unknown token") and
Python tracebacks.
- Fix: give `run_osascript` a `shell=` argument (or add a `run_shell` tool, routed through the same guest
  server) that takes a raw command and does the AppleScript quoting itself. This removes the whole quoting
  class.
- Keep the GUI-first rule in the prompt. Shell use correlated with failure: 31 of the 50 tasks that used shell
  failed.

### 6. Misread tasks and evaluator strictness, about 15 tasks: little to fix

The agent misread the task in 11 (for example, it made a Calendar event when `Course Timetable.xlsx` was meant,
and created a new document when the open docx held the text). The evaluator was arguably too strict in 5
(`ps` grep for "sh ", median-only `result.txt`, sheet-name conventions). Most are one-offs. The "use the file or
document that is already open" rule from group 4 covers the main pattern.

## Suggested order

1. **T1 Name Box redirect plus read-back**, and the **H1 guard continuation**. Small, confirmed, and worth the most.
2. **Group 4 notes** for Chrome, GIMP, Calc and General. Prompt and card text only.
3. **T2 to T5** (replace-typing, hotkey checks, fallback-click guard, batch parsing). Each needs unit tests in
   `tests/test_macos_*`.
4. **H2 re-prompt** and **H3 identical-call notice**. Change the harness settings version so results stay
   separate.
5. **`write_cells`**, **file-sheet `path=`**, **pop-up options in `fill_dialog`**, then `drag` / `select_text` /
   GTK slider.
6. **The `shell=` argument for `run_osascript`**.

Re-run the failed calc and gimp ids first, since they are the cheapest signal for T1 and the GIMP save note.
Then run the full set again.

**Rough upside.** If groups 1, 2 and 4 land and hold, about 25 to 35 of the 101 failures look recoverable, which
would move the score from 0.35 to roughly 0.50 to 0.57. That is an estimate from these diagnoses, not a
measurement.

## How much to trust this

- Each failure was diagnosed by one reviewer reading truncated tool calls plus the evaluator config. Calc, writer
  and gimp checks were confirmed against gold files or evaluator configs. Chrome 368d9ba4 and 82bc8d6a are
  unexplained: their final URLs look correct.
- These claims were checked directly:
  - T1: the call log for 4e6fcf72 and the guard code.
  - The GIMP export comparison: the evaluator config.
  - The Chrome active-tab grading: two evaluator configs.
  - H1: the guard code and the ends of 4 runs.
- T2 to T7 come from quoted tool results, and the cause has not been reproduced.
- This is one run per task, so single-task conclusions are noisy.

## Appendix: every failed task

| Task | Domain | Finished | Tags | Why it scored 0 |
|---|---|---|---|---|
| `59155008` | chrome | None | answer-in-chat-only, no-finish-call, misread-task | Evaluator checks the active tab URL equals https://www.babycenter.com/baby-names/details/carl-853 (the name detail page whose 'similar names' list is the answer). Final tab was a Google results page. |
| `47543840` | chrome | None | answer-in-chat-only, no-finish-call, tool-gap | Evaluator expects page state: url contains 'reservation#/vehicles', header locations/dates, and sort rank text == 'Number of Seats (High to Low)'. The page was still sorted by price. |
| `7f52cab9` | chrome | None | answer-in-chat-only, no-finish-call, tool-gap | Evaluator needs the active Google Shopping tab URL q='drip coffee maker' and the selected filter chips 'Black', '$25 - $60', 'On sale' (and no others). The tab ended on Amazon. |
| `f5d96daf` | chrome | None | answer-in-chat-only, no-finish-call, misread-task | Evaluator expects the active tab URL to have modelList = iphone-15-pro-max, iphone-14-pro-max, iphone-13-pro-max (apple.com/iphone/compare). |
| `82bc8d6a` | chrome | done | evaluator-strict, wrong-result-claimed-done | Evaluator: departing must equal 'next Monday' computed by the evaluator (timezone from IP geolocation). The state matches what we computed, so the miss is probably date ambiguity: the run was on Sunday Nov 22 and the … |
| `030eeff7` | chrome | fail | false-infeasible, app-knowledge-gap | Evaluator reads Preferences (enable_do_not_track) expecting 'true'; nothing was toggled. |
| `99146c54` | chrome | fail | false-infeasible, app-knowledge-gap | Evaluator reads Preferences profile.default_content_setting_values (data_delete_automacally) expecting 'true'; no setting was changed. The feature exists as 'Delete data sites have saved to your device when you close … |
| `93eabf48` | chrome | done | infeasible-missed, evaluator-strict | Evaluator is func 'infeasible': only a vm_fail call scores. Agent made the stand-in change and called vm_done. |
| `3720f614` | chrome | done | infeasible-missed, osascript-struggle | Task is infeasible (Chrome interface language cannot be changed from inside Chrome on macOS); only vm_fail scores. The agent also altered a system setting as a stand-in. |
| `9f3f70fc` | chrome | None | misread-task, env-issue, answer-in-chat-only | Evaluator expects the active tab's filter selector links (class filter-selector-link) to be exactly over $60, women, jerseys, nike (NBA store page) and URL containing those terms. The tab was on nike.com with no filte… |
| `f0b971a1` | chrome | None | no-finish-call, answer-in-chat-only, budget-loop | Evaluator checks the active tab URL is https://www.nfl.com/scores/2019/post4. Final tab was a news article. |
| `2ae9ba84` | chrome | done | not-saved, verification-gap | Evaluator reads profile.name from ~/user-data-dir/Default/Preferences after a pkill/relaunch of Chrome; the textbox value was uncommitted (Chrome profile name field commits on Enter/blur) or not yet flushed, so profil… |
| `44ee5668` | chrome | done | shell-workaround, wrong-result-claimed-done, verification-gap | Evaluator reads ~/user-data-dir/Default/History (the test Chrome profile) via check_history_deleted and finds youtube entries still there; the agent edited a different profile directory and the DB bypasses Chrome (als… |
| `2ad9387a` | chrome | done | tool-bug, osascript-struggle, env-issue | Evaluator reads ~/user-data-dir/Default/Bookmarks (bookmark_bar_folders_names == {'Favorites'}) after pkill+relaunch with the harness profile. The folder was created in a Chrome that was relaunched after the AppleScri… |
| `7a5a7856` | chrome | done | tool-bug, osascript-struggle, env-issue | Evaluator reads ~/user-data-dir/Default/Bookmarks and needs bookmark_bar URLs == only the Illustrated Transformer URL. The final bookmark was made in a Chrome relaunched outside the harness (no remote debugging, proba… |
| `82279c77` | chrome | done | verification-gap, wrong-result-claimed-done | Evaluator parses the URL for list_price_max=50000, maximum_distance=50, zip=10001 and fuel_slugs[]=electric. The final URL was never checked and probably mixes the typed 'electrification=electric' param or lacks fuel_… |
| `368d9ba4` | chrome | done | wrong-result-claimed-done, evaluator-strict | Evaluator needs active-tab URL split on '/' [-2] == '{month}-weather' (november-weather from host date) and URL containing '/manchester/'. The final URL looks like it satisfies both, so the cause of the 0 could not be… |
| `1704f00f` | chrome | done | wrong-result-claimed-done, tool-bug, budget-loop | Evaluator expects URL locationName='Zürich' (the city entry, not the airport), carCategory=large, sortBy=PRICE, and puDay/puMonth/doDay/doMonth from 'next Monday' to 'next Friday' (evaluator on a Sunday computes tomor… |
| `2a729ded` | gimp | None | budget-loop, env-issue, tool-gap | Evaluator exports dog_without_background.png and SSIM-compares it to dog_cutout_gold.png. Final state was a partially deleted/selected image (or an undone one), not a clean cutout. |
| `734d6579` | gimp | budget | budget-loop, tool-gap, tool-bug | check_green_background loads the harness export green_background_with_object.png and requires every non-black pixel of the expected image to be green in the export; the export is still white-background -> 0. |
| `a746add2` | gimp | done | env-issue, verification-gap | check_config_status reads GIMP's action-history file after the postconfig Esc + Cmd+Q and needs the line (history-item "filters-vignette" 1). Cause of the miss not visible from the run: either menu activation via AX d… |
| `7a4deb26` | gimp | fail | false-infeasible, budget-loop, tool-gap | Task is feasible: check_brightness_decrease_and_structure_sim needs the exported edited_darker.png to be darker than the original woman_sitting_by_the_tree.png. No change was ever applied, and vm_fail ends the run any… |
| `e2dd0213` | gimp | fail | false-infeasible, tool-gap, budget-loop | check_textbox_on_leftside: the exported leftside_textbox.png must have its leftmost dark text pixel within 5% of the width. Text was never moved, and vm_fail on a feasible task is wrong. |
| `2e6f678f` | gimp | budget | infeasible-missed, budget-loop, tool-gap | Evaluator func is 'infeasible': passes only if the agent ends with vm_fail. Agent hit the budget with no finish call. |
| `58d3eeeb` | gimp | None | infeasible-missed, answer-in-chat-only, no-finish-call | Evaluator func 'infeasible' requires a vm_fail call. None was made. |
| `62f7fd55` | gimp | done | infeasible-missed, shell-workaround | Evaluator func 'infeasible' needs vm_fail; agent called vm_done, so 0. |
| `dbbf4b99` | gimp | done | infeasible-missed, shell-workaround | Evaluator func 'infeasible' needs vm_fail; vm_done gives 0. |
| `554785e9` | gimp | done | overwrote-source, wrong-result-claimed-done | check_saturation_increase_and_structure_sim compares the harness export edited_colorful.png against the original woman_sitting_by_the_tree2.png and requires saturation(export) > saturation(original). The original had … |
| `72f83cdc` | gimp | done | overwrote-source, wrong-result-claimed-done | check_image_mirror(result=berry_mirror.png export, expected=berry.png) flips berry.png and requires SSIM>=0.99 with the export. berry.png on disk was already the flipped image, so flip(berry.png) is the un-flipped ori… |
| `f723c744` | gimp | done | overwrote-source, wrong-result-claimed-done, budget-loop | check_contrast_increase_and_structure_sim requires contrast(export berries_contrast.png) > contrast(original berries.png). The original had been overwritten with the stretched version, so contrast is equal (not strict… |
| `77b8ab4d` | gimp | done | shell-workaround, tool-bug, wrong-result-claimed-done | check_file_exists_and_structure_sim compares result export.jpg with expected The_Lost_River_Of_Dreams.jpg (SSIM). The expected file no longer existed (deleted by the agent), so the check fails. |
| `f4aec372` | gimp | None | tool-gap, budget-loop, no-finish-call | check_triangle_position: exported Triangle_In_The_Middle.png must have the yellow pixel centroid within 5% of the image centre; the triangle stayed off-centre. |
| `7767eef2` | gimp | done | wrong-result-claimed-done, tool-gap, shell-workaround | check_config_status reads ~/Library/Application Support/GIMP/3.2/gimprc and needs the line (theme-color-scheme light). The agent only wrote CSS files, never changed the preference, so gimprc still lacks/has dark. |
| `21df9241` | libreoffice_calc | budget | app-knowledge-gap, budget-loop, tool-bug | Gold keeps numeric values and uses number formats (B 'in millions (M)', C 'in billions (B)' with one decimal and ' M'/' B'); result has headers 'A2','B2', LET/TEXT formulas and wrong values; csv sidecar also mismatches. |
| `04d9aeaf` | libreoffice_calc | budget | budget-loop, tool-bug, osascript-struggle | Evaluator RNSheet2: sheet must be named 'Sheet2' with Year/CA/FA/OA headers and (Sheet1!B3-B2)/B2 formulas for 2015-2018; result sheet 'Asset_Changes' with formulas concatenated in A2 ('=...=...=...'), stray '2016=(Sh… |
| `0a2e43bf` | libreoffice_calc | budget | budget-loop, tool-gap, misread-task | Gold has Total row A12:G12 and one chart; result file on disk is identical to the original (nothing saved) and total row skips Jun. |
| `26a8440e` | libreoffice_calc | budget | budget-loop, tool-bug, tool-gap | RNSheet2 must be named 'Sheet2' with Month/Total and =SUM(Sheet1!B2:B11)... rows; result sheet 'Monthly Totals' contains A1='Mar=SUM(D2:D11)', A2='Apr=SUM...' only. |
| `30e3e107` | libreoffice_calc | None | budget-loop, tool-gap, app-knowledge-gap | Needs Sheet2 A1 blue fill #0000ff, bold white text, 'Demographic Profile', plus three pivot tables; result sheet 'Demographics' is empty (A1 text missing), no pivots. |
| `7a4e4bc8` | libreoffice_calc | budget | budget-loop, tool-bug, app-knowledge-gap | Gold: B..F headers Date, First Name, Last Name, Order ID, Sales with rows rearranged; final state is a partial/undone mess, never saved. |
| `8b1ce5f2` | libreoffice_calc | budget | budget-loop, tool-gap, not-saved | Gold has bgcolor FF0000 on every Sat/Sun cell (style check) and sheet_data; the file was never saved and is incomplete. |
| `0cecd4f3` | libreoffice_calc | done | evaluator-strict, verification-gap | Gold sheet names are 'LARS Resources', 'LARS Resources (Backup)', 'LARS Resources (Offline)', 'Sheet3'; sheet_name rule fails on the 3rd name (task wording is ambiguous: gold seems to treat the old Sheet2 as the renam… |
| `1273e544` | libreoffice_calc | done | evaluator-strict, tool-gap | Rule RNSheet2 vs ENSheet2: result must have a sheet literally named 'Sheet2' with the copied column; agent named it 'Revenue' and placed it before Sheet1, so the by-name lookup fails. |
| `2bd59342` | libreoffice_calc | done | infeasible-missed, shell-workaround | Evaluator func is 'infeasible': only vm_fail scores 1. |
| `01b269ae` | libreoffice_calc | budget | misread-task, tool-bug, budget-loop | Gold: Level (B) and Student (C) blanks filled from above (B1:E30). Result: junk in A1:A30, B1 'Level' duplicated, 'Primary\n' in L2, B/C blanks still empty. |
| `abed40dc` | libreoffice_calc | done | misread-task, wrong-result-claimed-done, verification-gap | Gold has the 6 names in D2:D7 (header in D1, B untouched); agent's D column is empty and C holds the names. |
| `6054afcb` | libreoffice_calc | budget | misread-task, budget-loop, osascript-struggle | Gold: rows with #N/A hidden (hidden row property) and original data intact; final file has modified cell contents and no hidden rows. |
| `a9f325aa` | libreoffice_calc | done | misread-task, wrong-result-claimed-done, tool-bug | Gold expects TRIM+PROPER output in C (e.g. 'The Bune Collector'); final file has column C empty and different strings in D. |
| `0326d92d` | libreoffice_calc | budget | tool-bug, budget-loop, wrong-result-claimed-done | Evaluator needs Total row, Growth row (G13) and two charts with titles Total and Growth; result sheet has A1:G19 of partly-wrong junk, 0 charts. |
| `1d17d234` | libreoffice_calc | done | tool-bug, wrong-result-claimed-done | sheet_data on Sheet2: gold has only A1,A2,C2 values; result has stray 'A1:C1' text in B27 (from a click-then-type that landed in a cell), so Sheet2 dimension A1:D28 vs A1:D2. |
| `347ef137` | libreoffice_calc | done | tool-bug, wrong-result-claimed-done | sheet_data: result has stray 'A2:J14' in J32 (dimension A1:J32 vs gold A1:I32); gold's charts plot only column I (monthly total). |
| `4172ea6e` | libreoffice_calc | done | tool-bug, wrong-result-claimed-done | sheet_data: result sheet is A1:C14 with stray B14='C1'; gold is A1:C10. |
| `4188d3a4` | libreoffice_calc | None | tool-bug, verification-gap, no-finish-call | Freeze rule: gold freeze_panes C2; result G14 (and stray G13='C2'). |
| `4f07fbe9` | libreoffice_calc | done | tool-bug, wrong-result-claimed-done, verification-gap | compare_table on the CSV/xlsx: junk text in other cells (A2/A3, text object) differs from gold which only changes D1. |
| `4e6fcf72` | libreoffice_calc | done | tool-bug, wrong-result-claimed-done, budget-loop | Gold (Employee_Age_By_Birthday_gold.xlsx) has header 'Age' in D1, clean column A and ages in D; final read shows D1 empty and junk in A1:A6. |
| `51719eea` | libreoffice_calc | budget | tool-bug, misread-task, budget-loop | Evaluator pivot_table on Sheet2 (row_fields/data_fields) vs gold; Sheet2 does not exist in the final file. |
| `7efeb4b1` | libreoffice_calc | budget | tool-bug, budget-loop, verification-gap | Gold: B2:B29 = 'No. 1'..'No. 29'; final B has 'B3','No. 3'..'No. 7' only. |
| `eb03d19a` | libreoffice_calc | budget | tool-bug, budget-loop, verification-gap | Gold has the transposed 5x4 table at B8:E12; final has text junk in B8:B11. |
| `1954cced` | libreoffice_calc | None | tool-gap, tool-bug, budget-loop | Evaluator pivot_table on RNSheet2 (row field Invoice No., data field Count) - no Sheet2/pivot was created; also A1 was damaged. |
| `535364ea` | libreoffice_calc | budget | tool-gap, budget-loop, wrong-result-claimed-done | Evaluator checks a real pivot table (pivot_table rule: row_fields, data_fields, filter, col_fields) on Sheet2; SUMIF tables on 'RevenueSummary' do not count. |
| `a01fbce3` | libreoffice_calc | fail | tool-gap, budget-loop, app-knowledge-gap | sheet_print comparison vs gold_ru csv/xlsx; commas not produced and A1/B1 are corrupted. |
| `aa3a8974` | libreoffice_calc | None | tool-gap, no-finish-call, budget-loop | check_pdf_pages expects /Users/admin/Resize_Cells_Fit_Page.pdf with 1 page; file never exported. |
| `035f41ba` | libreoffice_calc | done | wrong-result-claimed-done, evaluator-strict | Rule sheet_data RNSheet2 vs ENSheet2: result needs a sheet literally named 'Sheet2' (agent named it Year_Profit; Sheet1 order also differs) and gold leaves Net Sales/Total COGS blank (J = B-C-D-SUM(F:H)); agent's extr… |
| `12382c62` | libreoffice_calc | done | wrong-result-claimed-done, tool-bug, evaluator-strict | Rule chart RNSheet2 vs ENSheet2: needs sheet named 'Sheet2' holding the chart (agent's sheet is 'SalesChart', placed first). Chart also pointed at empty $SalesChart cells; A1 contains stray 'A1:C11'. |
| `37608790` | libreoffice_calc | done | wrong-result-claimed-done, app-knowledge-gap, verification-gap | Gold keeps A 'Benedict Cucumberbach Manager' and fills B first, C last, D rank. Result A2='Benedict', B2='Cucumberbach', C2='Manager', D empty. |
| `3a7c8185` | libreoffice_calc | done | wrong-result-claimed-done, verification-gap | sheet_data compares all of A1:F36; Discount column F is misaligned with the sorted rows (Type formulas OK). |
| `d681960f` | libreoffice_calc | None | wrong-result-claimed-done, stale-control-index, verification-gap | Gold has VLOOKUP formulas in F10:F23; the file has F empty and I10:I23 filled, so sheet_data mismatch. |
| `6e99a1ad` | libreoffice_calc | done | wrong-result-claimed-done, tool-bug, verification-gap | compare via sheet_print of the CSV/xlsx: A1 is 'C2:C8' rather than 'Purpose', so the printed sheet differs from gold. |
| `f9584479` | libreoffice_calc | done | wrong-result-claimed-done, verification-gap | Gold has F6=SUM(B6:E6) etc. (F6,F13,F20); these cells are empty in the final file, sheet_data mismatch. |
| `7e429b8d` | libreoffice_calc | None | wrong-result-claimed-done, tool-bug, no-finish-call | Gold has VLOOKUP only in F2:F12; F13 empty. Final F13 holds =VLOOKUP(E13...) (#N/A, blank in read_table) and earlier typed junk may remain; also formulas used $A$1:$B$7 vs gold $A$2:$B$7 (values equal). |
| `ecb0df7a` | libreoffice_calc | done | wrong-result-claimed-done, verification-gap | data_validation rule needs ranges covering D2:D29 / D2:D1048576 with list {Pass,Fail,Held}; file has validation only on D2:D20. |
| `6f81754e` | libreoffice_writer | done | answer-in-chat-only, misread-task, shell-workaround | compare_docx_lines compares the saved docx lines to gold: the doc should be edited so only the first record of each train id remains (8 lines). Document unchanged and the reported count was off by one. |
| `88fe4b2d` | libreoffice_writer | budget | budget-loop, tool-bug, app-knowledge-gap | compare_docx_files (ignore_blanks=False) vs gold with an empty paragraph after each sentence of the first paragraph; the document was left in the original state. |
| `936321ce` | libreoffice_writer | done | budget-loop, tool-gap, osascript-struggle | compare_docx_tables expects a table matching the gold table (rows/cols/cell text of the converted comma-text); result docx has a malformed/duplicated table plus leftover text and was never saved. |
| `6ada715d` | libreoffice_writer | fail | false-infeasible, misread-task | compare_contains_image expects the saved docx to contain an inline image (graphicData) like the gold; document was untouched. |
| `b21acd93` | libreoffice_writer | budget | misread-task, budget-loop | compare_line_spacing checks the original docx paragraph line spacing (para1 single 240, para2 double 480, para3 1.5 = 360) against gold; the open docx was untouched (all default). |
| `8472fece` | libreoffice_writer | budget | tool-gap, shell-workaround, budget-loop | evaluate_colored_words_in_tables needs table words coloured red (vowel-initial) / blue (others) in the saved docx; final file on Desktop was corrupt/unchanged, nothing saved from the app. |
| `0a0faba3` | libreoffice_writer | None | verification-gap, no-finish-call, wrong-result-claimed-done | check_tabstops requires every non-empty paragraph to have the same tab stops (one right tab at 9360) as the gold; with the tab stop undone tabs1 is empty vs tabs2 has 1 -> returns 0. (Text-side tabs after word 3 were … |
| `0810415c` | libreoffice_writer | done | wrong-result-claimed-done, tool-gap, budget-loop | compare_line_spacing compares every paragraph's line_spacing against the gold docx (only paras 1-2 double). All other paragraphs were also double, so mismatch -> 0. |
| `0b17a146` | libreoffice_writer | done | wrong-result-claimed-done, verification-gap, tool-gap | compare_subscript_contains zips paragraphs/runs of result vs gold and needs a run that is subscript in both; gold has 8 subscript '2' runs in H2O occurrences; the wrong character was formatted so no matching run. |
| `72b810ef` | libreoffice_writer | done | wrong-result-claimed-done, verification-gap, tool-gap | evaluate_strike_through_last_paragraph requires EVERY run of the last paragraph to have strike set; first ~2 sentences were not struck. |
| `adf5e2c3` | libreoffice_writer | done | wrong-result-claimed-done, app-knowledge-gap, verification-gap | compare_docx_files vs gold: gold has Steinberg as an auto-numbered list item (number not in text), and a real cross-reference field (REF ... \r \h, via Insert>Cross-reference>Numbered Paragraph) in paragraph 10 with t… |
| `185f29bd` | multi_apps | budget | budget-loop, tool-gap, app-knowledge-gap | compare_pdfs over 7 files; only 2 of 7 PDFs exist/match; remaining 5 missing. |
| `236833a3` | multi_apps | budget | budget-loop, tool-bug, osascript-struggle | compare_docx_files vs gold: paragraph text mismatch and file not saved with correct content (final message says needs to be saved). |
| `2c1ebcd7` | multi_apps | budget | budget-loop, wrong-result-claimed-done, tool-bug | compare_references vs gold (content_only, base 0.93): reference strings differ from gold (page ranges/ordering/format errors, hand-retyped). |
| `3e3fc409` | multi_apps | fail | budget-loop, tool-gap, tool-bug | compare_table sheet 'unseen_movies' vs gold; sheet missing/incorrect. |
| `415ef462` | multi_apps | budget | budget-loop, tool-bug, tool-gap | check_list expects diff of ~/Documents/Finance/receipts/aws-invoice-2312.pdf to be empty and tally_book updated; neither done. |
| `5990457f` | multi_apps | budget | budget-loop, tool-bug, env-issue | literal_match of last row vs Scholar values (name, citations, h, i10, top paper, PDF link); xlsx last row unchanged. |
| `5df7b33a` | multi_apps | budget | budget-loop, tool-gap | compare_archive of PDFs vs gold zip: only 2 of 8 chapter PDFs present. |
| `3680a5ee` | multi_apps | done | evaluator-strict, misread-task | First checker requires a soffice process whose command column matches 'sh ' (i.e. launched via a shell/terminal wrapper); 'open -a' produces none -> 'use no terminal'. Uncertain whether second check (compare_csv) also… |
| `7f35355e` | multi_apps | done | evaluator-strict, wrong-result-claimed-done, shell-workaround | compare_result_files vs result_gold.txt: file content must match gold (probably single numeric value). |
| `00fa164e` | multi_apps | done | misread-task, wrong-result-claimed-done | compare_docx_tables vs gold: table must contain only LLM-based rows with the exact xlsx values; agent included an extra non-LLM row and rounded values (0.333 vs 0.333333333333333). |
| `26660ad1` | multi_apps | None | misread-task, app-knowledge-gap, no-finish-call | compare_time_in_speedtest_results needs /Users/admin/Test/Speed/Speedtest Results Export-.csv (time within 60 min); file absent. |
| `3a93cae4` | multi_apps | done | misread-task | compare_table vs gold Course Timetable xlsx: spreadsheet unchanged, no lecture slot. |
| `42d25c08` | multi_apps | None | misread-task, shell-workaround, no-finish-call | compare_epub needs '/Documents/Novels/Pass Through/Pass Through.epub'; file named 'Passing through the book.epub' not found. |
| `7e287123` | multi_apps | None | misread-task, shell-workaround, no-finish-call | compare_table sheet_print vs gold: sheet has 5 rows with exactly four headers; agent output has extra column/rows. |
| `5bc63fb9` | multi_apps | done | shell-workaround, wrong-result-claimed-done, verification-gap | compare_docx_files_and_ignore_new_lines vs gold: paragraph structure/text (and highlight runs) differ from gold's 6 response paragraphs. |
| `1f18aa87` | multi_apps | done | tool-bug, wrong-result-claimed-done, verification-gap | compare_docx_files vs Answer_Gold.docx: paragraph text/format must match gold; agent's rewritten doc (or answers aaaad/aaaaa derived by guess) differed. |
| `74d5859f` | multi_apps | done | tool-bug, misread-task, wrong-result-claimed-done | check_json manifest name=happy-extension, version=0.0.1, background.scripts=[background_script.js], browser_action...; files under ~/Projects/happy-extension/. |
| `4c26e3f3` | multi_apps | budget | tool-gap, budget-loop, app-knowledge-gap | check_brightness_decrease_and_structure_sim needs /Desktop/background.png brighter than original; file absent. |
| `67890eb6` | multi_apps | done | wrong-result-claimed-done, misread-task, verification-gap | compare_table sheet_data vs gold: column order (title, year, authors, link), header row, and PDF URLs differ. |
| `6f4073b8` | multi_apps | None | wrong-result-claimed-done, verification-gap, no-finish-call | compare_conference_city_in_order needs exact city list per row; ~10 of 21 wrong. |
