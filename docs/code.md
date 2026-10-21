# Code

**Code** (`/code`) is the project view for an allowed folder. Open it from **Code** in the nav. A session here is the same session as [Chat](chat.md) and [Runs](runs.md): switching views keeps the conversation, and a live run is one event stream shared by the open views.

## Allowed folders

Otto does not read or edit an arbitrary directory. You allow a folder from Code, from the **Project** menu in Chat, or from Settings → Privacy & Security → Allowed folders.

The confirm step shows the path after symlinks are resolved.

| Mode | What the agent can do |
| --- | --- |
| Read only | Read the tree. File edits are refused. The shell's working directory is the folder, and the sandbox does not grant writes there. |
| Edit, with checkpoints | Read and edit. Before each turn in a git repository, Otto stores a snapshot under `refs/otto/checkpoints/…` using a temporary index, so your branch, index, and stash are not moved. |

Home, system directories, and any folder that contains SSH keys, keychains, or Otto's own data cannot be allowed.

Shell commands can still read most of the rest of the disk. The sandbox blocks the sensitive paths above. On systems without Seatbelt, the grant still bounds the file tools, not every shell read.

## Layout

| Region | Contents |
| --- | --- |
| Projects | Allowed folders. **+** starts a coding session in the selected folder. **×** on the selected folder removes it. Sessions that used it lose the folder; the files on disk stay. |
| Sessions | Chats already bound to the selected folder, with a dot while a run is in progress. The list updates on its own when a title or run changes, including a chat started from elsewhere. |
| Files | The folder tree, in the project list and in the Files tab. It loads one level at a time, and skips heavy directories such as `node_modules` and `.git`. Click a text file to edit it; Save writes your edit, and Undo all will not revert it. Other files open as a preview. |
| Conversation | The same transcript as Chat: collapsed thoughts, tool rows, sub-agent blocks, approval and question cards, a working indicator, and a Stop button while a run is going. A question from the agent (`ask_user`) appears as soon as it is asked. |
| Changes | Files touched since a checkpoint, with added, modified, and deleted status, line counts, and a coloured diff. A file Otto did not change is marked as changed outside Otto and is not reverted. A turn menu switches checkpoints. Revert turn asks first and restores only Otto's files. The undo icon on a file reverts that file. On a hunk, Accept keeps it and Reject reverts it. Revert is refused while a run is in progress, when the file was changed outside Otto, and when the file changed after Otto's last edit in that turn. |
| Output | Each command the agent ran, with its output and the exit code when the command failed. Running commands show live lines. |
| Timeline | The run timeline. |

**Reviewing changes.** Each changed file in the Changes tab shows a status icon, its name and folder, and line counts. Click a file to open its diff in place. The diff is syntax-highlighted with line numbers, collapses unchanged stretches between changes, and switches between unified and split views (the choice is remembered). Use the check icon on a change to keep it and the undo icon to reject it. The file editor highlights code, shows line numbers, marks unsaved edits with a dot, and saves with ⌘S. Deleting a session hides it at once and removes it after a few seconds; **Undo** in the toast brings it back. While you are scrolled up in a long conversation, a **Jump to latest** button returns you to the bottom. Icon-only buttons show a tooltip with their shortcut after a short hover or on keyboard focus.

Drag the edge of the project list or the inspector to change its width. Double-click the edge to put it back. The widths are remembered. On a narrow window both panes start collapsed; **Projects** and the panel icon in the header bring them back. **Chat** and **Run** in the header open the same session in those views. **Network on** / **Network off** turns shell network on for this folder while the global switch stays off. **×** next to the project name removes that folder from this session. The folder stays in Allowed folders, and the conversation stays open. It is unavailable while a run is in progress.

After a turn that changed files, a bar above the message box offers **Review**, **Undo all**, and **Keep**. **Undo all** restores files Otto changed and leaves edits made outside Otto. **Keep** only dismisses the bar. Hover a message to edit and resend it, or to restore files to the checkpoint taken before that message. Editing can restore files as well. Type `@` to mention a project file. The **+** in the message box attaches files the same way Chat does: pick them, paste them, or drop them, including folders. They are sent with the next message. While a run is going, Enter adds the text as context instead of starting a new turn. Esc stops the run. ⌘⏎ allows the newest approval card and ⌘⌫ rejects it; high-risk cards still wait for a click.

If a checkpoint cannot be taken, the turn still runs and the header says there is no checkpoint. Folders that are not a git repository keep a private copy of files the agent overwrites with the file tools. Changes made only through the shell in a non-git folder are not covered.

The agent can ask for a folder with `request_folder_access`. That does not grant it. You still confirm the path.

## Starting a session

With a project selected and no session open, you can type straight into the message box. Sending creates a coding session in that project and sends your text as its first message. The **+** next to Sessions starts an empty session instead. If there is no allowed folder yet, the box stays disabled until you allow one.

## Modes

The composer has four modes. Your last choice is remembered and applies to new sessions. The mode belongs to the session, so Chat, Run, and Code all see it. Chat and Run show a badge when it is Ask, Plan, or Debug. You change it only in Code. The switch is locked while a turn runs, and the server rejects a mode change until that turn finishes. **Shift+Tab** in the message box cycles through the modes.

| Mode | What it does |
| --- | --- |
| Agent | Reads and edits the project and runs commands. This is the default. |
| Ask | Read-only. The agent answers questions about the code. It cannot write or edit any file, including session output. |
| Plan | Read-only in the project. The agent investigates and writes a plan. Otto saves that plan to `/output/plan.md`, creating the file the first time and replacing it when a later Plan turn revises it. When the turn ends, **Build this plan** shows the plan and asks you to confirm before it switches to Agent and implements it. |
| Debug | Same permissions as Agent. The agent is asked to reproduce the problem first, state hypotheses, add temporary log lines marked `otto-debug`, make the smallest fix, then remove the markers and re-run. If the project has tests, it is asked to add a regression test that fails without the fix. The Changes list warns while any `otto-debug` marker is left in a file. |

Plan and Debug always run Deep, whatever the depth setting says. Runs that nobody is watching, such as schedules and triggers, always run as Agent.

**What is enforced.** In Ask and Plan, the file tools refuse writes and edits inside the project, including from subagents. Ask also refuses every file write, including `/output/`. The shell sandbox is not given write access to the project, and in Ask it is given no write paths at all. Actions that send, schedule, control the desktop, or change settings are denied. Ask and Plan turns do not take a checkpoint because nothing can change. When a Plan turn ends with a plan, Otto writes that reply to `/output/plan.md` in the session, which is not your project. The first plan creates the file. A later Plan turn that replies with a revised plan replaces it. A question or a short note leaves the file as it is.

**What is not.** What the model does with the instructions is not enforced. A local model can answer with questions instead of a plan; Code then shows that reply until a plan is saved, and **Build this plan** still works. Whether a Debug turn follows the order above, or cleans up its markers, depends on the model. The marker warning is how you catch a miss.

**Shell limits.** The shell sandbox always allows writes to the system temp folders (`/tmp`, `/private/var/folders`) and to any extra write paths set in Settings. It denies writes under `.git`, `.vscode`, and `.idea`, and writes to `.gitattributes`, `.envrc`, `AGENTS.md`, and `CLAUDE.md`. The file tools refuse those paths too, including when the sandbox is off. A project inside a temp folder or an extra write path cannot be made read-only for the shell. In that case Ask, Plan, and read-only folders refuse shell commands instead of pretending to protect the folder. On a system without Seatbelt, Ask and Plan refuse shell commands as well. Test commands (`pytest`, `npm test`, `pnpm test`, `yarn test`, `go test`, `cargo test`, `make test`) get up to 10 minutes. Other commands keep the ordinary shell timeout.

## Approvals

Commands and other gated actions show the same approval cards as Chat, so the rules in [action-policy.md](action-policy.md) apply here unchanged. Code now honors the same shortcuts as Chat:

- **Always allow** on a command card approves later command cards in that session without a click. It resets when you open another session.
- **Auto-approve commands** in Settings does the same for every session.
- Neither shortcut covers high-risk commands (for example `rm -rf`, force-push, raw disk writes) or any card the server marks as always-ask. Those wait for you.
- **Allow … this session** on a policy card grants only that category, as in Chat.
- A single read-only command (`git status`, `git diff`, `git log`, `ls`, `cat`, `rg`, `grep`, `find`, and the same kind of command) skips the card. A chain (`&&`, `|`, `;`) still asks, and `find -delete` or `find -exec` still asks.
- Each allowed folder can list command prefixes, such as `npm test*`, under Settings → Privacy & Security → Allowed folders. A matching `execute` skips the card for that project. A command the server marks as dangerous still asks. The same row has an optional host list: when it is filled in, shell network for that folder is limited to those hosts. Empty means every host while network is on.

Answering a card marks the run as running again, so the Runs page stops showing "Awaiting input".

## How the agent works on code

The coding agent works in the mapped folder (`/links/<name>/`). Scratch files go to `/output/` in the session. If no folder is mapped, it stops and asks you to map one.

It is asked to prove a change works, not only to make it:

- It searches with `glob` and `grep`, and with `search_code` for a symbol. `index_project` indexes up to 80 source files when Memory embeddings are on, so a later `search_code` can also match by meaning. Heavy directories such as `node_modules` and `.git` are skipped.
- It reads each file it will change, then looks for the project's own test, typecheck, lint, and build commands (package.json, pytest.ini, Makefile, CI config, README) before it runs anything. It does not invent a command.
- A bug fix starts with a test that fails for the bug. New behavior gets tests for the normal case, an edge case, and a failure case, in the project's existing test framework and layout. It does not add a test dependency without asking.
- A refactor runs the existing tests before and after.
- It runs the narrowest test first, then the wider suite for what it touched. If a check fails it fixes the cause and runs it again. It is told never to weaken, skip, or delete a test to get a pass.
- After edits it calls `project_check` when the project declares one. Otto runs at most two of: `npm run typecheck` (or `npx tsc --noEmit` when package.json has no typecheck script and `tsconfig.json` is present), `npm run lint`, and `ruff check .` when `pyproject.toml` or `ruff.toml` exists. When a turn wrote project files and such a command exists, Otto runs that check itself, capped at 20 seconds, and appends the result to the transcript. The agent's own `project_check` call gets 45 seconds.
- A dev server or watcher is started with `process_start`, read with `process_output` and `process_ports`, and stopped with `process_stop`. A session keeps at most four. They use the same sandbox and network grant as `execute`. Ask and Plan cannot start one. Starting and stopping still wait on the shell approval card.
- It commits, branches, or opens a pull request only when you ask. `git_commit` commits the whole working tree with no git hooks or filters, including edits you made outside Otto, and it refuses a repository that has no commit yet. `git_branch` creates a branch. `git_worktree` adds a worktree under `.otto/worktrees/<name>`. `git_pr` opens a pull request with `gh` and does not push; the branch has to already be on the remote. These four, plus `project_check`, wait on the shell approval card. The read-only and prefix shortcuts apply to `execute` only.
- It follows AGENTS.md and CLAUDE.md when they are included with the project. It cannot rewrite them, or anything under `.git`, `.vscode`, or `.idea`, or `.gitattributes` and `.envrc`. If the project has no AGENTS.md, a draft of project rules goes to `/output/rules.md` in the session.
- Its final report lists the files changed, the tests added, each command with pass or fail, and anything it could not verify.

The checks Otto runs itself, the protected paths, and the refusal when no folder is mapped are enforced. The rest is prompt guidance. A small local model can skip it, so read the report and the Output tab before you trust a change.

## Model

The model chip next to the composer is the same picker as in [Chat](chat.md). It changes the global model in Settings for the active provider, so it applies to every chat, not only this session. Open sessions pick up the new model on their next turn. Code does not start a new chat when the model changes, unlike Chat.
