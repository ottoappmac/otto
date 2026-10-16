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
| Files | One level of the folder at a time. Heavy directories such as `node_modules` and `.git` are skipped. |
| Conversation | The same transcript as Chat, with approval cards. |
| Changes | Files touched since the pre-turn checkpoint, a diff, and revert. Revert is refused while a run is in progress, and when the file changed after Otto's last edit. |
| Output | Command results from the turn. |
| Timeline | The run timeline. |

Drag the edge of the project list or the inspector to change its width. Double-click the edge to put it back. The widths are remembered. **Chat** and **Run** in the header open the same session in those views. The **×** next to the project name removes that folder from this session. The folder stays in Allowed folders, and the conversation stays open. It is unavailable while a run is in progress.

If a checkpoint cannot be taken, the turn still runs and the header says there is no checkpoint. Folders that are not a git repository keep a private copy of files the agent overwrites with the file tools. Changes made only through the shell in a non-git folder are not covered.

The agent can ask for a folder with `request_folder_access`. That does not grant it. You still confirm the path.

## Modes

The composer has four modes. Your last choice is remembered and applies to new sessions. The mode belongs to the session, so Chat, Run, and Code all see it. Chat and Run show a badge when it is Ask, Plan, or Debug. You change it only in Code. The switch is locked while a turn runs, and **Shift+Tab** in the message box cycles through the modes.

| Mode | What it does |
| --- | --- |
| Agent | Reads and edits the project and runs commands. This is the default. |
| Ask | Read-only. The agent answers questions about the code. It cannot write or edit any file, including session output. |
| Plan | Read-only in the project. The agent investigates and writes a plan. When the turn ends, **Build this plan** switches to Agent and asks it to implement the plan. |
| Debug | Same permissions as Agent. The agent is asked to reproduce the problem first, state hypotheses, add temporary log lines marked `otto-debug`, make the smallest fix, then remove the markers and re-run. The Changes list warns while any `otto-debug` marker is left in a file. |

Plan and Debug always run Deep, whatever the depth setting says. Runs that nobody is watching, such as schedules and triggers, always run as Agent.

**What is enforced.** In Ask and Plan, the file tools refuse writes and edits inside the project, including from subagents. Ask also refuses every file write, including `/output/`. The shell sandbox is not given write access to the project, and in Ask it is given no write paths at all. Actions that send, schedule, control the desktop, or change settings are denied. Ask and Plan turns do not take a checkpoint because nothing can change. The plan itself goes to `/output/plan.md` in the session, which is not your project.

**What is not.** What the model does with the instructions is not enforced. A local model can answer with the plan in chat instead of saving `/output/plan.md`; Code then shows the final answer as the plan and **Build this plan** still works. Whether a Debug turn follows the order above, or cleans up its markers, depends on the model. The marker warning is how you catch a miss.

**Shell limits.** The shell sandbox always allows writes to the system temp folders (`/tmp`, `/private/var/folders`) and to any extra write paths set in Settings. A project inside one of those cannot be made read-only for the shell. In that case Ask, Plan, and read-only folders refuse shell commands instead of pretending to protect the folder. On a system without Seatbelt, Ask and Plan refuse shell commands as well.

## Model

The model chip next to the composer is the same picker as in [Chat](chat.md). It changes the global model in Settings for the active provider, so it applies to every chat, not only this session. Open sessions pick up the new model on their next turn. Code does not start a new chat when the model changes, unlike Chat.
