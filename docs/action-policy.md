# Action policy

How Otto decides whether a high-risk tool call runs. The gate is
`ActionPolicyMiddleware` in `backend/action_policy/`. It is attached to
the orchestrator and to every sub-agent. Settings live on
`AppConfig.action_policy` and are edited under Settings → Privacy & Security.

## What it does

- Classifies a tool by the server it came from and the tool name. Unknown
  tools on a server Otto did not ship are `unclassified` and ask by default.
- Ask, allow, or deny per category. A session can remember an approval
  for shell, reading mail or messages, calendar/notes edits, and
  work-tool writes. The first inbox or chat read in a session still
  asks. Desktop control is remembered for one agent run, not the whole chat.
- Unattended runs (schedules, triggers, hooks) deny anything that is not
  on that schedule or trigger's allowlist. Creating another schedule or
  trigger is always denied there.
- After the agent reads mail, messages, or a web page, outbound sends
  that were set to allow drop back to ask, and a session grant does not
  cover them.
- `write_file` and `edit_file` are `workspace_write`. Inside a folder you
  allowed, Balanced lets them run and records the decision. Outside that
  folder the file tools refuse the path even if the category is Allow. A
  read-only grant blocks those edits and keeps the folder out of the
  shell's write paths.
- `execute` and shell triggers run under Seatbelt on macOS when the
  sandbox is on. Sensitive locations such as `~/.ssh` stay denied even
  when a write grant covers a parent directory. The current session's
  files directory is re-opened after that deny, so the shell can read
  uploads there. Ask still does not get write access to it. The rest of
  Otto's app-data directory stays denied. The environment is
  scrubbed of API keys either way. Shell reads are not limited to the
  allowed folder.
  If the sandbox is on and cannot be applied, the command does not run.
  Inside a folder you allowed, the sandbox also denies writes to `.git`,
  `.vscode`, `.idea`, `.gitattributes`, `.envrc`, `AGENTS.md`, and
  `CLAUDE.md`. The file tools refuse those paths as well, including when
  the sandbox is off. A hook, a task file, or a clean filter planted
  there would run later, and the instruction files are loaded into the
  prompt rather than rewritten by the agent.
  Read-only commands (`git status`, `ls`, `cat`, and the same kind of
  command) skip the shell approval card. A folder can also store command
  prefixes, such as `npm test*`, that skip the card for that project.
  The global network switch still defaults off. A folder can turn shell
  network on for itself, and a host list limits that traffic to those hosts.
- Ask and Plan mode (Code tab) are read-only. The policy denies
  sends, scheduling, desktop control, AppleScript, settings changes, and
  other writes while a session is in Ask or Plan, whatever the category
  setting says. The file tools refuse project writes, and Ask refuses every
  file write. The shell gets no write access to the project, and in Ask it
  gets no write paths. Seatbelt always allows writes to the system temp
  folders, so a project in `/tmp` or `/private/var/folders`, or under an
  extra write path, cannot be shielded from the shell. Ask, Plan, and
  read-only folders then refuse `execute` for that project. Unattended runs always
  run as Agent. See [code.md](code.md#modes).
- Every gated decision is appended to `action_audit.jsonl` in the Otto
  app-data directory.

## What it does not do

- AppleScript, desktop control, Mail, and Messages cannot be sandboxed.
  They need Apple Events or the UI. The controls on them are the policy
  gate, macOS permission prompts, and the audit log. Approving a send
  means the send happens.
- Regexes on command text and AppleScript are warnings. They are easy
  to bypass. The sandbox is the control for the shell.
- The local API token stops a sandboxed command from calling Otto's
  HTTP API. It does not stop another program running as you. That
  program can read the token file. The file is mode 0600.
- `sandbox-exec` is deprecated. The fail-closed behaviour and the
  switch in Settings are the mitigation if it disappears.
- Approval fatigue is real. Scoped grants and the Balanced defaults
  are there to keep the prompts on the actions that leave the machine
  or are hard to undo.

## Behaviour you will notice

- A schedule or trigger no longer silently approves high-risk actions.
- Shell network access is off until you enable it. pip, npm, and git
  fetch fail until then. The sandbox error points at that switch.
- A sub-agent running a command asks, the same as the main agent.
- API keys are not in the shell's environment.
