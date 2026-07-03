"""macOS implementation of ComputerNavigator.

Wraps the ``MacOSToolkit`` from ``macos_tools.py`` and the system-prompt
fragments that the ``ComputerVoyagerGraph`` agent needs.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from tools.navigation.computer.base import ComputerNavigator
from tools.navigation.computer.macos_tools import (
    CONTROL_INTERACTION_RULES,
    MACOS_READ_SCREEN_VISION,
    MACOS_TOOLS,
    MACOS_VISION_TOOLS,
    MacOSToolkit,
)


_SYSTEM_INSTRUCTIONS = """\
You are a macOS desktop automation agent.

get_screen_controls returns an indented tree of UI controls:
  AppName
   window 'Title'
    toolbar
     [1]B'Delete' [2]B'Archive'
    splitgroup
     [3]TF'Search'
     [4]CE'Inbox' [5]CE'Sent'

Format: [index]CODE'label'
- index: numeric ID for press_control, type_into_control, click, get_control_value
- CODE: role abbreviation (see control_interaction_rules for the full table)
- Indented lines without an index are structural containers (window, toolbar, \
group, splitgroup, menubar, …) — they show where controls sit in the UI but \
are not directly interactable.

<input_fields_note>
ST (StaticText) items are read-only labels — never type into them.
The actual input is always the nearest TF (TextField) or TA (TextArea) with a
similar name, e.g. label 'Subject:' → TF 'Subject'.
  → type_into_control(index) for TF / TA / CX
  → click(index) then type_text("...") for web-view editors (role G/WebArea
    with no corresponding TF), canvas inputs, and rich-text areas.
</input_fields_note>

<app_launch_protocol>
Step 0  list_apps('<search>') — ALWAYS call this first to find the exact app name.
        Pass a short lowercase substring, e.g. list_apps('slack').
        Use the name returned by list_apps() in all subsequent calls.
Step 1  launch_app('<ExactName>') — opens, activates, and waits for controls.
        → Controls returned?  Proceed to interact.
        → Timed out?  Continue to Step 2.
Step 2  spotlight_search('<ExactName>') → hotkey('return') → activate_app('<ExactName>')
        → get_screen_controls('<ExactName>') to verify controls.
        → Still no controls?  launch_app('/System/Applications/<ExactName>.app')
</app_launch_protocol>

{control_interaction_rules}

<rules>
Observation:
- ALWAYS call get_screen_controls after every action — indices change after \
actions (dialogs open, menus expand). Never reuse stale indices.
- Never guess an app name or control index — read them from tool output.
- Always call list_apps('<search>') before launch_app() to get the exact app name.

Tool selection:
- press_control for B-type roles (B, CB, RB, LN, MI, MB, PB).
- click(index) for all other interactive roles (SL, G, CE, RW, Image).
- If press_control has no visible effect, fall back to click(index).

Verify after acting (predict → act → verify):
- BEFORE a navigation/activation press, state the expected resulting state in \
one short phrase, e.g. "after this the window title should contain \
'2-introductions'" or "the checkbox value should flip to 1".
- AFTER the action, re-read and confirm that expectation actually happened. \
Use the CHEAPEST signal that proves it: get_screen_controls (window title is in \
the top 'window ...' line) or read_app_dom for Electron apps; read_screen / a \
screenshot only when there is no structured signal (canvas/image-only UI).
- If press_control returns a "state did not change / likely ignored" notice, or \
the re-read shows the SAME state you saw before, the press did NOT work — this \
is normal for Electron/Chromium controls (Slack, Discord, VS Code) where AXPress \
is accepted but never fires the real click. Do NOT press the same index again. \
Escalate: click(index) (needs the app frontmost) or, if focus cannot be \
obtained, report that the control could not be activated.

Batching:
- ALWAYS use batch_actions when you already have all the control indices from \
get_screen_controls and the full sequence is known (e.g. filling a form, clicking \
a button then typing). Do NOT make separate calls for each step.
- steps is a plain text string with one tool call per line (same syntax as \
individual calls). Maximum 10 steps per batch.
- Form fill example:
  batch_actions(steps="type_into_control(index=99, text='user@example.com')\n\
click(index=105)\ntype_into_control(index=105, text='Subject line')\n\
click(index=77)\ntype_text(text='Body content here')")

Completion:
- When the task is done, respond with your final answer in plain text. \
Do NOT call any more tools.
- Never repeat the same tool call with the same arguments — if an action \
had no effect, try a different approach or report what happened.

Errors:
- Never repeat a failing tool call — escalate to the next protocol step.
- If the screen state is unexpected, re-read and reassess before acting.
- Only use Spotlight if launch_app already timed out.

Electron / Chromium apps (Slack, Discord, VS Code, Teams):
- Their AX tree is built lazily — a first empty get_screen_controls is normal. \
Recover with activate_app → wait_for_controls(app, timeout=20).

Accessibility-disabled apps (OCR fallback):
- If get_screen_controls reports the app's accessibility interface is DISABLED \
(e.g. Slack), AX cannot read it and retrying will not help. Switch to OCR.
- READING needs no focus: read_screen(app) returns the window's actual text \
(occlusion-proof) even when the app is in the background.
- For READ/summarise tasks, reading is usually enough: read_screen(app), \
scroll_at(x, y, 'down') to page through, and report what you read. Do NOT try to \
drive search boxes or type unless the task truly requires changing the view.
- ACTING needs focus: to click or type you must hold focus. Prefer the \
focus-guarded tools — click_text(text, app) and type_text(text, app_name=app) — \
which activate the app, CONFIRM it is frontmost, and refuse (without acting) if \
they cannot get focus. Use find_text_on_screen + click_at only after focus is \
confirmed.
- OBSERVE AFTER EVERY ACTION: OCR coordinates do not go stale like AX indices, \
but you MUST still call read_screen(app) again after each click_text / type_text / \
scroll_at to confirm the screen actually changed before acting again. If the new \
read_screen looks identical to the previous one, the action did NOT have the \
intended effect — do NOT repeat the same click; reassess (different on-screen \
text, scroll first, or report what is visible).
- These OCR tools return text and work without a vision model.

Focus drift (critical):
- click_text / type_text(app_name=...) / activate_app tell you whether focus was \
obtained. If activate_app reports it could NOT bring the app to the foreground, or \
a tool says the wrong app holds focus, STOP — do not click/type. The host app is \
holding focus; retrying the same action will keep failing. Fall back to \
read_screen and report what is visible.
- NEVER repeat an identical click/hotkey/type call that already reported the wrong \
frontmost app — it will not start working on the 2nd, 5th, or 10th try.

Honesty:
- Report only what you actually read from the live UI. If you could not read the \
requested content, say so — never fabricate it from window titles or timestamps.
</rules>
"""

_VISION_ADDENDUM = """
<vision_rules>
capture_app_screenshot(app_name) — use only when:
- get_screen_controls returns few/no controls (Electron apps)
- Unexpected behaviour — verify actual screen state
- Final result verification

Do NOT screenshot every step — get_screen_controls is primary.

read_screen(app) — for you (a vision model) this returns BOTH the OCR text and \
a screenshot of the same window capture. Read the text for exact wording and look \
at the image for icons, badges, layout, and anything the text misses. No separate \
capture_app_screenshot is needed after read_screen.
</vision_rules>
"""


class MacOSNavigator(ComputerNavigator):
    """macOS desktop navigator backed by the Accessibility API + pyautogui.

    Parameters
    ----------
    toolkit : MacOSToolkit | None
        Pre-configured toolkit instance.  When ``None`` (default) the
        module-level default singleton is used (reads from ``Environment``).
        Pass a custom instance to override any config value::

            from utilities.environment import Environment
            from tools.navigation.computer import MacOSToolkit

            tk = MacOSToolkit(
                ax_ipc_timeout=Environment.get_ax_ipc_timeout(),
                scan_max_depth=Environment.get_scan_depth(),
                scan_max_elements=Environment.get_scan_max_elements(),
                scan_max_workers=Environment.get_scan_max_workers(),
            )
            nav = MacOSNavigator(toolkit=tk)
    """

    def __init__(self, toolkit: MacOSToolkit | None = None) -> None:
        self._toolkit = toolkit

    def get_tools(self, *, vision: bool = False) -> list[BaseTool]:
        if self._toolkit is not None:
            tools = list(self._toolkit.tools)
            read_screen_vision = self._toolkit.read_screen_vision
            vision_tools = self._toolkit.vision_tools
        else:
            tools = list(MACOS_TOOLS)
            read_screen_vision = MACOS_READ_SCREEN_VISION
            vision_tools = MACOS_VISION_TOOLS
        if vision:
            # Swap the text-only read_screen for the text+image combo so the
            # VLM also sees the captured window pixels. The tool name is
            # unchanged ("read_screen"), so prompts and pruning are unaffected.
            tools = [read_screen_vision if t.name == "read_screen" else t for t in tools]
            tools.extend(vision_tools)
        return tools

    def get_system_instructions(self, *, vision: bool = False) -> str:
        prompt = _SYSTEM_INSTRUCTIONS.format(
            control_interaction_rules=CONTROL_INTERACTION_RULES,
        )
        if vision:
            prompt += _VISION_ADDENDUM
        return prompt

    def get_control_interaction_rules(self) -> str:
        return CONTROL_INTERACTION_RULES
