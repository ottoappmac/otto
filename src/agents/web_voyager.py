"""Web Voyager agent — multi-action per turn, single execute_tools node."""
import asyncio
import base64
import functools
import logging
import uuid
from io import BytesIO
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Union

from langgraph.checkpoint.memory import MemorySaver

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langgraph.graph import END, START, StateGraph
from PIL import Image
from typing_extensions import TypedDict

from callbacks.agent_callback import AgentCallback, CallbackMixin
from chat_models.mlx.chat_vlm import MLXVLChatModel
from tools.schemas import BBox

logger = logging.getLogger(__name__)


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """\
<role>
You are a robot browsing the web, just like a human. You complete tasks by interacting
with web pages. Each iteration you receive an Observation: a screenshot with Numerical
Labels in the TOP LEFT corner of each Web Element, plus text descriptions.
Identify the Numerical Label of the element you need and choose an action.
</role>

<actions>
Available actions — each Action MUST be on its own line, using EXACTLY this format.
Copy the syntax exactly, including semicolons and brackets.

  Click [Numerical_Label]               — example: Click [14]
  Type [Numerical_Label]; [Content]     — example: Type [7]; hammers
  Scroll [Numerical_Label or WINDOW]; [up or down]
                                        — example: Scroll [WINDOW]; [down]
                                        — example: Scroll [3]; [up]
  Wait
  GoBack
  GoToUrl [url]                         — example: GoToUrl [https://example.com]
  WebSearch
  Log [your note here]                  — example: Log [cheapest hammer is $12]
  ANSWER; [content]                     — example: ANSWER; The cheapest hammer costs $12.

IMPORTANT: Scroll always requires TWO arguments separated by a semicolon.
  Correct:   Scroll [WINDOW]; [down]
  WRONG:     Scroll down the page
  WRONG:     Scroll [down]
</actions>

<rules>
CRITICAL — read and follow every rule below before responding.

1. You may output multiple Action lines per iteration. Each Action must be on its own
   line. Chain only independent actions (e.g. scroll then click). Do NOT chain actions
   that depend on the result of a previous one.
2. When clicking or typing, ensure you select the correct bounding box by its Numerical
   Label. Labels are in the top-left corner and share the colour of their bounding box.
3. If the element you need is NOT visible in the current screenshot, you MUST Scroll
   first (WINDOW or a scrollable container) to bring it into view before interacting.
   Do NOT guess labels for elements you cannot see.
4. Navigation bars, menus, footers, and buttons are often above or below the visible
   viewport. Before concluding an element does not exist:
   a. Scroll [WINDOW]; [up] to check for top navigation bars, headers, or sticky menus.
   b. Scroll [WINDOW]; [down] to check for content below the fold, footers, or
      "Load More" / pagination controls.
   c. Look for hamburger menus (≡), "More" links, or expandable sections that may
      reveal hidden navigation items — click them to expand.
   d. If a scrollable container (sidebar, dropdown, modal) is present, scroll INSIDE
      that container using its Numerical Label instead of WINDOW.
5. Do not interact with useless elements like Login, Sign-in, or donation prompts.
6. Select strategically to minimise wasted steps.
7. If you see a Web Element of type <iframe/>, its internal elements are NOT labeled.
   Use GoToUrl with the iframe's src URL to navigate directly to it as a full page.
8. Before acting, review your scratchpad for notes you saved with Log — they may
   contain key facts, intermediate findings, or instructions you recorded earlier.
   Treat logged notes as ground truth for this session.
9. Use ANSWER only when you have the information requested by the task. Include the
   complete answer after the semicolon.
</rules>

<response_format>
Your reply MUST strictly follow this format:

Thought: {{Your brief thoughts (briefly summarise the info that will help ANSWER)}}
Action: {{First action}}
Action: {{Second action (optional)}}
Action: {{More actions as needed (optional)}}

Then the User will provide:
Observation: {{A labeled screenshot Given by User}}
</response_format>
"""


def build_web_voyager_prompt(extra_system_prompt: Optional[str] = None) -> ChatPromptTemplate:
    """Local reproduction of wfh/web-voyager, updated to allow multiple actions per turn.

    Args:
        extra_system_prompt: Optional additional instructions appended to the system prompt.
    """
    system_content = _SYSTEM_TEMPLATE
    if extra_system_prompt:
        system_content = system_content.rstrip() + "\n\n" + extra_system_prompt.strip()
    return ChatPromptTemplate.from_messages([
        ("system", system_content),
        MessagesPlaceholder("scratchpad", optional=True),
        HumanMessagePromptTemplate.from_template([
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,{img}"}},
            {"type": "text", "text": "{bbox_descriptions}"},
            {"type": "text", "text": "{input}"},
        ]),
    ])


# ── State & result types ───────────────────────────────────────────────────────

class Action(TypedDict):
    action: str
    args: Optional[List[str]]


class ParsedResponse(TypedDict):
    thought: str
    predictions: List[Action]


class AgentState(TypedDict):
    input: str
    img: str
    bboxes: List[BBox]
    bbox_descriptions: str
    prediction: List[Action]
    thought: str           # most recent Thought from the LLM
    logs: List[str]        # notes written by the agent via the Log action
    scratchpad: List[BaseMessage]
    observation: str       # newline-joined observations from all actions
    step: int              # monotonic step counter (avoids regex in scratchpad)


class RunResult(TypedDict):
    answer: Optional[str]
    thoughts: List[str]          # One thought per agent step
    actions: List[List[Action]]  # One action list per agent step
    observations: List[str]      # One observation string per execute_tools step
    logs: List[str]


class AgentEvent(TypedDict):
    type: Literal["agent"]
    thought: str
    prediction: List[Action]
    img: str


class ToolsEvent(TypedDict):
    type: Literal["tools"]
    observation: str
    logs: List[str]


StreamEvent = Union[AgentEvent, ToolsEvent]


# ── WebVoyagerGraph ───────────────────────────────────────────────────────────

class WebVoyagerGraph(CallbackMixin):
    """Web Voyager agent that executes multiple browser actions per turn.

    The LLM may output several ``Action:`` lines per response. All are parsed
    into ``state["prediction"]`` as a list and executed sequentially in a
    single ``execute_tools`` node.

    The agent owns a ``PlaywrightComputerUseNavigator`` instance — no external
    browser setup is required.  Compatible with ChatOpenAI, ChatAnthropic, and
    MLXVLChatModel.

    Args:
        llm: The chat model to use.
        width: Browser viewport width in pixels.
        height: Browser viewport height in pixels.
        headless: Whether to run the browser in headless mode.
        system_prompt: Optional extra instructions appended to the default system
            prompt. Use this to inject task-specific guidance without replacing
            the core browsing instructions.

    Usage::

        from langchain_anthropic import ChatAnthropic
        agent = WebVoyagerGraph(ChatAnthropic(model="claude-sonnet-4-5", max_tokens=4096))
        result = await agent.arun("What is the capital of France?")
        print(result["answer"], result["thought"], result["actions"], result["logs"])

        # with extra system instructions
        agent = WebVoyagerGraph(llm, system_prompt="Always prefer English-language sources.")

        # streaming
        async for event in agent.stream("Find the LangGraph docs"):
            if event["type"] == "agent":
                print(event["thought"], event["prediction"])
    """

    def __init__(
        self,
        llm: BaseChatModel,
        width: int = 1280,
        height: int = 800,
        headless: bool = False,
        system_prompt: Optional[str] = None,
        callback: Optional[AgentCallback] = None,
    ):
        # Lazy import: the Python ``playwright`` package is deliberately NOT
        # bundled into the packaged backend (see scripts/build_backend.py —
        # browser automation ships via the external ``@playwright/mcp`` Node
        # service, not these in-process bindings).  Keeping this import at
        # module scope would crash every backend startup, because
        # ``agents/__init__.py`` imports this module eagerly.  Importing here
        # means the cost (and the playwright requirement) is only paid if a
        # ``WebVoyagerGraph`` is actually constructed.
        try:
            from tools.navigation.web.playwright_navigator import (  # noqa: PLC0415
                PlaywrightComputerUseNavigator,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "WebVoyagerGraph requires the in-process Playwright bindings, "
                "which are not bundled in the packaged app. Use the "
                "'playwright-mcp' MCP-based browser agent instead, or install "
                "the 'playwright' package for local development."
            ) from exc

        self.llm = llm
        self._callback = callback
        self.prompt_template = build_web_voyager_prompt(extra_system_prompt=system_prompt)
        self._is_mlx = isinstance(llm, MLXVLChatModel)
        self.navigator = PlaywrightComputerUseNavigator(
            width=width,
            height=height,
            headless_flag=headless,
        )
        self._checkpointer = MemorySaver()
        self._run_config: Optional[Dict[str, Any]] = None  # set by start()
        self._tool_map = self._build_tool_map()
        self.graph = self._build_graph()

    # ── Browser lifecycle ─────────────────────────────────────────────────────

    async def start(self, start_url: Optional[str] = None) -> None:
        """Connect the browser and optionally navigate to a starting URL."""
        self._run_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        await self.navigator.connect()
        if start_url:
            logger.info("Navigating to start URL: %s", start_url)
            await self.navigator.go_to(start_url)

    async def stop(self) -> None:
        """Close the browser."""
        if self.navigator.browser:
            logger.info("Closing browser")
            await self.navigator.browser.close()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _initial_state(self, question: str) -> AgentState:
        """Return a blank ``AgentState`` for the start of a new run."""
        return {
            "input": question,
            "img": "",
            "bboxes": [],
            "bbox_descriptions": "",
            "prediction": [],
            "thought": "",
            "logs": [],
            "scratchpad": [],
            "observation": "",
            "step": 0,
        }

    # ── Annotation ────────────────────────────────────────────────────────────

    async def _annotate(self, state: AgentState) -> AgentState:
        marked = await self.navigator.annotate()
        return {**state, "img": marked.img, "bboxes": marked.bboxes}

    def _format_descriptions(self, state: AgentState) -> AgentState:
        labels = []
        for i, bbox in enumerate(state["bboxes"]):
            text = (bbox.get("ariaLabel") or bbox.get("text", "")).strip()
            labels.append(f'{i} (<{bbox.get("type")}/>): "{text}"')
        return {**state, "bbox_descriptions": "\nValid Bounding Boxes:\n" + "\n".join(labels)}

    # ── VLM call ──────────────────────────────────────────────────────────────

    async def _call_vlm_native(self, state: AgentState) -> str:
        chain = self.prompt_template | self.llm | StrOutputParser()
        return await chain.ainvoke({
            "input": state["input"],
            "img": state["img"],
            "bbox_descriptions": state["bbox_descriptions"],
            "scratchpad": state["scratchpad"],
        })

    async def _call_vlm_mlx(self, state: AgentState) -> str:
        formatted = self.prompt_template.invoke({
            "input": state["input"],
            "img": state["img"],
            "bbox_descriptions": state["bbox_descriptions"],
            "scratchpad": state["scratchpad"],
        })
        text_parts = []
        for msg in formatted.messages:
            content = msg.content
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block["text"])

        pil_img = Image.open(BytesIO(base64.b64decode(state["img"])))
        fn = functools.partial(
            self.llm._generate,
            messages=[HumanMessage(content="\n".join(text_parts).strip())],
            images=[pil_img],
        )
        result = await asyncio.to_thread(fn)
        return result.generations[0].message.content

    async def _call_vlm(self, state: AgentState) -> str:
        if self._is_mlx:
            return await self._call_vlm_mlx(state)
        return await self._call_vlm_native(state)

    # ── Output parser ─────────────────────────────────────────────────────────

    def _parse_response(self, text: str) -> ParsedResponse:
        """Parse the raw LLM response into a ``ParsedResponse`` TypedDict.

        Returns a ``ParsedResponse`` with:
          - ``thought``     — text after ``Thought:`` (empty string if missing)
          - ``predictions`` — one ``Action`` per ``Action:`` line

        Never raises — missing or malformed fields produce safe defaults.
        """
        lines = text.strip().split("\n")

        thought = ""
        for line in lines:
            if line.startswith("Thought:"):
                thought = line[len("Thought:"):].strip()
                break

        prefix = "Action: "
        action_lines = [line for line in lines if line.startswith(prefix)]

        predictions: List[Action] = []
        for line in action_lines:
            try:
                action_str = line[len(prefix):]
                parts = action_str.split(" ", 1)
                action = parts[0].strip().rstrip(";")   # "ANSWER;" → "ANSWER"
                args = None
                if len(parts) == 2:
                    args = [a.strip().strip("[]") for a in parts[1].strip().split(";")]
                predictions.append({"action": action, "args": args})
            except Exception as exc:
                predictions.append({
                    "action": "retry",
                    "args": [f"Could not parse action line '{line}': {exc}"],
                })

        if not predictions:
            predictions = [{"action": "retry", "args": [f"Could not parse LLM output: {text}"]}]

        return {"thought": thought, "predictions": predictions}

    # ── Browser action implementations ────────────────────────────────────────

    async def _do_click(self, state: AgentState, pred: Action) -> str:
        args = pred["args"]
        if args is None or len(args) != 1:
            return f"Failed to click: bad args {args}"
        bbox = state["bboxes"][int(args[0])]
        result = await self.navigator.click(int(bbox["x"]), int(bbox["y"]))
        return result.output or result.error

    async def _do_type(self, state: AgentState, pred: Action) -> str:
        args = pred["args"]
        if args is None or len(args) != 2:
            return f"Failed to type: bad args {args}"
        bbox = state["bboxes"][int(args[0])]
        result = await self.navigator.type_text(int(bbox["x"]), int(bbox["y"]), args[1])
        await self.navigator.key_press(None, None, "Enter")
        return result.output or result.error

    async def _do_scroll(self, state: AgentState, pred: Action) -> str:
        args = pred["args"]
        if args is None or len(args) != 2:
            return "Failed to scroll: bad args"
        target, direction = args
        amount = 500
        if target.upper() == "WINDOW":
            signed = -amount if direction.lower() == "up" else amount
            await self.navigator.page.evaluate(f"window.scrollBy(0, {signed})")
            return f"Scrolled window {direction}"
        bbox = state["bboxes"][int(target)]
        result = await self.navigator.scroll(
            int(bbox["x"]), int(bbox["y"]), direction.lower(), amount
        )
        return result.output or result.error

    async def _do_wait(self, state: AgentState, pred: Action) -> str:
        await asyncio.sleep(5)
        return "Waited for 5s."

    async def _do_go_back(self, state: AgentState, pred: Action) -> str:
        await self.navigator.page.go_back()
        return f"Navigated back to {self.navigator.page.url}."

    async def _do_goto_url(self, state: AgentState, pred: Action) -> str:
        args = pred["args"]
        if args is None or len(args) != 1:
            return f"Failed to navigate: bad args {args}"
        result = await self.navigator.go_to(args[0].strip())
        return result.output or result.error

    async def _do_web_search(self, state: AgentState, pred: Action) -> str:
        result = await self.navigator.go_to("https://duckduckgo.com")
        return result.output or result.error

    async def _do_log(self, state: AgentState, pred: Action) -> str:
        args = pred["args"]
        if args is None or len(args) == 0:
            return "Note: (empty note)"
        note = " ".join(args).strip()
        return f"Note: {note}"

    def _build_tool_map(self) -> Dict:
        return {
            "Click": self._do_click,
            "Type": self._do_type,
            "Scroll": self._do_scroll,
            "Wait": self._do_wait,
            "GoBack": self._do_go_back,
            "GoToUrl": self._do_goto_url,
            "WebSearch": self._do_web_search,
            "Log": self._do_log,
        }

    # ── Graph nodes ───────────────────────────────────────────────────────────

    async def _agent_node(self, state: AgentState) -> AgentState:
        step = state.get("step", 0) + 1
        annotated = await self._annotate(state)
        with_descs = self._format_descriptions(annotated)
        raw_text = await self._call_vlm(with_descs)
        parsed = self._parse_response(raw_text)
        if annotated.get("img"):
            await self._emit_image(annotated["img"])
        await self._emit_info(f"Step {step}: {parsed['thought']}", type="thought")
        return {
            **annotated,
            "bbox_descriptions": with_descs["bbox_descriptions"],
            "thought": parsed["thought"],
            "prediction": parsed["predictions"],
        }

    async def _execute_tools_node(self, state: AgentState) -> AgentState:
        """Execute every action in state["prediction"] in order.

        Any exception from a handler is caught and returned as an error
        observation so the LLM can see what went wrong and self-correct.
        """
        observations = []
        new_logs: List[str] = []
        for pred in state["prediction"]:
            action = pred["action"]
            args = pred.get("args")
            if action.startswith("ANSWER"):
                continue
            handler = self._tool_map.get(action)
            if handler is None:
                msg = f"Unknown action '{action}' with args {args}"
                observations.append(
                    f"Error: {msg}. Check the action format and try again."
                )
                await self._emit_warning(msg, type="tool")
                continue
            try:
                obs = await handler(state, pred)
            except Exception as exc:
                obs = (
                    f"Error executing '{action}' with args {args}: {exc}. "
                    f"Check the action format and try again."
                )
                await self._emit_error(f"Action '{action}' failed: {exc}", type="tool")
            logger.debug("Action %s → %s", action, obs)
            observations.append(f"{action}: {obs}")
            # Extract log note directly from _do_log's "Note: ..." return value
            if action == "Log" and obs.startswith("Note: "):
                note = obs[len("Note: "):].strip()
                if note:
                    new_logs.append(note)

        return {
            **state,
            "logs": list(state.get("logs") or []) + new_logs,
            "observation": "\n".join(observations) if observations else "No actions executed",
        }

    def _update_scratchpad(self, state: AgentState) -> AgentState:
        step = state.get("step", 0) + 1
        old = state.get("scratchpad")
        txt = old[0].content if old else "Previous action observations:\n"
        thought = (state.get("thought") or "").strip()
        observation = (state.get("observation") or "").strip()
        entry = f"\n{step}."
        if thought:
            entry += f" Thought: {thought}"
        entry += f"\n   Observation: {observation or '(no output)'}"
        txt += entry
        return {**state, "step": step, "scratchpad": [SystemMessage(content=txt)]}

    # ── Graph ─────────────────────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(AgentState)

        g.add_node("agent", self._agent_node)
        g.add_edge(START, "agent")

        g.add_node("execute_tools", self._execute_tools_node)
        g.add_edge("execute_tools", "update_scratchpad")

        g.add_node("update_scratchpad", self._update_scratchpad)
        g.add_edge("update_scratchpad", "agent")

        def select_tool(state: AgentState) -> str:
            predictions = state["prediction"]
            if not predictions:
                return "agent"
            actions = [p["action"] for p in predictions]
            if any(a.startswith("ANSWER") for a in actions):
                return END
            if actions == ["retry"]:
                return "agent"
            return "execute_tools"

        g.add_conditional_edges("agent", select_tool)
        return g.compile(checkpointer=self._checkpointer)

    # ── Public interface ──────────────────────────────────────────────────────

    async def arun(
        self,
        question: str,
        start_url: Optional[str] = None,
        max_steps: int = 150,
    ) -> RunResult:
        """Run the agent to completion and return a :class:`RunResult`.

        On the step where the agent emits ``ANSWER``, the full thought, all
        actions from that step, and the answer text are captured and returned.
        """
        await self.start(start_url)
        answer: Optional[str] = None
        thoughts: List[str] = []
        actions: List[List[Action]] = []
        observations: List[str] = []
        await self._emit_info(f"Starting web navigation: {question}", type="status")
        try:
            async for event in self.graph.astream(
                self._initial_state(question),
                {**self._run_config, "recursion_limit": max_steps},
            ):
                if "execute_tools" in event:
                    obs = event["execute_tools"].get("observation", "")
                    if obs:
                        observations.append(obs)

                if "agent" in event:
                    state = event["agent"]
                    thought = state.get("thought", "")
                    predictions: List[Action] = state.get("prediction") or []
                    thoughts.append(thought)
                    actions.append(predictions)
                    for pred in predictions:
                        if pred.get("action", "").startswith("ANSWER"):
                            answer = (pred.get("args") or [None])[0]
        finally:
            await self.stop()

        logs = await self.collect_logs()
        await self._emit_info(
            f"Web navigation complete — {len(thoughts)} steps, answer: {answer}",
            type="status",
        )
        return RunResult(
            answer=answer,
            thoughts=thoughts,
            actions=actions,
            observations=observations,
            logs=logs,
        )

    async def stream(
        self,
        question: str,
        start_url: Optional[str] = None,
        max_steps: int = 150,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Async generator that yields one ``StreamEvent`` per graph node firing.

        Yields an ``AgentEvent`` after each agent turn (thought, prediction,
        annotated screenshot) and a ``ToolsEvent`` after each tool execution
        (observation, accumulated logs).  The browser is always closed in the
        ``finally`` block, even if the caller breaks out of the loop early.

        Usage::

            async for event in agent.stream(task):
                if event["type"] == "agent":
                    print(event["thought"], event["prediction"])
                elif event["type"] == "tools":
                    print(event["observation"], event["logs"])
        """
        await self.start(start_url)
        await self._emit_info(f"Starting web navigation: {question}", type="status")
        try:
            async for event in self.graph.astream(
                self._initial_state(question),
                {**self._run_config, "recursion_limit": max_steps},
            ):
                if "agent" in event:
                    state = event["agent"]
                    yield AgentEvent(
                        type="agent",
                        thought=state.get("thought", ""),
                        prediction=state.get("prediction") or [],
                        img=state.get("img", ""),
                    )

                if "execute_tools" in event:
                    tools_state = event["execute_tools"]
                    yield ToolsEvent(
                        type="tools",
                        observation=tools_state.get("observation", ""),
                        logs=list(tools_state.get("logs") or []),
                    )
        finally:
            await self.stop()

    async def collect_logs(self) -> List[str]:
        """Return the accumulated logs from the most recent run state."""
        if self._run_config is None:
            return []
        snapshot = await self.graph.aget_state(self._run_config)
        return list(snapshot.values.get("logs") or [])

    async def get_state_history(self) -> List[Any]:
        """Return the full state history of the last run in chronological order.

        Each entry is a ``StateSnapshot`` with:
          - ``.values``    — the full ``AgentState`` dict at that checkpoint
          - ``.next``      — tuple of node names scheduled to run next
          - ``.metadata``  — step number and source node name
        """
        if self._run_config is None:
            return []
        return [
            snapshot
            async for snapshot in self.graph.aget_state_history(self._run_config)
        ][::-1]  # reverse to chronological order (aget_state_history is newest-first)
