"""LangChain tool that lets the agent ask the user structured questions."""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel, Field, field_validator


def _coerce_options(value: Any) -> Optional[list[str]]:
    """Best-effort recovery when a model sends ``options`` as a string.

    Some local/small tool-calling models JSON-encode array-typed arguments as
    a string (e.g. ``'["A", "B"]'``) instead of emitting a real list, which
    fails strict pydantic list validation and burns a retry. Recover the
    common shapes here rather than rejecting and re-prompting the model.
    """
    if value is None or isinstance(value, list):
        return value
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return None
    try:
        parsed = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, list):
        return [str(o) for o in parsed]
    # Fallback for models that emit a delimited string instead of JSON.
    for sep in ("\n", ";", "|"):
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            if len(parts) > 1:
                return parts
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) > 1:
            return parts
    return [s]


class _AskUserInput(BaseModel):
    question: str = Field(description="The question to present to the user.")
    options: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional list of choices. When omitted the user types a "
            'free-text answer instead. When provided, append "Other…" as '
            "the last item unless the question is purely binary "
            "(e.g. Yes / No)."
        ),
    )
    allow_multiple: bool = Field(
        default=False,
        description="When True and options are provided, the user may select more than one option.",
    )

    @field_validator("options", mode="before")
    @classmethod
    def _validate_options(cls, v: Any) -> Any:
        return _coerce_options(v)


def build_ask_user_tools() -> list:
    """Build user-interaction tools for injection into the agent graph."""

    @tool(args_schema=_AskUserInput)
    def ask_user(
        question: str,
        options: list[str] | None = None,
        allow_multiple: bool = False,
    ) -> str:
        """Ask the user a question and wait for their answer.

        Supports both open-ended questions (free-text) and structured
        multiple-choice questions with predefined options.

        Use ``options`` whenever the user must choose between a known set of
        alternatives — this renders clickable buttons instead of a text box.
        Always include **"Other…"** as the final option when providing a list
        so the user can type a custom answer if none of the choices fit.

        Args:
            question: The question to present to the user.
            options: Optional list of choices. When omitted the user
                     types a free-text answer instead.  When provided,
                     append ``"Other…"`` as the last item unless the
                     question is purely binary (e.g. Yes / No).
            allow_multiple: When True and options are provided, the user
                            may select more than one option.

        Returns:
            The user's answer as a string (or comma-separated answers when
            allow_multiple is True and the user picks several options).
        """
        payload: dict = {"type": "ask_user", "question": question}
        if options:
            payload["options"] = options
            payload["allow_multiple"] = allow_multiple

        result = interrupt(payload)

        if isinstance(result, dict):
            decisions = result.get("decisions") or []
            if decisions and isinstance(decisions[0], dict):
                d = decisions[0]
                answers = d.get("answers") or []
                if answers:
                    return ", ".join(str(a) for a in answers)
                answer = d.get("answer")
                if answer is not None:
                    return str(answer)
        return str(result)

    return [ask_user]
