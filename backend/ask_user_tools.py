"""LangChain tool that lets the agent ask the user structured questions."""

from __future__ import annotations

from langchain_core.tools import tool
from langgraph.types import interrupt


def build_ask_user_tools() -> list:
    """Build user-interaction tools for injection into the agent graph."""

    @tool
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
