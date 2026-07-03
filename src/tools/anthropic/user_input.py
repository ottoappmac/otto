from typing import ClassVar
from typing_extensions import Literal
from anthropic.types.beta.beta_tool_param import BetaToolParam

from tools.schemas import ToolResult
from tools.anthropic.base import BaseAnthropicTool
from callbacks.base import WebCallbackHandler


class UserInputTool(BaseAnthropicTool):
    """
    A tool that allows the agent to request input from the human user.
    """
    name: ClassVar[Literal["user_input"]] = "user_input"
    api_type: ClassVar[Literal["custom"]] = "custom"

    def __init__(self, callback: WebCallbackHandler):
        self.callback = callback
        super().__init__()

    async def __call__(self, *, question: str, **kwargs) -> ToolResult:
        output = ""
        error = ""
        try:
            user_input = await self.callback.on_user_input(question)
            if user_input:
                output = f"Successfully obtained User input - Question: {question}, User input: {user_input}"
            else:
                error = "Error - Could not request User for input"
        except Exception as e:
            error = f"Error - Could not request User for input: {e}"
        return ToolResult(output=output, error=error)

    def to_params(self) -> BetaToolParam:
        return {
            "name": self.name,
            "input_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Your question."
                    }
                },
                "required": ["question"],
                "additionalProperties": False,
                "description": (
                    "Schema for a User Input Tool. The tool allows you to "
                    "ask the User for input based on a question."
                )
            }
        }
