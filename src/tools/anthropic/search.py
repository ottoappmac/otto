from typing import ClassVar
from typing_extensions import Literal
from anthropic.types.beta.beta_tool_param import BetaToolParam

from tools.schemas import ToolResult
from tools.base import Navigator
from tools.anthropic.base import BaseAnthropicTool


class SearchTool(BaseAnthropicTool):
    """
    A tool that allows the agent to do a web search.
    """
    name: ClassVar[Literal["search"]] = "search"
    api_type: ClassVar[Literal["custom"]] = "custom"

    def __init__(self, navigator: Navigator):
        self.navigator = navigator
        super().__init__()

    async def __call__(self, *, search_text: str, **kwargs) -> ToolResult:
        res = await self.navigator.search(search_text)
        return res

    def to_params(self) -> BetaToolParam:
        return {
            "name": self.name,
            "input_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "search_text": {
                        "type": "string",
                        "description": "The search text."
                    }
                },
                "required": ["search_text"],
                "additionalProperties": False,
                "description": (
                    "Schema for a Web Search Tool. The tool allows you to "
                    "do a web search on the browser."
                )
            }
        }
