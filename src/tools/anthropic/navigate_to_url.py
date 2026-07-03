from typing import ClassVar
from typing_extensions import Literal
from anthropic.types.beta.beta_tool_param import BetaToolParam

from tools.schemas import ToolResult
from tools.base import Navigator
from tools.anthropic.base import BaseAnthropicTool


class NavigateToUrlTool(BaseAnthropicTool):
    """
    A tool that allows the agent to navigate to a URL.
    """
    name: ClassVar[Literal["navigate_to_url"]] = "navigate_to_url"
    api_type: ClassVar[Literal["custom"]] = "custom"

    def __init__(self, navigator: Navigator):
        self.navigator = navigator
        super().__init__()

    async def __call__(self, *, url: str, **kwargs) -> ToolResult:
        res = await self.navigator.go_to(url)
        return res

    def to_params(self) -> BetaToolParam:
        return {
            "name": self.name,
            "input_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The web url."
                    }
                },
                "required": ["url"],
                "additionalProperties": False,
                "description": (
                    "Schema for a Navigate to URL Tool. The tool allows you to "
                    "navigate to a url on the browser."
                )
            }
        }
