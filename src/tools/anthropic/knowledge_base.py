from typing import ClassVar
from typing_extensions import Literal
from anthropic.types.beta.beta_tool_param import BetaToolParam

from tools.schemas import ToolResult
from tools.base import Navigator
from tools.anthropic.base import BaseAnthropicTool


class KnowledgeBaseTool(BaseAnthropicTool):
    """
    A tool that allows the agent to query a knowledge base.
    """
    name: ClassVar[Literal["knowledge_base"]] = "knowledge_base"
    api_type: ClassVar[Literal["custom"]] = "custom"

    def __init__(self, navigator: Navigator):
        self.navigator = navigator
        super().__init__()

    async def __call__(self, *, query: str, **kwargs) -> ToolResult:
        res = await self.navigator.ask_knowledge_base(query)
        return res

    def to_params(self) -> BetaToolParam:
        return {
            "name": self.name,
            "input_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query."
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
                "description": (
                    "Schema for a Knowledge Base Query Tool. The tool allows you to "
                    "do a query search on a knowledge base."
                )
            }
        }
