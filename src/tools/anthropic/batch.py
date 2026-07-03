from typing import ClassVar, Any, List, Tuple
from typing_extensions import Literal
import json
from logging import getLogger

from anthropic.types.beta.beta_tool_param import BetaToolParam
from tools.anthropic.base import BaseAnthropicTool
from tools.schemas import ToolFailure, ToolResult, ImageToolResult

logger = getLogger()

BATCH_EXECUTION_DESCRIPTION = str(
    "# Invoke multiple other tool calls simultaneously\n "
    "* Used to ensure efficiency by executing multiple tool calls in one cycle. \n"
)

ListToolResult = List[ToolResult | ImageToolResult | ToolFailure | None]

BatchExecutionReturnType = Tuple[ListToolResult, ToolResult] | List[ToolResult | ImageToolResult | ToolFailure | None] | None


class BatchExecutionTool(BaseAnthropicTool):
    """
    A tool that allows the agent to execute multiple tool calls in a single interaction.
    Claude 3.7 has a strong inbuilt limitation where it executes a single tool per cycle.
    This tool works around that to execute multiple tool calls at once to increase efficiency.
    """
    name: ClassVar[Literal["batch"]] = "batch"
    api_type: ClassVar[Literal["custom"]] = "custom"

    def __init__(self, tool: BaseAnthropicTool):
        super().__init__()
        self.tool = tool

    async def _process_tool_call(self, tool_name: str, tool_args: Any):
        try:
            return await self.tool(**tool_args)
        except Exception as e:
            return ToolFailure(error=str(e), name=tool_name)

    async def __call__(self, *, invocations, **kwargs) -> BatchExecutionReturnType:
        tool_name = ""
        if kwargs["tool_name"]:
            tool_name = kwargs["tool_name"]
        if tool_name == "batch":
            results: ListToolResult = []
            for invocation in invocations:
                tool_call_result = await self._process_tool_call(
                        invocation["name"],
                        invocation["arguments"]
                    )
                results.append(tool_call_result)

            batch_tool_result = ToolResult(
                output=f"Batch tool executed the following invocations {json.dumps(invocations)}\n\n" +
                       "\n".join([result.output for result in results if result and hasattr(result, 'output') and result.output]),
            )
            return results, batch_tool_result
        else:
            single_tool_call = await self._process_tool_call(
                tool_name=tool_name,
                tool_args=kwargs
            )
            return [single_tool_call]

    def to_params(self) -> BetaToolParam:
        return {
            "name": self.name,
            "description": BATCH_EXECUTION_DESCRIPTION,
            "input_schema": {
                "type": "object",
                "properties": {
                    "invocations": {
                        "type": "array",
                        "properties": {
                            "name": {
                                "types": "string",
                                "description": "The name of the tool to invoke"
                            },
                            "arguments": {
                                "types": "object",
                                "description": "The arguments to the tool"
                            }
                        },
                        "required": ["name", "arguments"]
                    }
                },
                "required": ["invocations"]
            }
        }
