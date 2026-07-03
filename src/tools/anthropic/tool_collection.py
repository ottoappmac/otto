"""Collection classes for managing multiple tools."""
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, Field, ConfigDict
from datetime import timedelta
from anthropic.types.beta import BetaToolUnionParam, BetaToolParam
from mcp.types import ImageContent, TextContent, EmbeddedResource, BlobResourceContents
from mcp.client.session import ClientSession
from tools.anthropic.base import BaseAnthropicTool
from tools.schemas import ToolFailure, ToolResult, ImageToolResult
from tools.anthropic.mcps import MCPHelper

from langchain_mcp_adapters.client import MultiServerMCPClient


class ToolCollection(BaseModel):
    """A collection of anthropic-defined tools."""
    tools: Optional[List[BaseAnthropicTool]] = Field(default=[])
    tool_map: Optional[Dict[str, Any]] = Field(default=None)
    tools_params: Optional[List[BetaToolUnionParam]] = Field(default=None)
    mcps: Optional[MultiServerMCPClient] = Field(default=None)
    helper: Optional[MCPHelper] = Field(default=None, exclude=True)
    mcp_map: Optional[Dict[str, ClientSession]] = Field(default={})
    mcp_params: Optional[List[BetaToolParam]] = Field(default=[])
    all_params: List[BetaToolUnionParam | BetaToolParam]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    async def from_params(cls, tools: Optional[List[BaseAnthropicTool]] = [], mcps: Optional[MultiServerMCPClient] = None):
        tool_map = {}
        tools_params = []
        mcp_map = {}
        mcp_params = []
        all_params = []
        helper = None

        if not tools and not mcps:
            raise ValueError("tools or mcps must be set.")
        if tools:
            tool_map = {tool.to_params()["name"]: tool for tool in tools}
            tools_params = [tool.to_params() for tool in tools]
            all_params += tools_params

        if mcps:
            helper = MCPHelper(mcps)
            mcp_map, mcp_params = await helper.build_mcp_map_and_params()
            all_params += mcp_params
        return cls(
            tools=tools,
            tool_map=tool_map,
            tools_params=tools_params,
            mcps=mcps,
            helper=helper,
            mcp_map=mcp_map,
            mcp_params=mcp_params,
            all_params=all_params
        )

    def to_params(self) -> Sequence[BetaToolUnionParam]:
        return self.all_params

    def _get_mcp_result(self, content: EmbeddedResource | ImageContent | TextContent) -> ToolResult | ImageToolResult | None:
        str_res = ""
        if isinstance(content, EmbeddedResource):
            resource = content.resource
            if isinstance(resource, BlobResourceContents):
                str_res = resource.blob
            else:
                str_res = resource.text
            return ToolResult(output=str_res)
        elif isinstance(content, ImageContent):
            return ImageToolResult(base64_image=content.data)
        elif isinstance(content, TextContent):
            return ToolResult(output=content.text)
        return None

    async def run(self, *, name: str, tool_input: dict[str, Any]) -> List[ToolResult | ImageToolResult] | List[ToolFailure]:
        try:
            if mcp := self.mcp_map.get(name) if self.mcp_map else None:
                results = []
                res = await mcp.call_tool(name=name, arguments=tool_input, read_timeout_seconds=timedelta(100))
                for content in res.content:
                    tool_result = self._get_mcp_result(content)
                    if tool_result:
                        results.append(tool_result)
                return results

            if self.tool_map:
                tool = self.tool_map.get(name)
                tool_input["tool_name"] = name
                if not tool:
                    return [ToolFailure(error=f"Tool {name} is invalid", name=name)]

                res = await tool(**tool_input)
                if isinstance(res, ToolResult):
                    return [res]
                return res
        except Exception as e:
            return [ToolFailure(error=f"{e}", name=name)]

        return [ToolFailure(error=f"Invalid tool. Tool name: {name}, Tool input: {tool_input}", name=name)]
