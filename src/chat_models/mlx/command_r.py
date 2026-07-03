"""Custom chat model for Cohere Command R MLX models without chat_template.

mlx-community/c4ai-command-r-v01-2bit and similar models have tokenizers that
don't include chat_template in tokenizer_config.json. This wrapper formats
messages using the Command R special tokens directly, bypassing apply_chat_template.

Format (from tokenizer special tokens):
    <|START_OF_TURN_TOKEN|><|USER_TOKEN|>
    {content}
    <|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
    {assistant response}
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Type, Union

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from langchain_community.llms.mlx_pipeline import MLXPipeline

# Command R special tokens (from tokenizer_config.json)
START_OF_TURN = "<|START_OF_TURN_TOKEN|>"
END_OF_TURN = "<|END_OF_TURN_TOKEN|>"
USER_TOKEN = "<|USER_TOKEN|>"
CHATBOT_TOKEN = "<|CHATBOT_TOKEN|>"
SYSTEM_TOKEN = "<|SYSTEM_TOKEN|>"


def _messages_to_command_r_prompt(messages: List[BaseMessage]) -> str:
    """Convert LangChain messages to Command R prompt format."""
    if not messages:
        raise ValueError("At least one message must be provided")
    if not isinstance(messages[-1], HumanMessage):
        raise ValueError("Last message must be a HumanMessage")

    parts: List[str] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            parts.append(
                f"{START_OF_TURN}{SYSTEM_TOKEN}\n{msg.content}{END_OF_TURN}"
            )
        elif isinstance(msg, HumanMessage):
            parts.append(
                f"{START_OF_TURN}{USER_TOKEN}\n{msg.content}{END_OF_TURN}"
            )
        elif isinstance(msg, AIMessage):
            parts.append(
                f"{START_OF_TURN}{CHATBOT_TOKEN}\n{msg.content}{END_OF_TURN}"
            )
        else:
            raise ValueError(f"Unsupported message type: {type(msg)}")

    # Add generation prompt: we want the model to generate the next assistant turn
    parts.append(f"{START_OF_TURN}{CHATBOT_TOKEN}\n")
    return "".join(parts)


class CommandRMLXChat(BaseChatModel):
    """Chat model for Cohere Command R MLX models that lack tokenizer chat_template."""

    llm: MLXPipeline

    def __init__(self, llm: MLXPipeline, **kwargs: Any):
        super().__init__(llm=llm, **kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = _messages_to_command_r_prompt(messages)
        llm_result = self.llm._generate(
            prompts=[prompt],
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
        return self._to_chat_result(llm_result)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = _messages_to_command_r_prompt(messages)
        llm_result = await self.llm._agenerate(
            prompts=[prompt],
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
        return self._to_chat_result(llm_result)

    @staticmethod
    def _to_chat_result(llm_result: LLMResult) -> ChatResult:
        generations = []
        for g in llm_result.generations[0]:
            generations.append(
                ChatGeneration(
                    message=AIMessage(content=g.text),
                    generation_info=g.generation_info,
                )
            )
        return ChatResult(generations=generations, llm_output=llm_result.llm_output)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, Literal["auto", "none"], bool]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool definitions for create_agent compatibility."""
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None and tool_choice:
            if len(formatted_tools) != 1:
                raise ValueError(
                    "When specifying `tool_choice`, you must provide exactly one "
                    f"tool. Received {len(formatted_tools)} tools."
                )
            if isinstance(tool_choice, str):
                if tool_choice not in ("auto", "none"):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
            elif isinstance(tool_choice, bool):
                tool_choice = formatted_tools[0]
            elif isinstance(tool_choice, dict):
                if (
                    formatted_tools[0]["function"]["name"]
                    != tool_choice["function"]["name"]
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tool was {formatted_tools[0]['function']['name']}."
                    )
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {type(tool_choice)}"
                )
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "command-r-mlx-chat"
