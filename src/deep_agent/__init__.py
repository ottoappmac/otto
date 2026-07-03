"""DeepAgent — enum-driven orchestrator with callback support.

Quick start::

    from deep_agent import DeepAgent, ToolOption, SubAgentOption

    agent = DeepAgent(
        tools=[ToolOption.DOC_READER, ToolOption.WEB_RESEARCHER],
        subagents=[SubAgentOption.WEB_VOYAGER],
    )
    answer = await agent.arun("Summarise the Python 3.12 release notes.")
"""

from deep_agent.agent import DeepAgent, print_message
from deep_agent.options import SubAgentOption, ToolOption

__all__ = [
    "DeepAgent",
    "ToolOption",
    "SubAgentOption",
    "print_message",
]
