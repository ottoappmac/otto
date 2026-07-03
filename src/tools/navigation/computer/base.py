"""Abstract base class for OS-level desktop navigation.

``ComputerNavigator`` defines the contract that platform-specific
implementations (macOS, Windows, Linux) must satisfy.  The
``ComputerVoyagerGraph`` agent programs against this interface so
it works with any concrete navigator without code changes.

Subclass checklist
------------------
1. Implement every ``@abstractmethod``.
2. ``get_tools()`` must return a list of LangChain ``BaseTool`` instances
   that ``create_react_agent`` / ``MLXReActWrapper`` can bind.
3. ``get_system_instructions()`` should return OS-specific prompts
   (e.g. control-interaction rules, app-launch protocols).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.tools import BaseTool


class ComputerNavigator(ABC):
    """Platform-agnostic interface for desktop automation agents."""

    @abstractmethod
    def get_tools(self, *, vision: bool = False) -> list[BaseTool]:
        """Return the LangChain tools for this platform.

        Args:
            vision: When ``True``, include vision-only tools (e.g. pixel
                screenshot) that require a VLM to interpret.
        """
        ...

    @abstractmethod
    def get_system_instructions(self, *, vision: bool = False) -> str:
        """Return the system prompt fragment with platform-specific rules.

        This is appended to the agent's system message and should include
        tool-usage rules, interaction protocols, and any OS-specific caveats.

        Args:
            vision: When ``True``, include guidance for vision tools
                (e.g. ``capture_app_screenshot``).
        """
        ...

    @abstractmethod
    def get_control_interaction_rules(self) -> str:
        """Return the control/role → valid-tool mapping for the agent prompt."""
        ...
