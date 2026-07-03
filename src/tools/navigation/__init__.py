"""Navigation tools — web and desktop automation.

``PlaywrightComputerUseNavigator`` is exposed via a lazy ``__getattr__``
so that simply touching this package (e.g. ``from tools.navigation.computer
import MacOSNavigator``) does not transitively import the ``playwright``
Python package.  The bundled backend ships without ``playwright``
(browser automation goes through the external ``@playwright/mcp`` Node
process), so an eager import here would crash any code path that only
needs the desktop / MCP-client tools.  Loading lazily keeps the bundle
slim while preserving the public re-export for callers that genuinely
need the navigator class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tools.navigation.computer.base import ComputerNavigator
from tools.navigation.computer.macos_navigator import MacOSNavigator
from tools.navigation.computer.macos_tools import (
    CONTROL_INTERACTION_RULES,
    MACOS_TOOLS,
    MacOSToolkit,
)

if TYPE_CHECKING:
    from tools.navigation.web.playwright_navigator import (
        PlaywrightComputerUseNavigator,
    )

__all__ = [
    "PlaywrightComputerUseNavigator",
    "ComputerNavigator",
    "MacOSNavigator",
    "MacOSToolkit",
    "MACOS_TOOLS",
    "CONTROL_INTERACTION_RULES",
]


def __getattr__(name: str) -> Any:
    if name == "PlaywrightComputerUseNavigator":
        from tools.navigation.web.playwright_navigator import (
            PlaywrightComputerUseNavigator as _Nav,
        )
        return _Nav
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
