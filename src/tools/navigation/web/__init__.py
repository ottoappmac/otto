"""Web navigation tools — Playwright-based browser automation.

``PlaywrightComputerUseNavigator`` is exposed via a lazy ``__getattr__``
because it pulls in the ``playwright`` Python package at module load.
The bundled backend ships without ``playwright`` (the MCP client wrapper
``create_playwright_mcp_client`` talks to an external Node process and
needs only HTTP), so importing the navigator eagerly here would crash
every loader that just wants the MCP client.  The lazy hook preserves
``from tools.navigation.web import PlaywrightComputerUseNavigator`` for
the dev environment where ``playwright`` is installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tools.navigation.web.playwright_mcp import (
    PlaywrightMCP,
    create_playwright_mcp_client,
)

if TYPE_CHECKING:
    from tools.navigation.web.playwright_navigator import (
        PlaywrightComputerUseNavigator,
    )

__all__ = [
    "PlaywrightComputerUseNavigator",
    "PlaywrightMCP",
    "create_playwright_mcp_client",
]


def __getattr__(name: str) -> Any:
    if name == "PlaywrightComputerUseNavigator":
        from tools.navigation.web.playwright_navigator import (
            PlaywrightComputerUseNavigator as _Nav,
        )
        return _Nav
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
