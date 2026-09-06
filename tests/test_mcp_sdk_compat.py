"""Guard the Python MCP SDK against a 2.x upgrade that breaks Playwright MCP.

``mcp`` 2.x removed ``RequestContext`` from ``mcp.shared.context``.
``langchain-mcp-adapters`` 0.2.x still imports that name at module load, so
connecting Playwright MCP (and every other MCP client) fails with::

    cannot import name 'RequestContext' from 'mcp.shared.context'

The packaged app hit this because ``uv pip install -e .`` is unfrozen and
resolves ``mcp>=1.0.0`` to 2.x.  Pin stays ``mcp>=1.0.0,<2`` until the
client migrates to ``langchain.mcp``.
"""

from __future__ import annotations

from importlib.metadata import version

from backend.mcp_builder import MCPSpec, render_requirements_txt


def test_mcp_sdk_major_is_below_2():
    major = int(version("mcp").split(".", 1)[0])
    assert major < 2, (
        f"mcp {version('mcp')} is incompatible with langchain-mcp-adapters; "
        "pin mcp>=1.0.0,<2"
    )


def test_request_context_still_exported():
    from mcp.shared.context import RequestContext

    assert RequestContext is not None


def test_langchain_mcp_adapters_callbacks_import():
    """This is the import that crashed OTTO.app against mcp 2.x."""
    import langchain_mcp_adapters.callbacks  # noqa: F401
    from langchain_mcp_adapters.client import MultiServerMCPClient

    assert MultiServerMCPClient is not None


def test_fastmcp_import_path_still_exists():
    from mcp.server.fastmcp import FastMCP

    assert FastMCP is not None


def test_generated_mcp_requirements_pin_below_v2():
    spec = MCPSpec(id="compat", name="Compat", description="pin check")
    text = render_requirements_txt(spec)
    assert "mcp>=1.0.0,<2" in text
