"""Built-in stdio MCPs must still be able to launch from under Application Support/Otto."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from backend import mcp_sandbox


def test_server_dir_reopened_after_otto_deny(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    server_dir = home / "Library" / "Application Support" / "Otto" / "mcp_server" / "edgar-sec"
    server_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(mcp_sandbox.sys, "platform", "darwin")
    monkeypatch.setattr(mcp_sandbox, "is_supported", lambda: True)
    monkeypatch.setattr(mcp_sandbox, "resolve_sandbox_exec", lambda: "/usr/bin/sandbox-exec")

    mcp_sandbox.wrap_builtin_stdio(["python", "server.py"], server_id="edgar-sec", work_dir=server_dir)

    profile = (server_dir / ".edgar-sec.sb").read_text()
    real = str(server_dir.resolve())
    otto_deny = profile.index("Application Support/Otto\"))")
    # A later rule wins, so the reopen must come after the Otto-wide deny.
    assert profile.rindex(f'(allow file-read* (subpath "{real}"))') > otto_deny
    assert profile.rindex(f'(allow file-write* (subpath "{real}"))') > otto_deny


def _load_edgar():
    path = Path(__file__).resolve().parents[1] / "backend" / "builtin_mcps" / "edgar_sec" / "server.py"
    spec = importlib.util.spec_from_file_location("edgar_sec_server", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_company_facts_summary_keeps_core_and_recent(monkeypatch):
    pytest.importorskip("mcp.server.fastmcp")
    edgar = _load_edgar()
    facts = {}
    # 250 retired tags sorting alphabetically before the core ones.
    for i in range(250):
        facts[f"A{i:03d}Old"] = {"label": "old", "units": {"USD": [{"end": "2016-01-01", "val": 1}]}}
    facts["Revenues"] = {"label": "Revenue", "units": {"USD": [{"end": "2026-07-26", "val": 5}]}}
    facts["ZRecentOther"] = {"label": "new", "units": {"USD": [{"end": "2026-07-26", "val": 2}]}}

    class Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"entityName": "X", "facts": {"us-gaap": facts}}

    class Client:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, *a, **k): return Resp()

    monkeypatch.setattr(edgar, "_client", lambda **k: Client())
    monkeypatch.setattr(edgar, "_normalize_cik", lambda v: "0000000001")
    out = edgar.get_company_facts("X")
    keys = list(out["concepts_summary"])
    assert len(keys) == 200
    assert keys[0] == "Revenues"
    assert keys[1] == "ZRecentOther"
    assert out["concepts_count"] == 252
