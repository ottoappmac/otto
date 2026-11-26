"""LinkedIn built-in MCP: registration, launcher settings, and tool classification."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "backend" / "builtin_mcps" / "linkedin"

# Every tool mcp-server-linkedin 4.30.0 lists with its remote catalog off.
# A version bump that adds or renames a tool must update this and the catalog.
EXPECTED_TOOLS = frozenset({
    "archive_conversation", "close_session", "connect_with_person", "dismiss_job",
    "get_company_employees", "get_company_posts", "get_company_profile",
    "get_conversation", "get_easy_apply_form", "get_feed", "get_inbox",
    "get_job_apply_url", "get_job_details", "get_my_profile", "get_person_profile",
    "get_saved_jobs", "get_session_status", "get_sidebar_profiles",
    "reply_to_conversation", "search_companies", "search_conversations",
    "search_jobs", "search_people", "search_posts", "send_message",
})


def _load_launcher():
    spec = importlib.util.spec_from_file_location("linkedin_launcher", SRC / "server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mcp():
    from backend.builtin_mcps.registry import BUILTIN_MCPS
    return next(m for m in BUILTIN_MCPS if m.id == "linkedin")


def test_registered_disabled_with_pinned_hash_locked_package():
    mcp = _mcp()
    assert mcp.enabled is False
    assert mcp.runtime == "python"
    req = (SRC / "requirements.txt").read_text()
    assert "mcp-server-linkedin==4.30.0" in req
    # Every pinned requirement carries hashes, so uv verifies the downloads.
    pins = [ln for ln in req.splitlines() if ln and not ln.startswith((" ", "#"))]
    assert pins and all("==" in ln for ln in pins)
    assert req.count("--hash=sha256:") >= len(pins)


def test_launcher_forces_local_only_settings(tmp_path, monkeypatch):
    launcher = _load_launcher()
    monkeypatch.setattr(launcher, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(launcher, "HOME_DIR", tmp_path / "home")
    env = {"HOME": "/Users/someone", "LINKEDIN_MCP_REMOTE_TOOLS": "on"}

    argv = launcher.prepare(env)

    assert env["LINKEDIN_MCP_REMOTE_TOOLS"] == "off"
    assert env["LINKEDIN_MCP_CHECK_FOR_UPDATES"] == "off"
    assert env["HOME"] == str(tmp_path / "home")
    assert "--no-auto-import" in argv and "--no-daemon" in argv
    profile = Path(argv[argv.index("--user-data-dir") + 1])
    # The package deletes the profile's parent on logout; it must be ours alone.
    assert profile.parent.name == "state" and profile.parent.parent == SRC


def test_every_tool_is_classified_and_writes_are_gated():
    from backend.action_policy.catalog import KNOWN_SERVER_IDS, classify_tool, server_tools

    assert "linkedin" in KNOWN_SERVER_IDS
    classified = {name for (server, name) in server_tools() if server == "linkedin"}
    assert classified == EXPECTED_TOOLS

    def cat(name):
        return classify_tool(name, server_id="linkedin").category

    for name in ("send_message", "reply_to_conversation", "connect_with_person"):
        assert cat(name) == "message_send", name
    for name in ("get_inbox", "get_conversation", "search_conversations"):
        assert cat(name) == "message_read", name
    assert cat("archive_conversation") == "destructive"
    assert cat("dismiss_job") == "destructive"
    assert cat("get_easy_apply_form") == "browser_automation"
    for name in ("search_jobs", "get_job_details", "get_person_profile"):
        assert cat(name) is None, name


def test_not_seatbelt_wrapped():
    from backend.action_policy.catalog import UNSANDBOXED_SERVER_IDS

    assert "linkedin" in UNSANDBOXED_SERVER_IDS
