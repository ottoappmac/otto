"""Action policy: catalog, decisions, config, unattended resumes, sandbox."""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _mcp_tool_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for deco in node.decorator_list:
            if _decorator_name(deco) != "tool":
                continue
            explicit = None
            if isinstance(deco, ast.Call):
                for keyword in deco.keywords:
                    if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                        explicit = str(keyword.value.value)
            names.add(explicit or node.name)
    return names


def _macos_native_names() -> set[str]:
    path = ROOT / "src/tools/navigation/computer/macos_tools.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for deco in node.decorator_list:
            if not isinstance(deco, ast.Call) or _decorator_name(deco) != "tool":
                continue
            if deco.args and isinstance(deco.args[0], ast.Constant):
                names.add(str(deco.args[0].value))
    return names


def test_builtin_mcp_tools_are_classified():
    from backend.action_policy.catalog import server_tools
    from backend.builtin_mcps.registry import BUILTIN_MCPS

    known = server_tools()
    missing: list[str] = []
    for mcp in BUILTIN_MCPS:
        server_py = ROOT / "backend/builtin_mcps" / mcp.source_dir_name / "server.py"
        if not server_py.is_file():
            continue
        for name in _mcp_tool_names(server_py):
            if (mcp.id, name) not in known:
                missing.append(f"{mcp.id}.{name}")
    assert missing == [], "unclassified built-in tools: " + ", ".join(missing)


def test_macos_native_tools_are_classified():
    from backend.action_policy.catalog import server_tools

    known = {name for (server, name) in server_tools() if server == "macos-native"}
    missing = sorted(_macos_native_names() - known)
    assert missing == [], "unclassified macos-native tools: " + ", ".join(missing)


def _decide(name, args=None, *, server_id=None, attended=True, allowlist=(), grants=(), tainted=False, categories=None, macos=True, rate_count=0, rate_limit=0, agent_name=None, overrides=None):
    from backend.action_policy.catalog import balanced_categories, classify_tool
    from backend.action_policy.gate import decide

    classification = classify_tool(name, server_id=server_id, args=args or {})
    return decide(
        classification,
        categories=categories or balanced_categories(),
        macos_automation_enabled=macos,
        attended=attended,
        allowlist=frozenset(allowlist),
        session_grants=frozenset(grants),
        invocation_grants=frozenset(),
        tainted=tainted,
        rate_count=rate_count,
        rate_limit=rate_limit,
        agent_name=agent_name,
        agent_overrides=overrides,
    )


def test_shell_asks_when_attended_and_denies_when_not():
    asked = _decide("execute", {"command": "echo hi"})
    assert asked.effect == "ask"
    assert asked.category == "shell"
    denied = _decide("execute", {"command": "echo hi"}, attended=False)
    assert denied.effect == "deny"
    allowed = _decide("execute", {"command": "echo hi"}, attended=False, allowlist=["shell"])
    assert allowed.effect == "allow"


def test_dangerous_shell_is_not_covered_by_an_allowlist():
    decision = _decide(
        "execute",
        {"command": "osascript -e 'return 1'"},
        attended=False,
        allowlist=["shell"],
    )
    assert decision.effect == "deny"
    assert decision.labels


def test_session_grant_covers_shell_but_not_email():
    granted = _decide("execute", {"command": "echo hi"}, grants=["shell"])
    assert granted.effect == "allow"
    mail = _decide("send_message", {"to": "a@b.c", "subject": "hi"}, server_id="macos-mail", grants=["email_send"])
    assert mail.effect == "ask"
    assert mail.grantable is False


def test_taint_upgrades_an_outbound_allow_and_ignores_grants():
    from backend.action_policy.catalog import balanced_categories

    cats = balanced_categories()
    cats["message_send"] = "allow"
    upgraded = _decide(
        "send_message",
        {"text": "hi"},
        server_id="macos-messages",
        categories=cats,
        tainted=True,
        grants=["message_send"],
    )
    assert upgraded.effect == "ask"


def test_rate_limit_denies_after_the_cap():
    decision = _decide(
        "send_message",
        {"to": "a@b.c"},
        server_id="macos-mail",
        rate_count=5,
        rate_limit=5,
    )
    assert decision.effect == "deny"


def test_agent_override_can_only_tighten():
    looser = _decide(
        "execute",
        {"command": "echo hi"},
        agent_name="helper",
        overrides={"helper": {"shell": "allow"}},
    )
    assert looser.effect == "ask"
    tighter = _decide(
        "execute",
        {"command": "echo hi"},
        agent_name="helper",
        overrides={"helper": {"shell": "deny"}},
    )
    assert tighter.effect == "deny"


def test_kill_switch_denies_macos_servers():
    decision = _decide(
        "send_message",
        {"text": "hi"},
        server_id="macos-messages",
        macos=False,
    )
    assert decision.effect == "deny"


def test_persistence_is_never_allowlisted():
    decision = _decide("create_schedule", {"id": "x"}, attended=False, allowlist=["persistence"])
    assert decision.effect == "deny"


def test_unattended_decisions_match_action_count():
    from backend.action_policy.gate import unattended_hitl_decisions

    rejected = unattended_hitl_decisions(
        {"action_requests": [{}, {}], "risk": {"category": "shell"}},
        set(),
    )
    assert rejected == [
        {"type": "reject", "message": "shell is not pre-authorised for unattended runs"},
        {"type": "reject", "message": "shell is not pre-authorised for unattended runs"},
    ]
    approved = unattended_hitl_decisions(
        {"action_requests": [{}, {}], "risk": {"category": "shell"}},
        {"shell"},
    )
    assert approved == [{"type": "approve"}, {"type": "approve"}]
    cancelled = unattended_hitl_decisions(
        {"type": "request_credential", "action_requests": [{}]},
        {"shell"},
    )
    assert cancelled == [{"answer": "cancelled"}]


def test_allowlist_rejects_persistence_and_unknown_names():
    from backend.action_policy.gate import validate_allowlist

    assert validate_allowlist(["shell"]) is None
    assert validate_allowlist(["persistence"]) is not None
    assert validate_allowlist(["not-a-category"]) is not None


def test_existing_config_gains_action_policy_defaults():
    from backend.config import AppConfig

    cfg = AppConfig.model_validate({"auto_approve_commands": True})
    assert cfg.auto_approve_commands is True
    assert cfg.action_policy.preset == "balanced"
    assert cfg.action_policy.categories["shell"] == "ask"
    assert cfg.action_policy.categories["mail_read"] == "ask"
    assert cfg.action_policy.categories["message_read"] == "ask"
    assert cfg.action_policy.categories["browser_automation"] == "allow"
    assert cfg.action_policy.execute_sandbox.enabled is True
    assert cfg.action_policy.execute_sandbox.network is False
    assert cfg.action_policy.macos_automation_enabled is True


def test_invalid_category_mode_falls_back_to_the_default():
    from backend.config import ActionPolicyConfig

    policy = ActionPolicyConfig.model_validate({"categories": {"shell": "yolo", "email_send": "deny"}})
    assert policy.categories["shell"] == "ask"
    assert policy.categories["email_send"] == "deny"


def test_settings_tools_cannot_enable_macos_servers_while_the_switch_is_off(monkeypatch):
    from backend.config import AppConfig
    from backend.settings_tools import build_settings_tools

    saved = {"called": False}

    class _Server:
        id = "macos-mail"
        enabled = False

    class _Policy:
        macos_automation_enabled = False

    class _Cfg:
        action_policy = _Policy()
        mcp_servers = [_Server()]

        def save(self):
            saved["called"] = True

    monkeypatch.setattr(AppConfig, "load", staticmethod(lambda: _Cfg()))
    toggle = next(tool for tool in build_settings_tools() if tool.name == "toggle_mcp_server")
    result = toggle.invoke({"server_id": "macos-mail", "enabled": True})
    assert "cannot be enabled" in result
    assert saved["called"] is False


def test_execute_is_not_double_gated_by_interrupt_on():
    source = (ROOT / "backend/session_manager.py").read_text(encoding="utf-8")
    assert "interrupt_on=None" in source
    assert 'interrupt_on={"execute": True}' not in source


@pytest.mark.asyncio
async def test_middleware_ask_then_approve_runs_the_tool(monkeypatch):
    from backend.action_policy.middleware import ActionPolicyMiddleware
    from backend.action_policy.runtime import RunScope, begin_invocation, end_invocation, reset_scope, resolve, set_scope
    from backend.config import AppConfig

    monkeypatch.setattr(AppConfig, "load", staticmethod(lambda: AppConfig()))
    middleware = ActionPolicyMiddleware(agent_name="orchestrator")
    scope_token = set_scope(RunScope(session_id="s-test", attended=True, allowlist=frozenset()))
    invocation_token = begin_invocation()
    called = {"n": 0}

    async def handler(_request):
        called["n"] += 1
        return SimpleNamespace(content="ok")

    request = SimpleNamespace(
        tool_call={"name": "execute", "id": "call-1", "args": {"command": "echo hi"}},
        tool=SimpleNamespace(metadata={}),
    )

    async def answer():
        import asyncio
        from backend.action_policy.runtime import has_pending
        for _ in range(50):
            pending = __import__("backend.action_policy.runtime", fromlist=["_pending"])._pending.get("s-test", {})
            if pending:
                request_id = next(iter(pending))
                assert has_pending("s-test", request_id)
                resolve("s-test", request_id, [{"type": "approve"}])
                return
            await asyncio.sleep(0.02)
        raise AssertionError("approval was never published")

    import asyncio
    task = asyncio.create_task(answer())
    try:
        result = await middleware.awrap_tool_call(request, handler)
    finally:
        end_invocation(invocation_token)
        reset_scope(scope_token)
        task.cancel()
    assert called["n"] == 1
    assert result.content == "ok"


def _macos_side_effects():
    """Every macOS tool the catalog treats as a side effect, not a read."""
    from backend.action_policy.catalog import MACOS_SERVER_IDS, server_tools

    return [
        (server, name, category)
        for (server, name), category in server_tools().items()
        if server in MACOS_SERVER_IDS and category is not None
    ]


def _tool_request(server_id, name, args, call_id="call-1"):
    return SimpleNamespace(
        tool_call={"name": name, "id": call_id, "args": args},
        tool=SimpleNamespace(metadata={"otto_server_id": server_id}),
    )


async def _await_card(session_id):
    import asyncio
    from backend.action_policy.runtime import pop_approvals

    for _ in range(50):
        cards = pop_approvals(session_id)
        if cards:
            return cards[0]
        await asyncio.sleep(0.02)
    raise AssertionError(f"no approval card for {session_id}")


@pytest.mark.asyncio
async def test_kill_switch_never_calls_macos_side_effect_tools(monkeypatch):
    """The tool body stands in for Mail, Messages, AppleScript, and clicks.

    With the switch off, none of those functions run, including reading
    the inbox. The tool stays on an already-open session until the next
    call, and that call is refused.
    """
    import asyncio
    from backend.action_policy.middleware import ActionPolicyMiddleware
    from backend.action_policy.runtime import RunScope, begin_invocation, clear_session, end_invocation, reset_scope, set_scope
    from backend.config import AppConfig

    cfg = AppConfig()
    cfg.action_policy.macos_automation_enabled = False
    monkeypatch.setattr(AppConfig, "load", staticmethod(lambda: cfg))
    monkeypatch.setattr("backend.action_policy.audit.record", lambda **kwargs: None)

    effects = _macos_side_effects()
    assert len(effects) >= 20
    middleware = ActionPolicyMiddleware(agent_name="orchestrator")
    session_id = "s-macos-off"
    scope_token = set_scope(RunScope(session_id=session_id, attended=True, allowlist=frozenset()))
    invocation_token = begin_invocation()
    called = []

    async def handler(request):
        called.append(request.tool_call["name"])
        return SimpleNamespace(content="should-not-run")

    try:
        for server, name, _category in effects:
            result = await asyncio.wait_for(
                middleware.awrap_tool_call(_tool_request(server, name, {}, call_id=name), handler),
                timeout=1,
            )
            assert "Blocked by action policy" in result.content
            assert "macOS automation is turned off" in result.content
        assert called == []

        inbox = await middleware.awrap_tool_call(
            _tool_request(
                "macos-mail", "list_messages",
                {"limit": 20, "mailbox_name": "inbox", "unread_only": True},
                call_id="inbox",
            ),
            handler,
        )
        assert "Blocked by action policy" in inbox.content
        assert "macOS automation is turned off" in inbox.content
        assert called == []
    finally:
        end_invocation(invocation_token)
        reset_scope(scope_token)
        clear_session(session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("server", "name", "args", "category", "title", "snippet"),
    [
        ("macos-mail", "list_messages", {"limit": 20, "mailbox_name": "inbox", "unread_only": True}, "mail_read", "Read email?", "inbox"),
        ("macos-messages", "read_messages", {"limit": 10}, "message_read", "Read messages?", "limit"),
        ("macos-mail", "send_message", {"to": "a@b.c", "subject": "Hello", "content": "secret body"}, "email_send", "Send email?", "To: a@b.c"),
        ("macos-messages", "send_message", {"buddy": "+15551212", "text": "ping"}, "message_send", "Send message?", "+15551212"),
        ("macos-osascript", "run_osascript", {"script": 'tell application "Finder" to activate'}, "applescript", "Run AppleScript?", "Finder"),
        ("macos-native", "click", {"x": 12, "y": 34}, "desktop_control", "Control this Mac?", "click"),
    ],
)
async def test_macos_ask_card_blocks_until_reject(monkeypatch, server, name, args, category, title, snippet):
    import asyncio
    from backend.action_policy.middleware import ActionPolicyMiddleware
    from backend.action_policy.runtime import RunScope, begin_invocation, clear_session, end_invocation, reset_scope, resolve, set_scope
    from backend.config import AppConfig

    monkeypatch.setattr(AppConfig, "load", staticmethod(lambda: AppConfig()))
    monkeypatch.setattr("backend.action_policy.audit.record", lambda **kwargs: None)
    middleware = ActionPolicyMiddleware(agent_name="orchestrator")
    session_id = f"s-ask-{server}-{name}"
    scope_token = set_scope(RunScope(session_id=session_id, attended=True, allowlist=frozenset()))
    invocation_token = begin_invocation()
    called = {"n": 0}

    async def handler(_request):
        called["n"] += 1
        return SimpleNamespace(content="ran")

    async def reject_card():
        card = await _await_card(session_id)
        meta = card["metadata"]
        assert card["content"] == title
        assert meta["type"] == "action_policy"
        assert meta["risk"]["category"] == category
        assert meta["risk"]["no_auto_approve"] is True
        assert meta["risk"]["grantable"] is (category in ("mail_read", "message_read"))
        assert meta["review_configs"][0]["allowed_decisions"] == ["approve", "reject"]
        description = meta["action_requests"][0]["description"]
        assert snippet in description
        resolve(session_id, meta["request_id"], [{"type": "reject", "message": "no"}])

    task = asyncio.create_task(reject_card())
    try:
        result = await asyncio.wait_for(
            middleware.awrap_tool_call(_tool_request(server, name, args), handler),
            timeout=2,
        )
        await task
    finally:
        end_invocation(invocation_token)
        reset_scope(scope_token)
        clear_session(session_id)
        task.cancel()
    assert called["n"] == 0
    assert "Not run" in result.content


@pytest.mark.asyncio
async def test_macos_mail_runs_only_after_approval_and_grant_does_not_stick(monkeypatch):
    import asyncio
    from backend.action_policy.middleware import ActionPolicyMiddleware
    from backend.action_policy.runtime import (
        RunScope, begin_invocation, clear_session, end_invocation, reset_scope, resolve, session_grants, set_scope,
    )
    from backend.config import AppConfig

    monkeypatch.setattr(AppConfig, "load", staticmethod(lambda: AppConfig()))
    monkeypatch.setattr("backend.action_policy.audit.record", lambda **kwargs: None)
    middleware = ActionPolicyMiddleware(agent_name="orchestrator")
    session_id = "s-mail-grant"
    scope_token = set_scope(RunScope(session_id=session_id, attended=True, allowlist=frozenset()))
    invocation_token = begin_invocation()
    called = {"n": 0}

    async def handler(_request):
        called["n"] += 1
        return SimpleNamespace(content="sent")

    async def approve(grant):
        card = await _await_card(session_id)
        decision = {"type": "approve"}
        if grant:
            decision["grant_category"] = "email_send"
        resolve(session_id, card["metadata"]["request_id"], [decision])

    request = _tool_request(
        "macos-mail", "send_message",
        {"to": "a@b.c", "subject": "Hello", "content": "body"},
    )
    first = asyncio.create_task(approve(True))
    try:
        result = await asyncio.wait_for(middleware.awrap_tool_call(request, handler), timeout=2)
        await first
        assert result.content == "sent"
        assert called["n"] == 1
        assert "email_send" not in session_grants(session_id)

        second = asyncio.create_task(approve(False))
        result = await asyncio.wait_for(
            middleware.awrap_tool_call(request, handler, ),
            timeout=2,
        )
        await second
        assert called["n"] == 2
        assert result.content == "sent"
    finally:
        end_invocation(invocation_token)
        reset_scope(scope_token)
        clear_session(session_id)


@pytest.mark.asyncio
async def test_desktop_approval_covers_the_rest_of_the_run_only(monkeypatch):
    import asyncio
    from backend.action_policy.middleware import ActionPolicyMiddleware
    from backend.action_policy.runtime import (
        RunScope, begin_invocation, clear_session, end_invocation, reset_scope, resolve, set_scope,
    )
    from backend.config import AppConfig

    monkeypatch.setattr(AppConfig, "load", staticmethod(lambda: AppConfig()))
    monkeypatch.setattr("backend.action_policy.audit.record", lambda **kwargs: None)
    middleware = ActionPolicyMiddleware(agent_name="macos-desktop-agent")
    session_id = "s-desktop"
    scope_token = set_scope(RunScope(session_id=session_id, attended=True, allowlist=frozenset()))
    invocation_token = begin_invocation()
    called = {"n": 0}

    async def handler(_request):
        called["n"] += 1
        return SimpleNamespace(content="clicked")

    async def approve_once():
        card = await _await_card(session_id)
        assert card["metadata"]["risk"]["category"] == "desktop_control"
        resolve(session_id, card["metadata"]["request_id"], [{"type": "approve"}])

    click = _tool_request("macos-native", "click", {"x": 1, "y": 2})
    task = asyncio.create_task(approve_once())
    try:
        await asyncio.wait_for(middleware.awrap_tool_call(click, handler), timeout=2)
        await task
        # Same agent run: the next click must not wait for another card.
        await asyncio.wait_for(middleware.awrap_tool_call(click, handler), timeout=1)
        assert called["n"] == 2
    finally:
        end_invocation(invocation_token)
        reset_scope(scope_token)
        clear_session(session_id)
        task.cancel()


@pytest.mark.asyncio
async def test_unattended_macos_send_does_not_run_or_ask(monkeypatch):
    import asyncio
    from backend.action_policy.middleware import ActionPolicyMiddleware
    from backend.action_policy.runtime import RunScope, begin_invocation, clear_session, end_invocation, pop_approvals, reset_scope, set_scope
    from backend.config import AppConfig

    monkeypatch.setattr(AppConfig, "load", staticmethod(lambda: AppConfig()))
    monkeypatch.setattr("backend.action_policy.audit.record", lambda **kwargs: None)
    middleware = ActionPolicyMiddleware(agent_name="orchestrator")
    session_id = "s-unattended-mail"
    scope_token = set_scope(RunScope(session_id=session_id, attended=False, allowlist=frozenset()))
    invocation_token = begin_invocation()
    called = {"n": 0}

    async def handler(_request):
        called["n"] += 1
        return SimpleNamespace(content="sent")

    try:
        result = await asyncio.wait_for(
            middleware.awrap_tool_call(
                _tool_request("macos-mail", "send_message", {"to": "a@b.c", "subject": "Hi", "content": "x"}),
                handler,
            ),
            timeout=1,
        )
        assert called["n"] == 0
        assert "Blocked by action policy" in result.content
        assert pop_approvals(session_id) == []

        reset_scope(scope_token)
        scope_token = set_scope(RunScope(
            session_id=session_id, attended=False, allowlist=frozenset({"email_send"}),
        ))
        allowed = await asyncio.wait_for(
            middleware.awrap_tool_call(
                _tool_request("macos-mail", "send_message", {"to": "a@b.c", "subject": "Hi", "content": "x"}),
                handler,
            ),
            timeout=1,
        )
        assert allowed.content == "sent"
        assert called["n"] == 1
    finally:
        end_invocation(invocation_token)
        reset_scope(scope_token)
        clear_session(session_id)


def test_kill_switch_drops_macos_servers_at_attach_time():
    from types import SimpleNamespace as NS
    from backend.action_policy.catalog import MACOS_SERVER_IDS, drop_macos_servers

    servers = [NS(id=server_id) for server_id in sorted(MACOS_SERVER_IDS)]
    servers.append(NS(id="slack"))
    kept = drop_macos_servers(servers, macos_automation_enabled=False)
    assert [server.id for server in kept] == ["slack"]
    assert drop_macos_servers(servers, macos_automation_enabled=True) == list(servers)


def test_settings_tools_refuse_every_macos_server_while_the_switch_is_off(monkeypatch):
    from backend.action_policy.catalog import MACOS_SERVER_IDS
    from backend.config import AppConfig
    from backend.settings_tools import build_settings_tools

    class _Policy:
        macos_automation_enabled = False

    class _Cfg:
        action_policy = _Policy()
        mcp_servers = []

        def save(self):
            raise AssertionError("policy switch must not be saved open by the agent")

    cfg = _Cfg()
    monkeypatch.setattr(AppConfig, "load", staticmethod(lambda: cfg))
    toggle = next(tool for tool in build_settings_tools() if tool.name == "toggle_mcp_server")
    for server_id in sorted(MACOS_SERVER_IDS):
        cfg.mcp_servers = [SimpleNamespace(id=server_id, enabled=False)]
        result = toggle.invoke({"server_id": server_id, "enabled": True})
        assert "cannot be enabled" in result


def test_scrub_env_drops_secrets_and_the_api_token():
    from backend.exec_sandbox import scrub_env

    cleaned = scrub_env({
        "PATH": "/usr/bin",
        "HOME": "/Users/me",
        "OPENAI_API_KEY": "sk-test",
        "OTTO_API_TOKEN": "launch-token",
        "AWS_SECRET_ACCESS_KEY": "aws",
        "LANG": "en_US.UTF-8",
    })
    assert cleaned["PATH"] == "/usr/bin"
    assert "OPENAI_API_KEY" not in cleaned
    assert "OTTO_API_TOKEN" not in cleaned
    assert "AWS_SECRET_ACCESS_KEY" not in cleaned
    assert cleaned["LANG"] == "en_US.UTF-8"


@pytest.mark.skipif(sys.platform != "darwin" or not os.path.exists("/usr/bin/sandbox-exec"), reason="Seatbelt is macOS-only")
def test_sandbox_blocks_writes_secrets_osascript_and_timeouts(tmp_path):
    from backend.exec_sandbox import run_shell, scrub_env

    inside = tmp_path / "inside"
    inside.mkdir()
    # pytest's tmp dir lives under /var/folders, which the profile allows
    # as a temp directory. The denied write has to be outside that.
    outside = Path.home() / ".otto-action-policy-sandbox-test"
    env = scrub_env({
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(Path.home()),
        "USER": os.environ.get("USER", "user"),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "LANG": "en_US.UTF-8",
        "OPENAI_API_KEY": "sk-should-not-appear",
    })
    assert "OPENAI_API_KEY" not in env

    written = run_shell(
        f"echo ok > {inside / 'out.txt'}",
        cwd=inside,
        env=env,
        timeout=15,
        sandbox=True,
        network=False,
        write_paths=[inside],
        profile_dir=inside,
    )
    assert (inside / "out.txt").read_text(encoding="utf-8").strip() == "ok", written.stderr or written.error

    try:
        denied = run_shell(
            f"mkdir -p {outside} && echo nope > {outside / 'nope.txt'}",
            cwd=inside,
            env=env,
            timeout=15,
            sandbox=True,
            network=False,
            write_paths=[inside],
            profile_dir=inside,
        )
        assert not (outside / "nope.txt").exists(), denied.stdout
    finally:
        if outside.exists():
            import shutil
            shutil.rmtree(outside, ignore_errors=True)

    secret = run_shell(
        'cat "$HOME/Library/Application Support/Otto/config.json"',
        cwd=inside,
        env=env,
        timeout=15,
        sandbox=True,
        network=False,
        write_paths=[inside],
        profile_dir=inside,
    )
    assert secret.returncode != 0

    ssh = run_shell(
        'ls "$HOME/.ssh"',
        cwd=inside,
        env=env,
        timeout=15,
        sandbox=True,
        network=False,
        write_paths=[inside],
        profile_dir=inside,
    )
    assert ssh.returncode != 0

    osa = run_shell(
        "osascript -e 'return 1'",
        cwd=inside,
        env=env,
        timeout=15,
        sandbox=True,
        network=False,
        write_paths=[inside],
        profile_dir=inside,
    )
    assert osa.returncode != 0

    leaked = run_shell(
        "printf %s \"$OPENAI_API_KEY\"",
        cwd=inside,
        env=env,
        timeout=15,
        sandbox=True,
        network=False,
        write_paths=[inside],
        profile_dir=inside,
    )
    assert "sk-should-not-appear" not in leaked.stdout

    timed = run_shell(
        "sleep 30",
        cwd=inside,
        env=env,
        timeout=1,
        sandbox=True,
        network=False,
        write_paths=[inside],
        profile_dir=inside,
    )
    assert timed.returncode != 0
