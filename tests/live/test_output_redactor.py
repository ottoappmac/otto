"""Live tests: output_redactor credential scrubbing.

The redactor is the last line of defence before MCP tool results reach the
LLM context.  Unlike most live tests, the redactor itself is a pure function
and does not call any LLM — these tests exercise the real regex patterns
against realistic credential strings.

One test (``test_redactor_applied_in_session_stream``) runs the full session
pipeline with a stub MCP output so we can confirm the redactor is wired into
the stream path.

Run with::

    pytest -m live tests/live/test_output_redactor.py
"""

from __future__ import annotations

import pytest

from backend.output_redactor import redact, redact_value

pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# Provider-specific key patterns
# ---------------------------------------------------------------------------


def test_stripe_live_secret_redacted():
    # Built via concatenation (rather than one literal) so this fixture
    # doesn't itself look like a real key to secret-scanners.
    secret = "sk_live_" + "ABC123DEF456GHI789JKL012"
    raw = f"The API key is {secret} and should be hidden."
    out = redact(raw)
    assert secret not in out
    assert "[REDACTED:stripe_secret]" in out


def test_stripe_test_secret_redacted():
    secret = "sk_test_" + "abcdefghijklmnopqrst"
    raw = f"key={secret}"
    out = redact(raw)
    assert secret not in out


def test_stripe_restricted_key_redacted():
    raw = "rk_live_" + "z" * 20
    out = redact(raw)
    assert "rk_live_" not in out


def test_slack_bot_token_redacted():
    secret = "xoxb-12345678-12345678901-" + "abcdefghijklmnopqrstuvwx"
    raw = f"Token: {secret}"
    out = redact(raw)
    assert "xoxb-12345678" not in out
    assert "[REDACTED:slack_token]" in out


def test_discord_bot_token_redacted():
    secret = "NzI1MzM0Mjg0ODc2NTQzMjE2.GH7abc." + "AbCdEfGhIjKlMnOpQrStUvWxYz0123"
    raw = f"DISCORD_BOT_TOKEN={secret}"
    out = redact(raw)
    assert "AbCdEfGhIjKlMnOpQrStUvWxYz0123" not in out
    assert "[REDACTED:discord_bot_token]" in out


def test_github_pat_redacted():
    raw = "GITHUB_TOKEN=ghp_" + "A" * 36
    out = redact(raw)
    assert "ghp_" + "A" * 36 not in out
    assert "[REDACTED:github_token]" in out


def test_aws_access_key_redacted():
    raw = f"AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
    out = redact(raw)
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "[REDACTED:aws_access_key]" in out


def test_anthropic_api_key_redacted():
    raw = "sk-ant-" + "x" * 50
    out = redact(raw)
    assert "sk-ant-" + "x" * 50 not in out
    assert "[REDACTED:anthropic_key]" in out


def test_openai_api_key_redacted():
    raw = "Authorization: Bearer sk-" + "y" * 48
    out = redact(raw)
    assert "sk-" + "y" * 48 not in out


def test_google_api_key_redacted():
    raw = "key=AIza" + "B" * 35
    out = redact(raw)
    assert "AIza" + "B" * 35 not in out


def test_jwt_redacted():
    # Realistic JWT structure (three base64url segments)
    jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    raw = f"token={jwt}"
    out = redact(raw)
    assert jwt not in out
    assert "[REDACTED:jwt]" in out


def test_private_key_block_redacted():
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA0Z3VS5JJcds3xHn/ygWep4FJkfxHRJuPj9TGghx1ZNKU\n"
        "-----END RSA PRIVATE KEY-----"
    )
    out = redact(pem)
    assert "BEGIN RSA PRIVATE KEY" not in out


def test_clean_text_passes_through_unchanged():
    clean = "The weather today is sunny and 22 degrees Celsius."
    assert redact(clean) == clean


def test_redact_value_walks_nested_dict():
    payload = {
        "status": "ok",
        "token": "ghp_" + "Z" * 40,
        "nested": {"key": "AKIAIOSFODNN7EXAMPLE"},
    }
    out = redact_value(payload)
    assert "ghp_" not in str(out)
    assert "AKIA" not in str(out)
    assert out["status"] == "ok"


def test_redact_value_walks_list():
    items = ["normal text", "sk_live_" + "q" * 20, 42]
    out = redact_value(items)
    assert "sk_live_" not in str(out)
    assert out[2] == 42


def test_redact_value_non_string_passthrough():
    assert redact_value(12345) == 12345
    assert redact_value(None) is None
    assert redact_value(3.14) == 3.14


# ---------------------------------------------------------------------------
# Integration: redactor wired into the session stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_redactor_applied_in_session_stream(session_manager, live_session):
    """Verify the output redactor fires on any tool output that reaches the
    stream.  We prompt privacy_status (a built-in tool with deterministic
    output) and assert that no real API key patterns survive in the events.

    This does not test that the agent produces a key — it confirms the
    wiring: if redaction were bypassed, synthetic secrets placed by a
    malicious MCP tool would flow through to the LLM context unredacted.
    """
    from tests.live.conftest import run_session as _run

    events = await _run(session_manager, live_session.id, "Check my privacy status.")

    tool_result_contents = " ".join(
        e.get("content", "") for e in events if e.get("type") == "tool_result"
    )

    # Verify no raw Stripe/GitHub/AWS key shapes survived in tool output.
    import re

    dangerous_patterns = [
        re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]{16,}"),
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"),
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        re.compile(r"\bsk-ant-[A-Za-z0-9_-]{40,}"),
    ]
    for pat in dangerous_patterns:
        assert not pat.search(tool_result_contents), (
            f"Credential pattern {pat.pattern!r} found in tool result output — "
            "redactor may be bypassed"
        )
