"""Last-line-of-defence credential scrubber.

Even with our other controls (env-var injection at spawn, generated-code
auditing, no vault read endpoint), an MCP tool's response could still
contain a credential by accident — e.g. a ``stripe_sdk_error.user_message``
that echoes the API key, or a Slack ``files.upload`` whose response
includes the bot token in a debug field.

This module runs every MCP tool result through a regex pass before the
result reaches the LLM context.  Patterns are ordered by specificity:
provider-specific tokens first (so ``sk_live_…`` is recognised as a
Stripe key, not a generic JWT), then format-shape patterns last.

It is intentionally conservative — false positives are vastly preferable
to leaks.  When in doubt, redact.

Add new patterns by editing :data:`PATTERNS` below.  Each pattern's
``(name, regex)`` pair is exposed in the redaction marker so an operator
can grep audit logs for which class of token tripped the filter.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# Compiled once at import time.  Order matters: more-specific patterns
# come first so we identify the *kind* of token in the redacted output.
PATTERNS: list[tuple[str, re.Pattern]] = [
    # Stripe — sk_live_*, sk_test_*, rk_live_*, rk_test_* (restricted), pk_*
    ("stripe_secret", re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("stripe_restricted", re.compile(r"\brk_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("stripe_publishable", re.compile(r"\bpk_(?:live|test)_[A-Za-z0-9]{16,}")),

    # Slack — xoxp- (user), xoxb- (bot), xoxa- (app), xoxr- (refresh)
    ("slack_token", re.compile(r"\bxox[pbar]-[\d]+-[\d]+(?:-[\d]+)?-[a-zA-Z0-9]+")),

    # GitHub — ghp_ (PAT), ghs_ (server), gho_ (OAuth), ghu_ (user-server)
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}")),

    # AWS access key + likely-paired secret
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("aws_secret_key", re.compile(
        r"(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])"
        r"(?=.{0,200}(?:secret|aws|key))",
        re.IGNORECASE | re.DOTALL,
    )),

    # Google — API keys (AIza...) and OAuth client secrets (GOCSPX-)
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("google_oauth_secret", re.compile(r"\bGOCSPX-[A-Za-z0-9_-]{28,}\b")),

    # Anthropic / OpenAI — sk-ant-…, sk-…
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{40,}")),
    ("openai_key", re.compile(r"\bsk-(?!ant-)[A-Za-z0-9]{20,}")),

    # Generic Bearer headers (catches OAuth bearer tokens of any shape)
    ("bearer_header", re.compile(r"(?i)(?:authorization:\s*bearer\s+)([A-Za-z0-9._\-]+)")),

    # JWT — three base64url segments separated by dots
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b")),

    # Discord bot token — three base64url segments (id.timestamp.hmac).
    # Not a standard JWT (no "eyJ" header), so it needs its own pattern.
    # Placed after "jwt" since it's a less-specific three-segment shape.
    ("discord_bot_token", re.compile(
        r"\b[A-Za-z0-9_-]{24,28}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,}\b"
    )),

    # Private keys / SSH keys — header markers
    ("private_key_block", re.compile(
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----[\s\S]{0,4096}?"
        r"-----END (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----"
    )),
]


def redact(text: str) -> str:
    """Return *text* with all known credential shapes blanked out.

    Each redaction is replaced with a marker like ``[REDACTED:slack_token]``
    so callers downstream can see *what kind* was filtered without seeing
    the value.  Counts are logged so an operator can spot tools that
    consistently leak.
    """
    if not text:
        return text
    counts: dict[str, int] = {}
    for name, pat in PATTERNS:
        def _sub(_match: re.Match, _n=name) -> str:
            counts[_n] = counts.get(_n, 0) + 1
            return f"[REDACTED:{_n}]"
        text = pat.sub(_sub, text)
    if counts:
        logger.warning("output_redactor: redacted %s", counts)
    return text


def redact_value(value: Any) -> Any:
    """Recursive variant for arbitrary JSON-shaped values.

    Strings get :func:`redact`; lists/dicts are walked; other types pass
    through unchanged.  Depth is unbounded — MCP responses are bounded
    by tool output size limits upstream, so a malicious server can't
    OOM us by handing back a 1M-deep tree without us already failing
    elsewhere first.
    """
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_value(v) for v in value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    return value


__all__ = ["redact", "redact_value", "PATTERNS"]
