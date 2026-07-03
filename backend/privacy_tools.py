"""LangChain tools that let the orchestrator agent inspect and control
the privacy lock at runtime.

Design principles
-----------------
* **Read-only by default.**  ``privacy_status`` never mutates state; the
  agent can always check the current lock without side-effects.
* **Explicit engage / disengage.**  The agent must call dedicated tools to
  change lock state — no single "toggle" so the intent in the transcript
  is unambiguous.
* **Audit-log aware.**  Every engage / disengage via a tool call is
  recorded in the append-only audit log exactly like a UI action, so
  agent-driven lock changes are fully traceable.
* **Provider check.**  The agent can dry-run a provider name against the
  current policy before recommending a model switch — avoids suggesting
  a cloud LLM to the user while the lock is engaged.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def build_privacy_tools() -> list:
    """Return the privacy-lock tool set for injection into the agent graph."""

    @tool
    def privacy_status() -> str:
        """Return the current privacy lock status.

        Reports whether the lock is engaged, which LLM providers are
        allowed, when the lock was last engaged, and whether the macOS
        kernel-level packet filter (pf) anchor is active.

        Use this before recommending a model switch or before attempting
        to engage / disengage the lock.
        """
        from backend.config import AppConfig
        from backend.privacy_lock import is_engaged, pf_status

        cfg = AppConfig.load()
        p = cfg.privacy
        engaged = is_engaged(cfg)

        lines = [
            f"Privacy lock: {'ENGAGED' if engaged else 'disengaged'}",
            f"Allowed providers: {', '.join(p.local_only_providers) if p.local_only_providers else '(none configured)'}",
        ]
        if engaged:
            lines.append(f"Engaged at: {p.engaged_at or 'unknown'}")
            lines.append(f"Audit token: {p.audit_token or '(none)'}")

        pf = pf_status(cfg)
        if pf.get("available"):
            rule_count = pf.get("rule_count", 0)
            has_block = pf.get("has_block_rule", False)
            lines.append(
                f"Kernel firewall (pf anchor '{p.pf_anchor}'): "
                f"{'active' if has_block else 'loaded but no block rule'}, "
                f"{rule_count} rule(s)"
            )
        else:
            lines.append(
                f"Kernel firewall: not active ({pf.get('reason', 'unknown reason')})"
            )

        return "\n".join(lines)

    @tool
    def engage_privacy_lock() -> str:
        """Engage the privacy lock.

        Once engaged, any attempt to use a cloud LLM provider (Anthropic,
        OpenAI, Cohere) will be refused and logged.  Only local providers
        (afm, mlx, omlx, exo) are permitted.

        This is idempotent — calling it when already engaged does not
        rotate the audit token or reset the engagement timestamp.

        Returns a confirmation with the audit token for this session.
        """
        from backend.config import AppConfig
        from backend.privacy_lock import engage

        cfg = AppConfig.load()
        result = engage(cfg)
        cfg.save()
        cfg.apply_to_environ()

        already = not result.get("rotated", True)
        token = result.get("audit_token", "")
        allowed = result.get("allowed_providers", [])
        status = "already engaged" if already else "engaged"
        return (
            f"Privacy lock {status}.\n"
            f"Audit token: {token}\n"
            f"Allowed providers: {', '.join(allowed)}\n"
            "Cloud LLMs (Anthropic, OpenAI) are now blocked for all new sessions."
        )

    @tool
    def disengage_privacy_lock() -> str:
        """Disengage the privacy lock.

        Once disengaged, cloud LLM providers are permitted again.  The
        prior engagement window is preserved in the audit log.

        The kernel-level packet filter (pf) anchor — if installed — is
        NOT automatically removed.  The user must run
        ``sudo pfctl -a otto.privacy -F all`` in Terminal to clear it.

        Returns a confirmation with the prior audit token.
        """
        from backend.config import AppConfig
        from backend.privacy_lock import disengage

        cfg = AppConfig.load()
        result = disengage(cfg)
        cfg.save()
        cfg.apply_to_environ()

        was = result.get("was_engaged", True)
        prior_token = result.get("audit_token", "")
        if not was:
            return "Privacy lock was already disengaged — no change made."
        return (
            f"Privacy lock disengaged.\n"
            f"Prior audit token: {prior_token}\n"
            "Cloud LLMs are now permitted.  Note: the kernel pf anchor (if "
            "installed) must be removed manually with: "
            "sudo pfctl -a otto.privacy -F all"
        )

    @tool
    def check_provider_allowed(provider: str) -> str:
        """Check whether a given LLM provider is allowed under the current privacy lock.

        Args:
            provider: The provider name to check (e.g. "anthropic", "mlx", "afm").

        Returns a plain-English verdict.  Safe to call at any time —
        never mutates state or writes to the audit log.
        """
        from backend.config import AppConfig
        from backend.privacy_lock import PrivacyLockActive, enforce_provider_allowed

        cfg = AppConfig.load()
        if not cfg.privacy.enabled:
            return (
                f"Privacy lock is not engaged — '{provider}' is allowed "
                "(all providers are currently permitted)."
            )
        try:
            enforce_provider_allowed(provider, cfg)
            return (
                f"'{provider}' is ALLOWED under the current privacy lock "
                f"(local-only providers: {', '.join(cfg.privacy.local_only_providers)})."
            )
        except PrivacyLockActive:
            allowed = ", ".join(cfg.privacy.local_only_providers)
            return (
                f"'{provider}' is BLOCKED — it sends data off-device and the "
                f"privacy lock is engaged.\n"
                f"Allowed providers: {allowed}\n"
                "To use this provider, disengage the lock first with "
                "disengage_privacy_lock()."
            )

    @tool
    def privacy_audit_log(limit: int = 20) -> str:
        """Return the most recent privacy audit log entries.

        Args:
            limit: Number of most-recent entries to return (1–100, default 20).

        Each entry records an engage, disengage, or refused-provider event
        with a UTC timestamp and the session audit token.
        """
        limit = max(1, min(limit, 100))
        from backend.privacy_lock import tail_audit

        events = tail_audit(limit)
        if not events:
            return "Privacy audit log is empty — no engage/disengage events recorded yet."

        lines = [f"Last {len(events)} privacy audit event(s) (newest first):"]
        for e in events:
            ts = e.get("ts", "?")
            event = e.get("event", "?")
            token = e.get("audit_token", "")[:8]
            detail = ""
            if event == "refuse_provider":
                detail = f" — blocked provider: {e.get('provider', '?')}"
            elif event == "engage" and e.get("rotated"):
                detail = " (new session)"
            elif event == "disengage" and not e.get("was_engaged"):
                detail = " (was already off)"
            lines.append(f"  {ts}  {event}{detail}  token=…{token}")

        return "\n".join(lines)

    return [
        privacy_status,
        engage_privacy_lock,
        disengage_privacy_lock,
        check_provider_allowed,
        privacy_audit_log,
    ]
