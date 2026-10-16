"""Pure decision function. No I/O, no LangGraph.

The middleware, the scheduler, and the tests all call :func:`decide`
so the rules live in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.action_policy.catalog import (
    INVOCATION_GRANTABLE,
    MACOS_SERVER_IDS,
    OUTBOUND_CATEGORIES,
    NO_AUTO_APPROVE,
    SESSION_GRANTABLE,
    Classification,
)

_RANK = {"allow": 0, "ask": 1, "deny": 2}

# Nobody is at the keyboard. A prompt here would hang or, worse, be
# auto-approved by an old driver. Ask becomes deny unless the user
# pre-authorised that category on the schedule or trigger.
UNATTENDED_SOURCES = frozenset({
    "schedule",
    "trigger",
    "claude-hook",
    "openclaw",
    "hook",
})


def is_unattended(trigger_source: str | None) -> bool:
    return (trigger_source or "") in UNATTENDED_SOURCES


def validate_allowlist(actions: list[str] | None) -> str | None:
    """Reject unknown categories and ``persistence`` (a run must not spawn more runs)."""
    from backend.action_policy.catalog import CATEGORIES
    if not actions:
        return None
    bad = [name for name in actions if name not in CATEGORIES or name == "persistence"]
    if bad:
        return (
            "These actions cannot be pre-authorised: "
            + ", ".join(bad)
            + ". persistence is never allowed unattended."
        )
    return None


def unattended_hitl_decisions(metadata: dict | None, allowlist: set[str] | frozenset[str]) -> list[dict]:
    """Decisions for a graph interrupt when nobody is present.

    One decision per action. A category that is not on the allowlist is
    rejected. Legacy ``execute`` interrupts (no risk stamp) count as shell.
    """
    metadata = metadata or {}
    actions = metadata.get("action_requests") or []
    count = len(actions) or 1
    if metadata.get("type") == "request_credential":
        return [{"answer": "cancelled"} for _ in range(count)]
    risk = metadata.get("risk") or {}
    category = str(risk.get("category") or "")
    if not category:
        names = [a.get("name") for a in actions if isinstance(a, dict)]
        if not names or all(name == "execute" for name in names):
            category = "shell"
    if category and category in allowlist and category != "persistence":
        return [{"type": "approve"} for _ in range(count)]
    reason = (
        f"{category or 'this action'} is not pre-authorised for unattended runs"
    )
    return [{"type": "reject", "message": reason} for _ in range(count)]


def stricter(left: str, right: str) -> str:
    return left if _RANK.get(left, 1) >= _RANK.get(right, 1) else right


@dataclass(frozen=True)
class ActionDecision:
    effect: str  # allow | ask | deny
    category: str | None
    labels: tuple[str, ...]
    grantable: bool
    no_auto_approve: bool
    reason: str

    @property
    def title(self) -> str:
        from backend.action_policy.catalog import CATEGORY_TITLES
        if self.category and self.category in CATEGORY_TITLES:
            return CATEGORY_TITLES[self.category]
        return "Run this action?"


def effective_mode(categories: dict[str, str], category: str, agent_name: str | None, overrides: dict[str, dict[str, str]]) -> str:
    """Base mode, tightened by a per-agent override. A looser override is ignored."""
    base = categories.get(category, "ask")
    if base not in _RANK:
        base = "ask"
    if not agent_name:
        return base
    override = (overrides.get(agent_name) or {}).get(category)
    if override not in _RANK:
        return base
    return stricter(base, override)


def decide(
    classification: Classification,
    *,
    categories: dict[str, str],
    macos_automation_enabled: bool,
    attended: bool,
    allowlist: frozenset[str] | set[str],
    session_grants: frozenset[str] | set[str],
    invocation_grants: frozenset[str] | set[str],
    tainted: bool,
    rate_count: int,
    rate_limit: int,
    agent_name: str | None = None,
    agent_overrides: dict[str, dict[str, str]] | None = None,
) -> ActionDecision:
    """Return what to do with one tool call.

    ``rate_limit`` of 0 means no cap. ``rate_count`` is how many times
    this session has already run the primary category.
    """
    labels = classification.labels
    overrides = agent_overrides or {}

    # Before the "reads are not gated" shortcut. Listing an inbox is a read,
    # and the switch has to stop it on a session that already has the tool.
    if (
        not macos_automation_enabled
        and classification.server_id in MACOS_SERVER_IDS
    ):
        return ActionDecision(
            "deny",
            classification.category or "desktop_control",
            labels,
            False,
            True,
            "macOS automation is turned off in Settings",
        )

    if classification.category is None and not classification.also:
        return ActionDecision("allow", None, labels, False, False, "not gated")

    involved = classification.categories or ("unclassified",)
    modes = [
        effective_mode(categories, name, agent_name, overrides) for name in involved
    ]
    mode = modes[0]
    for item in modes[1:]:
        mode = stricter(mode, item)
    # The category we show is the strictest one, so the card matches the rule.
    category = involved[0]
    for name, item in zip(involved, modes):
        if item == mode:
            category = name
            break

    danger = bool(labels)
    grantable = category in SESSION_GRANTABLE and not danger
    no_auto = danger or category not in SESSION_GRANTABLE or category in NO_AUTO_APPROVE

    if rate_limit > 0 and rate_count >= rate_limit:
        return ActionDecision(
            "deny", category, labels, False, True,
            f"session limit of {rate_limit} {category.replace('_', ' ')} actions reached",
        )

    if tainted and category in OUTBOUND_CATEGORIES and mode == "allow":
        mode = "ask"

    if mode == "deny":
        return ActionDecision("deny", category, labels, False, True, f"{category} is set to deny")

    if mode == "allow":
        return ActionDecision("allow", category, labels, False, no_auto, "allowed by policy")

    # mode == ask
    if not attended:
        if category == "persistence":
            return ActionDecision(
                "deny", category, labels, False, True,
                "unattended runs cannot create more unattended runs",
            )
        if category in allowlist and not danger:
            return ActionDecision(
                "allow", category, labels, False, True,
                "pre-authorised on this schedule or trigger",
            )
        if category in allowlist and danger:
            return ActionDecision(
                "deny", category, labels, False, True,
                "pre-authorisation does not cover this dangerous variant",
            )
        return ActionDecision(
            "deny", category, labels, False, True,
            "no one is present to approve this, and it is not pre-authorised",
        )

    if category in session_grants and category in SESSION_GRANTABLE and not danger:
        if not (tainted and category in OUTBOUND_CATEGORIES):
            return ActionDecision(
                "allow", category, labels, True, no_auto,
                "allowed for the rest of this session",
            )
    if category in invocation_grants and category in INVOCATION_GRANTABLE and not danger:
        return ActionDecision(
            "allow", category, labels, False, True,
            "allowed for the rest of this agent run",
        )

    return ActionDecision("ask", category, labels, grantable, no_auto, "needs approval")
