"""Gate every tool call on the orchestrator and on sub-agents.

Sub-agent approvals cannot use LangGraph ``interrupt()``. A child graph
invoked with ``ainvoke`` swallows the interrupt, so the parent continues
as if the tool finished. This middleware asks through
:func:`backend.action_policy.runtime.request_approval` instead, which
the websocket resolves without resuming the graph.

The orchestrator's ``execute`` tool is gated here too. ``interrupt_on``
is left unset for it, so there is one card rather than two, and a
sub-agent ``execute`` is covered by the same code.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from backend.action_policy.catalog import (
    INGEST_TOOLS,
    INVOCATION_GRANTABLE,
    describe_action,
    classify_tool,
    locked_down_categories,
)
from backend.action_policy.gate import ActionDecision, decide
from backend.action_policy.runtime import (
    bump_count,
    current_count,
    current_scope,
    grant_invocation,
    grant_session,
    invocation_grants,
    is_tainted,
    mark_tainted,
    request_approval,
    session_grants,
)

logger = logging.getLogger(__name__)


def _policy_view() -> tuple[dict[str, str], bool, dict[str, dict[str, str]], dict[str, int]]:
    """Categories, kill switch, overrides, rate limits. Unreadable settings fail closed."""
    try:
        from backend.config import AppConfig
        policy = AppConfig.load().action_policy
        return (
            dict(policy.categories),
            bool(policy.macos_automation_enabled),
            {k: dict(v) for k, v in policy.agent_overrides.items()},
            dict(policy.rate_limit_per_session),
        )
    except Exception:
        logger.exception("action policy: could not read settings; denying gated tools")
        return locked_down_categories(), False, {}, {}


def _server_id(request: ToolCallRequest) -> str | None:
    tool = request.tool
    if tool is None:
        return None
    meta = getattr(tool, "metadata", None) or {}
    value = meta.get("otto_server_id")
    return str(value) if value else None


def _blocked(request: ToolCallRequest, text: str) -> ToolMessage:
    return ToolMessage(
        content=text,
        name=request.tool_call.get("name") or "tool",
        tool_call_id=request.tool_call.get("id") or "",
        status="error",
    )


class ActionPolicyMiddleware(AgentMiddleware):
    """Apply :func:`backend.action_policy.gate.decide` before a tool runs."""

    def __init__(self, *, agent_name: str | None = None) -> None:
        self.agent_name = agent_name

    def _decide(self, request: ToolCallRequest) -> tuple[ActionDecision, str, dict, str | None]:
        name = request.tool_call.get("name") or ""
        args = request.tool_call.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        server = _server_id(request)
        classification = classify_tool(name, server_id=server, args=args)
        categories, macos_on, overrides, limits = _policy_view()
        scope = current_scope()
        session_id = scope.session_id if scope else ""
        attended = True if scope is None else scope.attended
        allowlist = frozenset() if scope is None else scope.allowlist
        category = classification.category
        decision = decide(
            classification,
            categories=categories,
            macos_automation_enabled=macos_on,
            attended=attended,
            allowlist=allowlist,
            session_grants=session_grants(session_id) if session_id else frozenset(),
            invocation_grants=invocation_grants(),
            tainted=is_tainted(session_id) if session_id else False,
            rate_count=current_count(session_id, category) if category else 0,
            rate_limit=int(limits.get(category, 0)) if category else 0,
            agent_name=self.agent_name,
            agent_overrides=overrides,
        )
        return decision, name, args, server

    def _audit(self, decision: ActionDecision, name: str, server: str | None, args: dict, effect: str) -> None:
        from backend.action_policy.audit import record
        scope = current_scope()
        record(
            session_id=scope.session_id if scope else None,
            agent=self.agent_name,
            tool=name,
            server_id=server,
            category=decision.category,
            effect=effect,
            reason=decision.reason,
            attended=True if scope is None else scope.attended,
            args=args,
            labels=decision.labels,
        )

    def _note_ingest(self, name: str) -> None:
        scope = current_scope()
        if scope is not None and name in INGEST_TOOLS:
            mark_tainted(scope.session_id)

    def _after_success(self, name: str, decision: ActionDecision) -> None:
        scope = current_scope()
        session_id = scope.session_id if scope else ""
        if decision.category and decision.effect != "deny":
            bump_count(session_id, decision.category)
        self._note_ingest(name)

    async def _ask(self, request: ToolCallRequest, decision: ActionDecision, name: str, args: dict) -> list[dict]:
        scope = current_scope()
        session_id = scope.session_id if scope else ""
        request_id = uuid.uuid4().hex
        description = describe_action(name, args, decision.category)
        message = {
            "type": "hitl_request",
            "content": decision.title,
            "metadata": {
                "type": "action_policy",
                "request_id": request_id,
                "action_requests": [{
                    "name": name,
                    "args": args,
                    "description": description,
                }],
                "review_configs": [{
                    "action_name": name,
                    "allowed_decisions": ["approve", "reject"],
                }],
                "risk": {
                    "category": decision.category,
                    "labels": list(decision.labels),
                    "no_auto_approve": decision.no_auto_approve,
                    "grantable": decision.grantable,
                    "title": decision.title,
                },
            },
        }
        if not session_id:
            logger.warning("action policy: ask with no session; refusing %s", name)
            return [{"type": "reject", "message": "No session to ask in"}]
        return await request_approval(session_id, message)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        decision, name, args, server = self._decide(request)
        if decision.effect == "allow" and decision.category is None:
            result = handler(request)
            self._note_ingest(name)
            return result
        if decision.effect == "deny":
            self._audit(decision, name, server, args, "deny")
            return _blocked(request, f"Blocked by action policy: {decision.reason}.")
        if decision.effect == "ask":
            self._audit(decision, name, server, args, "deny")
            return _blocked(
                request,
                "Blocked by action policy: this action needs approval and the "
                "synchronous tool path cannot ask. Retry is not useful.",
            )
        self._audit(decision, name, server, args, "allow")
        result = handler(request)
        self._after_success(name, decision)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        decision, name, args, server = self._decide(request)
        if decision.effect == "allow" and decision.category is None:
            result = await handler(request)
            self._note_ingest(name)
            return result

        if decision.effect == "deny":
            self._audit(decision, name, server, args, "deny")
            return _blocked(request, f"Blocked by action policy: {decision.reason}.")

        if decision.effect == "ask":
            decisions = await self._ask(request, decision, name, args)
            first = decisions[0] if decisions else {}
            if first.get("type") != "approve":
                reason = str(first.get("message") or "You rejected this action")
                self._audit(decision, name, server, args, "reject")
                return _blocked(request, f"Not run: {reason}.")
            grant = first.get("grant_category")
            scope = current_scope()
            if (
                decision.grantable
                and grant
                and grant == decision.category
                and scope is not None
            ):
                grant_session(scope.session_id, str(grant))
            if decision.category in INVOCATION_GRANTABLE:
                grant_invocation(decision.category)
            self._audit(decision, name, server, args, "approve")
        else:
            self._audit(decision, name, server, args, "allow")

        result = await handler(request)
        self._after_success(name, decision)
        return result
