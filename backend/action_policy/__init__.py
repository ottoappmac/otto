"""Server-enforced policy for high-risk agent actions.

The catalog classifies tools. :mod:`backend.action_policy.gate` turns a
classification plus the saved settings into allow / ask / deny.
:class:`backend.action_policy.middleware.ActionPolicyMiddleware` applies
that decision on the orchestrator and every sub-agent.
"""

from backend.action_policy.catalog import (
    CATEGORIES,
    MACOS_SERVER_IDS,
    classify_tool,
)
from backend.action_policy.gate import ActionDecision, decide

__all__ = [
    "CATEGORIES",
    "MACOS_SERVER_IDS",
    "ActionDecision",
    "classify_tool",
    "decide",
]
