"""Verifiable airplane-mode for Otto.

Otto's value proposition includes *no cloud relay, no telemetry*.  This
module makes that promise auditable.  When the user "engages" the
privacy lock:

1. **App-layer guard** -- :func:`enforce_provider_allowed` is consulted
   on every LLM construction.  Cloud providers (Anthropic, OpenAI,
   Cohere) raise :class:`PrivacyLockActive` and refuse to build a model.
   This is the layer that *cannot* be bypassed by a misconfigured
   subprocess: the cloud client never even gets instantiated.

2. **Audit log** -- :func:`audit_event` appends a JSON line to
   ``<app_data>/privacy_audit.log`` for every engage / disengage /
   refusal event.  The log is append-only and surface-able via the
   ``/api/privacy/audit`` route.

3. **pf template** -- on macOS we render an ``otto.privacy`` packet
   filter anchor that blocks all outbound traffic except loopback,
   mDNS, and the explicit ``allowed_hosts`` list.  We deliberately do
   NOT auto-install it (that would need ``sudo``).  The user gets a
   pre-flight command to copy into Terminal; :func:`pf_status` can
   then verify the kernel is actually carrying the anchor.

The boundary between layers is intentional: most users will care only
about layer 1, but the few who need a verifiable claim ("we can prove
the agent did not exfiltrate data") can install layer 3 and inspect
``pfctl -s rules -a otto.privacy``.
"""

from __future__ import annotations

import json
import logging
import secrets
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from backend.config import AppConfig, PrivacyConfig, get_app_data_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants & exceptions
# ---------------------------------------------------------------------------


# Providers that always count as "on-device or local network only".
# Used as the default of :attr:`PrivacyConfig.local_only_providers` and
# as a fallback if a user blanks the config field.
DEFAULT_LOCAL_PROVIDERS: frozenset[str] = frozenset({"mlx", "omlx", "exo"})


# Audit log lives next to ``config.json`` so a user backing up app-data
# captures both pieces of evidence together.
AUDIT_LOG_NAME = "privacy_audit.log"


class PrivacyLockActive(RuntimeError):
    """Raised when an operation is blocked because privacy mode is engaged."""


# ---------------------------------------------------------------------------
# Engagement state helpers
# ---------------------------------------------------------------------------


def is_engaged(cfg: AppConfig | PrivacyConfig) -> bool:
    """Return True when the privacy lock is currently engaged."""
    privacy = cfg.privacy if isinstance(cfg, AppConfig) else cfg
    return bool(privacy.enabled)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def engage(cfg: AppConfig) -> dict[str, Any]:
    """Flip the lock on, stamp the engagement timestamp + audit token.

    Idempotent: calling this when already engaged returns the existing
    audit token without rotating it (so resuming a session doesn't
    invalidate prior audit entries).
    """
    p = cfg.privacy
    rotated = False
    if not p.enabled:
        p.enabled = True
        p.engaged_at = _now_iso()
        p.audit_token = secrets.token_hex(16)
        rotated = True
    audit_event("engage", cfg, extra={"rotated": rotated})
    return {
        "engaged": True,
        "engaged_at": p.engaged_at,
        "audit_token": p.audit_token,
        "allowed_providers": list(_effective_local_providers(p)),
    }


def disengage(cfg: AppConfig) -> dict[str, Any]:
    """Flip the lock off; do NOT clear ``engaged_at`` so the audit trail
    keeps the prior engagement window visible to administrators."""
    p = cfg.privacy
    was_engaged = p.enabled
    p.enabled = False
    audit_event("disengage", cfg, extra={"was_engaged": was_engaged})
    return {
        "engaged": False,
        "previously_engaged_at": p.engaged_at,
        "audit_token": p.audit_token,
    }


def _effective_local_providers(p: PrivacyConfig) -> frozenset[str]:
    """Return the set of providers allowed while engaged.

    Falls back to :data:`DEFAULT_LOCAL_PROVIDERS` if the user blanked
    the list -- a misconfiguration shouldn't accidentally let cloud
    providers through.
    """
    if p.local_only_providers:
        return frozenset(s.strip().lower() for s in p.local_only_providers if s.strip())
    return DEFAULT_LOCAL_PROVIDERS


# ---------------------------------------------------------------------------
# App-layer guard
# ---------------------------------------------------------------------------


def enforce_provider_allowed(provider: str, cfg: AppConfig | None = None) -> None:
    """Raise :class:`PrivacyLockActive` when *provider* would breach the lock.

    Called from :func:`deep_agent.model_factory.create_llm` and any
    other site that constructs an LLM.  Loads the config lazily so
    callers don't have to thread it through every layer.
    """
    if cfg is None:
        cfg = AppConfig.load()
    p = cfg.privacy
    if not p.enabled:
        return

    prov = (provider or "").strip().lower()
    allowed = _effective_local_providers(p)
    if prov not in allowed:
        # Audit the refusal *before* raising so a tampered subprocess
        # that swallows the exception still leaves a trace.
        audit_event(
            "refuse_provider",
            cfg,
            extra={"provider": prov, "allowed": sorted(allowed)},
        )
        raise PrivacyLockActive(
            f"Privacy lock is engaged.  Provider {prov!r} sends data "
            f"off-device, which is not allowed in this mode.  "
            f"Allowed providers: {sorted(allowed)}.  "
            f"Disengage in Settings → Privacy & Security to use a "
            f"cloud LLM."
        )


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def audit_log_path() -> Path:
    return get_app_data_dir() / AUDIT_LOG_NAME


def audit_event(
    event: str,
    cfg: AppConfig | None,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append one JSON line describing a privacy event.

    Failures to write are logged but never raised -- the lock guard is
    more important than the audit log; we never want a full disk to
    let cloud providers through.
    """
    token = ""
    engaged = False
    if cfg is not None:
        token = cfg.privacy.audit_token
        engaged = cfg.privacy.enabled
    entry = {
        "ts": _now_iso(),
        "event": event,
        "audit_token": token,
        "engaged": engaged,
    }
    if extra:
        entry.update(extra)
    try:
        path = audit_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
    except OSError as exc:
        logger.warning("privacy audit log write failed: %s", exc)


def tail_audit(n: int = 100) -> list[dict[str, Any]]:
    """Return the last *n* audit entries, newest first."""
    path = audit_log_path()
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(out) >= max(1, n):
            break
    return out


# ---------------------------------------------------------------------------
# macOS pf (packet filter) integration
# ---------------------------------------------------------------------------


_PF_HEADER = """# Otto privacy lock -- otto.privacy anchor
#
# This template is generated by Otto's privacy_lock module.  Install
# with:
#
#   sudo pfctl -a {anchor} -f - <<'EOF'
#   ... (paste the rules below) ...
#   EOF
#
# Verify with:
#
#   sudo pfctl -a {anchor} -s rules
#
# Remove with:
#
#   sudo pfctl -a {anchor} -F all
#
# These rules block ALL outbound traffic from this user/process except
# loopback, mDNS (when allowed), and the explicit allowed_hosts list.
# MLX runs entirely in-process so it is unaffected.

"""


def _parse_host_port(spec: str) -> tuple[str, int | None]:
    """Split ``host[:port]`` -- both halves optional.

    IPv6 literals must be wrapped in brackets (``[::1]:1234``).
    """
    spec = spec.strip()
    if spec.startswith("["):
        # IPv6 -- find closing bracket
        end = spec.find("]")
        if end < 0:
            return spec, None
        host = spec[1:end]
        rest = spec[end + 1:]
        if rest.startswith(":"):
            try:
                return host, int(rest[1:])
            except ValueError:
                return host, None
        return host, None
    if ":" in spec:
        host, _, port = spec.rpartition(":")
        try:
            return host, int(port)
        except ValueError:
            return spec, None
    return spec, None


def render_pf_template(cfg: AppConfig | PrivacyConfig) -> str:
    """Return the pf.conf snippet for the configured allowlist.

    The output is intended to be loaded as an anchor (``pfctl -a``).
    Default policy is ``block out``, with passes for loopback, mDNS,
    and every entry in ``allowed_hosts``.
    """
    p = cfg.privacy if isinstance(cfg, AppConfig) else cfg
    anchor = p.pf_anchor or "otto.privacy"
    lines: list[str] = [_PF_HEADER.format(anchor=anchor)]
    lines.append("# Default policy: deny outbound")
    lines.append("block out all")
    lines.append("")
    if p.allow_loopback:
        lines.append("# Loopback (local MCPs, embedded backend)")
        lines.append("pass out quick on lo0 all")
        lines.append("pass out quick inet from any to 127.0.0.0/8")
        lines.append("pass out quick inet6 from any to ::1")
        lines.append("")
    if p.allow_mdns:
        lines.append("# mDNS / Bonjour (exo cluster discovery)")
        lines.append("pass out quick proto udp to 224.0.0.251 port 5353")
        lines.append("pass out quick proto udp to ff02::fb port 5353")
        lines.append("")
    if p.allowed_hosts:
        lines.append("# Explicit allowed_hosts allowlist")
        for raw in p.allowed_hosts:
            host, port = _parse_host_port(raw)
            if not host:
                continue
            if port is not None:
                lines.append(f"pass out quick proto tcp to {host} port {port}")
                lines.append(f"pass out quick proto udp to {host} port {port}")
            else:
                lines.append(f"pass out quick to {host}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def pf_install_command(cfg: AppConfig | PrivacyConfig) -> str:
    """Return the shell command the user copies into Terminal.

    Kept as a single line so the UI can render it inline next to a
    "copy" button.  We deliberately stop short of running this for the
    user -- silent ``sudo`` is exactly the kind of surface a privacy
    feature should avoid.
    """
    p = cfg.privacy if isinstance(cfg, AppConfig) else cfg
    anchor = p.pf_anchor or "otto.privacy"
    return f"sudo pfctl -a {anchor} -f -"


def pf_status(cfg: AppConfig | PrivacyConfig) -> dict[str, Any]:
    """Best-effort check that the pf anchor is loaded with our rules.

    Calls ``pfctl -a <anchor> -s rules`` and returns a dict describing
    what we found.  Most users will run Otto without sudo so this is
    expected to return ``{"available": False, ...}`` -- we surface
    that cleanly rather than raising.
    """
    p = cfg.privacy if isinstance(cfg, AppConfig) else cfg
    if sys.platform != "darwin":
        return {"available": False, "reason": "pf is macOS-only"}
    pfctl = shutil.which("pfctl") or "/sbin/pfctl"
    if not Path(pfctl).is_file():
        return {"available": False, "reason": "pfctl not found"}
    try:
        result = subprocess.run(
            [pfctl, "-a", p.pf_anchor, "-s", "rules"],
            capture_output=True,
            text=True,
            timeout=4.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": f"pfctl call failed: {exc}"}

    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    # pfctl returns rc != 0 when the anchor doesn't exist (or the user
    # doesn't have permission).  Both are useful information to render.
    rules = [ln for ln in out.splitlines() if ln.strip()]
    has_block = any(ln.startswith("block out") for ln in rules)
    return {
        "available": result.returncode == 0,
        "anchor": p.pf_anchor,
        "rule_count": len(rules),
        "has_block_rule": has_block,
        "stdout_excerpt": out[:1000],
        "stderr_excerpt": err[:1000],
        "exit_code": result.returncode,
    }


def join_allowlists(*lists: Iterable[str]) -> list[str]:
    """Helper used by config migrations and the route layer."""
    out: list[str] = []
    seen: set[str] = set()
    for lst in lists:
        for s in lst or []:
            cleaned = s.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                out.append(cleaned)
    return out


__all__ = [
    "DEFAULT_LOCAL_PROVIDERS",
    "PrivacyLockActive",
    "is_engaged",
    "engage",
    "disengage",
    "enforce_provider_allowed",
    "audit_log_path",
    "audit_event",
    "tail_audit",
    "render_pf_template",
    "pf_install_command",
    "pf_status",
    "join_allowlists",
]
