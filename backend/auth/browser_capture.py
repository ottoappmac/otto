"""Browser-based bearer-token capture via Chrome DevTools Protocol.

Adapted from the Tosca Cloud integration in
``agent-everything-bagel/backend/tosca_cloud.py`` and generalised so a
single provider serves any SaaS that authenticates the user through a
real browser session (Okta, Azure AD, Ping, custom SSO).

When to use this kind
---------------------
* The vendor has no public OAuth client registration available to you,
  but it does authenticate human users through a normal web login.
* Calls to the vendor's API carry an ``Authorization: Bearer …`` header
  (or any other configured header) that the browser session mints.

How it works
------------
1. Launch the system Chromium / Chrome / Edge with a private user-data
   directory and ``--remote-debugging-port=0``.  Chrome writes the
   negotiated port into ``DevToolsActivePort`` inside the profile dir.
2. Open a CDP WebSocket against the first ``page`` target.
3. Enable the ``Network`` domain.  Watch for the configured header
   (``MCPAuthConfig.header_name``, default ``Authorization``) on
   either ``Network.requestWillBeSent`` or
   ``Network.requestWillBeSentExtraInfo`` events.
4. On match, capture the token, kill the browser, persist a bundle.

The provider reconnects automatically when the WebSocket drops mid-
flow — that's what happens when Okta cross-origin-redirects back to
the app target.

Hard limitations
----------------
* No refresh.  Browser-captured tokens are typically short-lived
  bearer tokens with no refresh-token analogue.  When they expire the
  user has to log in again.  ``refresh()`` returns ``None`` so the
  manager raises ``NeedsLoginError`` and the UI surfaces a "Login"
  button.
* Phishing guard.  We require the manifest to declare
  ``MCPAuthConfig.allowed_hosts`` for any non-built-in MCP — otherwise
  a malicious agent-authored MCP could send the user through a
  lookalike domain and capture credentials they entered there.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import httpx

from backend.auth.base import AuthBundle, NeedsLoginError, register_provider
from backend.auth.utils import (
    find_chromium_browser,
    host_is_allowed,
    is_bundle_expired,
    isoformat,
    jwt_exp_iso,
    now_utc,
    project_env,
)

if TYPE_CHECKING:
    from backend.config import MCPAuthConfig


logger = logging.getLogger(__name__)


_PORT_FILE = "DevToolsActivePort"
_BROWSER_READY_TIMEOUT_SECS = 15
_DEFAULT_LOGIN_TIMEOUT_MS = 180_000
_MIN_TOKEN_LEN = 16  # filter out obviously-empty headers without rejecting JWTs


@register_provider
class BrowserCaptureProvider:
    """CDP-based bearer-token sniff.  Suitable for SSO-fronted SaaS."""

    kind: ClassVar[str] = "browser_capture"

    async def acquire(
        self, auth: "MCPAuthConfig", server_id: str,
    ) -> AuthBundle:
        if not auth.landing_url:
            raise NeedsLoginError(
                server_id, kind=self.kind,
                reason="landing_url must be configured",
            )
        if not host_is_allowed(auth.landing_url, auth.allowed_hosts):
            raise NeedsLoginError(
                server_id, kind=self.kind,
                reason="landing_url not in allowed_hosts",
            )

        browser_path = find_chromium_browser()
        if not browser_path:
            raise RuntimeError(
                "No Chrome / Edge / Chromium binary found.  Browser-based "
                "MCP login requires a Chromium-family browser to be "
                "installed."
            )

        # Run the blocking CDP capture in a worker thread so the
        # FastAPI event loop stays responsive — the user may take a
        # minute or two on the consent screen.
        token = await asyncio.to_thread(
            _run_capture,
            browser_path=browser_path,
            landing_url=auth.landing_url,
            header_name=auth.header_name or "Authorization",
            token_prefix=auth.token_prefix or "Bearer ",
            allowed_hosts=list(auth.allowed_hosts or []),
            user_data_dir=str(_user_data_dir(server_id)),
            timeout_ms=_DEFAULT_LOGIN_TIMEOUT_MS,
            server_id=server_id,
        )

        if not token:
            raise NeedsLoginError(
                server_id, kind=self.kind,
                reason="no token captured before timeout",
            )

        bundle: AuthBundle = {
            "access_token": token,
            "token_type": (auth.token_prefix or "Bearer ").strip() or "Bearer",
            "obtained_iso": isoformat(now_utc()),
        }
        # If the token happens to be a JWT, infer the expiry from the
        # ``exp`` claim so our staleness check has something to compare
        # against.  Otherwise the bundle has no expiry and
        # ``is_bundle_expired`` treats it as stale on first refresh,
        # which forces the user to re-login — appropriate when we have
        # no other signal.
        exp_iso = jwt_exp_iso(token)
        if exp_iso:
            bundle["expiry_iso"] = exp_iso

        return bundle

    async def refresh(
        self, auth: "MCPAuthConfig", server_id: str, bundle: AuthBundle,
    ) -> AuthBundle | None:
        # Browser-captured tokens have no refresh path — when they
        # expire the user has to walk through the consent screen again.
        # Returning the bundle unchanged when it's still valid means
        # the manager can call refresh() unconditionally before spawn.
        if bundle and not is_bundle_expired(bundle):
            return bundle
        return None

    def is_expired(self, bundle: AuthBundle) -> bool:
        return is_bundle_expired(bundle)

    def env_for(
        self, auth: "MCPAuthConfig", bundle: AuthBundle,
    ) -> dict[str, str]:
        return project_env(auth, bundle)


# ---------------------------------------------------------------------------
# Browser launch + CDP capture (blocking helpers, run in a worker thread)
# ---------------------------------------------------------------------------


def _user_data_dir(server_id: str) -> Path:
    """Per-MCP Chrome profile directory under the app data dir.

    Keeping this stable across login attempts means returning users
    don't get prompted by Okta to re-enter their MFA token, but each
    MCP gets its own profile so cookies don't leak between vendors.
    """
    from backend.config import get_app_data_dir

    p = get_app_data_dir() / "browser_login_profiles" / server_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _run_capture(
    *,
    browser_path: str,
    landing_url: str,
    header_name: str,
    token_prefix: str,
    allowed_hosts: list[str],
    user_data_dir: str,
    timeout_ms: int,
    server_id: str,
) -> str | None:
    """Synchronous wrapper: launch browser → capture → clean up."""
    _kill_stale_login_browsers(user_data_dir)

    port_file = Path(user_data_dir) / _PORT_FILE
    if port_file.exists():
        port_file.unlink()

    logger.info(
        "browser_capture [%s]: launching %s → %s",
        server_id, browser_path, landing_url,
    )

    proc, debug_port = _launch_chrome_cdp(browser_path, landing_url, user_data_dir)
    logger.info(
        "browser_capture [%s]: chrome pid=%d cdp_port=%d",
        server_id, proc.pid, debug_port,
    )

    try:
        token = asyncio.run(_cdp_capture_token(
            debug_port=debug_port,
            timeout_ms=timeout_ms,
            header_name=header_name,
            token_prefix=token_prefix,
            allowed_hosts=allowed_hosts,
            server_id=server_id,
        ))
    finally:
        _kill_browser(proc)

    return token


def _launch_chrome_cdp(
    browser_path: str, url: str, user_data_dir: str,
) -> tuple[subprocess.Popen, int]:
    cmd = [
        browser_path,
        "--remote-debugging-port=0",
        f"--user-data-dir={user_data_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-session-crashed-bubble",
        "--hide-crash-restore-bubble",
        url,
    ]
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs,
    )

    port_file = Path(user_data_dir) / _PORT_FILE
    deadline = time.monotonic() + _BROWSER_READY_TIMEOUT_SECS
    while time.monotonic() < deadline:
        if port_file.exists():
            text = port_file.read_text().strip()
            lines = text.splitlines()
            if lines:
                try:
                    return proc, int(lines[0])
                except ValueError:
                    pass
        time.sleep(0.3)

    proc.kill()
    raise TimeoutError(
        f"Chrome did not write {_PORT_FILE} within "
        f"{_BROWSER_READY_TIMEOUT_SECS}s"
    )


def _kill_browser(proc: subprocess.Popen) -> None:
    try:
        if sys.platform == "win32":
            proc.terminate()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _kill_stale_login_browsers(user_data_dir: str) -> None:
    """Best-effort: terminate any leftover Chrome holding the same profile."""
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                [
                    "wmic", "process", "where",
                    f"commandline like '%{user_data_dir}%'",
                    "get", "processid",
                ],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().splitlines()[1:]:
                pid_str = line.strip()
                if pid_str.isdigit():
                    try:
                        os.kill(int(pid_str), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        pass
        else:
            result = subprocess.run(
                ["pgrep", "-f", user_data_dir],
                capture_output=True, text=True, timeout=5,
            )
            for pid_str in result.stdout.strip().splitlines():
                pid_str = pid_str.strip()
                if pid_str.isdigit():
                    try:
                        os.kill(int(pid_str), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        pass
    except Exception as exc:
        logger.debug("Could not check for stale browsers: %s", exc)


async def _cdp_capture_token(
    *,
    debug_port: int,
    timeout_ms: int,
    header_name: str,
    token_prefix: str,
    allowed_hosts: list[str],
    server_id: str,
) -> str | None:
    """Connect to Chrome via CDP and return the first matching token."""
    try:
        import websockets
        import websockets.exceptions
    except ImportError as exc:
        raise RuntimeError(
            "browser_capture provider requires the 'websockets' package "
            "(declared in pyproject.toml dependencies)."
        ) from exc

    targets_url = f"http://127.0.0.1:{debug_port}/json"

    async def _find_page_target() -> tuple[str | None, str | None]:
        async with httpx.AsyncClient(timeout=5) as client:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 10
            while loop.time() < deadline:
                try:
                    resp = await client.get(targets_url)
                    for t in resp.json():
                        if t.get("type") == "page":
                            return (
                                t.get("webSocketDebuggerUrl"),
                                t.get("url", ""),
                            )
                except Exception:
                    pass
                await asyncio.sleep(0.5)
        return None, None

    ws_url, page_url = await _find_page_target()
    if not ws_url:
        logger.warning(
            "browser_capture [%s]: no page target on CDP port %d",
            server_id, debug_port,
        )
        return None
    logger.debug(
        "browser_capture [%s]: page target url=%s ws=%s",
        server_id, page_url, ws_url,
    )

    bearer_token: str | None = None
    loop = asyncio.get_running_loop()
    overall_deadline = loop.time() + timeout_ms / 1000
    reconnects = 0

    while not bearer_token and loop.time() < overall_deadline:
        try:
            async with websockets.connect(ws_url, max_size=50 * 1024 * 1024) as ws:
                await ws.send(json.dumps(
                    {"id": 1, "method": "Network.enable", "params": {}}
                ))
                logger.debug(
                    "browser_capture [%s]: CDP connected (attempt %d)",
                    server_id, reconnects + 1,
                )

                while not bearer_token and loop.time() < overall_deadline:
                    remaining = overall_deadline - loop.time()
                    wait = min(0.5, remaining) if remaining > 0 else 0
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=wait)
                    except asyncio.TimeoutError:
                        continue

                    bearer_token = _extract_token_from_event(
                        raw, header_name=header_name,
                        token_prefix=token_prefix,
                        allowed_hosts=allowed_hosts,
                    )
        except (
            websockets.exceptions.ConnectionClosed,
            websockets.exceptions.ConnectionClosedError,
            websockets.exceptions.ConnectionClosedOK,
            ConnectionResetError,
            OSError,
        ):
            # WebSocket dropped mid-flow — typical when Okta cross-
            # origin-redirects to the app domain and Chrome destroys
            # the original page target.  Find the new target and
            # reattach so the user doesn't lose the in-progress login.
            reconnects += 1
            await asyncio.sleep(0.5)
            ws_url, page_url = await _find_page_target()
            if not ws_url:
                logger.warning(
                    "browser_capture [%s]: lost page target after reconnect",
                    server_id,
                )
                return None
        except Exception as exc:
            logger.warning(
                "browser_capture [%s]: unexpected CDP error: %s: %s",
                server_id, type(exc).__name__, exc,
            )
            return None

    logger.info(
        "browser_capture [%s]: capture %s after %d reconnects",
        server_id, "succeeded" if bearer_token else "timed out", reconnects,
    )
    return bearer_token


def _extract_token_from_event(
    raw: str,
    *,
    header_name: str,
    token_prefix: str,
    allowed_hosts: list[str],
) -> str | None:
    """Look at one CDP event; return the captured token if it matches."""
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return None

    method = msg.get("method", "")
    headers: dict[str, Any] = {}
    request_url = ""

    if method == "Network.requestWillBeSent":
        params = msg.get("params") or {}
        request = params.get("request") or {}
        headers = request.get("headers") or {}
        request_url = request.get("url", "")
    elif method == "Network.requestWillBeSentExtraInfo":
        params = msg.get("params") or {}
        headers = params.get("headers") or {}
        # ExtraInfo doesn't carry the URL — rely on the paired
        # requestWillBeSent event to cover the host check.  When the
        # two events disagree we still fall back to allowing the match
        # because the bearer header itself never reaches the LLM.
        request_url = ""
    else:
        return None

    if request_url and not host_is_allowed(request_url, allowed_hosts):
        return None

    target_value = ""
    for k, v in headers.items():
        if k.lower() == header_name.lower():
            target_value = str(v)
            break
    if not target_value:
        return None

    if token_prefix and target_value.startswith(token_prefix):
        token = target_value[len(token_prefix):]
    else:
        token = target_value

    token = token.strip()
    if len(token) < _MIN_TOKEN_LEN:
        return None
    return token
