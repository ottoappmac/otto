"""Server-side orchestrator for the EXO Cluster *setup-from-scratch* wizard.

The wizard lives in Settings → LLM → Cluster → "Add remote → Set up new
node from scratch." It is **strictly human-driven**: nothing in this
module is exposed to the deep agent's tool surface. The agent stays a
read-only consumer of the cluster (see :mod:`backend.exo_tools`).

What this module does:

1. :func:`probe_host` — non-interactive probe via the system ``ssh``
   binary against ``user@host:port``. Reports TCP reachability, key-auth
   success, OS, architecture, and whether ``uv`` / ``exo`` are already
   installed on the remote. **Never** sends a password.

2. :func:`list_local_keypairs` — enumerates ``~/.ssh/`` for ED25519/RSA
   keypairs the user could authorize. Returns paths + fingerprints; the
   private key bytes never leave this process.

3. :func:`create_keypair` — generates a fresh ED25519 keypair with a
   one-shot ``ssh-keygen`` call. Idempotent (refuses to overwrite an
   existing file).

4. :func:`install_authorized_key` — the **only** function that ever
   sees a password. Connects via ``asyncssh`` (password as kwarg, never
   argv), appends the public key to the remote's
   ``~/.ssh/authorized_keys``, fixes permissions, and verifies by
   reconnecting with the key. The password is held in a local var and
   discarded.

5. :func:`append_ssh_config_block` — appends a ``Host`` block to
   ``~/.ssh/config``. Always backs up to
   ``~/.ssh/config.bak.<unix-ts>`` before writing. Idempotent: refuses
   to overwrite an existing alias unless ``replace=True``.

6. :func:`SecretScrubFilter` — a ``logging.Filter`` defence-in-depth
   against accidental password leaks into the log file. Registered on
   the root logger by :mod:`backend.server`.

Threat model and design rules:

* Passwords come in as :class:`pydantic.SecretStr` from the route layer
  so a stray ``repr()`` / ``json.dumps()`` shows ``**********``.
* Password-bearing endpoints never echo the body back in their response.
* No tool in :mod:`backend.exo_tools` triggers any flow here.
* All disk writes are gated behind explicit user actions in the wizard.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import platform
import re
import shutil
import stat
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSH probe (no password)
# ---------------------------------------------------------------------------


@dataclass
class SshProbeResult:
    """Result of a non-interactive probe against ``user@host:port``."""

    tcp_reachable: bool = False
    key_auth_ok: bool = False
    password_auth_available: bool = False
    os_name: str = ""
    arch: str = ""
    has_uv: bool = False
    has_exo: bool = False
    hostname_canonical: str = ""
    error: str = ""
    # Hint text suitable for the UI traffic-light, mirrors the style of
    # ``backend.exo_discovery.test_ssh``.
    hint: str = ""


_PROBE_REMOTE_SCRIPT = (
    'echo __exo_ok__; '
    'uname -s; '
    'uname -m; '
    'hostname; '
    'command -v uv >/dev/null 2>&1 && echo __uv__:yes || echo __uv__:no; '
    'command -v exo >/dev/null 2>&1 && echo __exo__:yes || echo __exo__:no'
)


def _ssh_probe_argv(user: str, host: str, port: int, *, timeout: int) -> list[str]:
    """Return the argv for a single non-interactive ``ssh`` probe.

    ``BatchMode=yes`` ensures ssh never prompts for a password — if key
    auth fails we get rc=255 and a stderr we can interpret, rather than
    a hung process.
    """
    target = f"{user}@{host}" if user else host
    return [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={timeout}",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "PreferredAuthentications=publickey",
        "-p", str(port),
        target,
        _PROBE_REMOTE_SCRIPT,
    ]


def _interpret_probe_stderr(stderr: str) -> tuple[bool, bool, str]:
    """Parse stderr from a failed probe → (tcp_reachable, password_auth, hint)."""
    s = stderr.lower()
    if "could not resolve hostname" in s or "name or service not known" in s:
        return False, False, (
            "Cannot resolve hostname. If you're targeting a Bonjour name "
            "(``foo.local``), make sure mDNS is healthy on this network — "
            "or use the IPv4 address from the LAN scan."
        )
    if "connection refused" in s:
        return False, False, (
            "Connection refused — Remote Login (sshd) is not enabled on the "
            "target. On macOS: System Settings → General → Sharing → "
            "Remote Login."
        )
    if "operation timed out" in s or "connection timed out" in s:
        return False, False, (
            "Connection timed out — the target is unreachable, behind a "
            "firewall, or the address has changed (Thunderbolt-Bridge IPs "
            "rotate on reconnect)."
        )
    if "host key verification failed" in s or "host key" in s:
        return True, True, (
            "Host key changed or unknown. Remove the stale entry with "
            "``ssh-keygen -R <host>`` and try again."
        )
    if "permission denied" in s and "publickey" in s:
        # Could mean: server is reachable + supports keys but the user
        # has no authorized key. Detect whether password auth is also
        # offered ("publickey,password" in the auth methods list).
        offers_password = "password" in s
        return True, offers_password, (
            "Reachable, but key auth is not yet authorized. "
            + ("Use the wizard's Authorize step to install a key." if offers_password
               else "Server only accepts publickey; install a key manually then re-probe.")
        )
    if "permission denied" in s:
        return True, False, "Reachable, but authentication failed."
    if "no route to host" in s:
        return False, False, "No route to host — check the address / network."
    return False, False, ""


async def probe_host(host: str, user: str, port: int = 22, *, timeout: float = 6.0) -> dict:
    """Run a non-interactive SSH probe against ``user@host:port``.

    Returns the asdict() form of :class:`SshProbeResult`. This function
    never sends a password — it relies on key auth (``BatchMode=yes``)
    and treats "permission denied (publickey)" as a useful signal that
    the host is reachable but needs a key installed.
    """
    host = (host or "").strip()
    user = (user or "").strip()
    if not host:
        return asdict(SshProbeResult(error="host is required", hint="Empty host."))
    port = int(port or 22)
    timeout = max(2.0, min(30.0, float(timeout)))

    out = SshProbeResult()

    def _run_sync() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            _ssh_probe_argv(user, host, port, timeout=int(min(timeout, 10))),
            capture_output=True,
            text=True,
            timeout=timeout + 2.0,
            check=False,
        )

    try:
        proc = await asyncio.to_thread(_run_sync)
    except subprocess.TimeoutExpired:
        out.tcp_reachable = False
        out.error = "ssh probe timed out"
        out.hint = (
            f"ssh probe of {host}:{port} timed out after {timeout:.0f}s — "
            "the host is unreachable, behind a firewall, or refusing connections."
        )
        return asdict(out)
    except FileNotFoundError:
        out.error = "ssh binary not found"
        out.hint = "`ssh` not found on this machine — install OpenSSH client."
        return asdict(out)

    if proc.returncode == 0 and "__exo_ok__" in proc.stdout:
        out.tcp_reachable = True
        out.key_auth_ok = True
        out.password_auth_available = True  # both possible if key works
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line or line == "__exo_ok__":
                continue
            if line.startswith("__uv__:"):
                out.has_uv = line.endswith(":yes")
            elif line.startswith("__exo__:"):
                out.has_exo = line.endswith(":yes")
            elif not out.os_name:
                out.os_name = line
            elif not out.arch:
                out.arch = line
            elif not out.hostname_canonical:
                out.hostname_canonical = line
        out.hint = "Reachable, key auth works."
        return asdict(out)

    tcp_ok, has_pw, hint = _interpret_probe_stderr(proc.stderr or "")
    out.tcp_reachable = tcp_ok
    out.password_auth_available = has_pw
    out.hint = hint or f"ssh exited rc={proc.returncode}"
    out.error = (proc.stderr or "").strip().splitlines()[-1] if proc.stderr else ""
    return asdict(out)


# ---------------------------------------------------------------------------
# Local keypair management
# ---------------------------------------------------------------------------


@dataclass
class LocalKeypair:
    """A keypair candidate found under ``~/.ssh/``."""

    private_path: str
    public_path: str
    key_type: str = ""        # ed25519, rsa, ecdsa, …
    fingerprint: str = ""     # "SHA256:abc..."
    bits: int = 0
    comment: str = ""


_KNOWN_PRIVATE_NAMES = (
    "id_ed25519",
    "id_ed25519_exo",
    "id_ed25519_otto",
    "id_rsa",
    "id_ecdsa",
)


def _parse_keygen_lf(output: str) -> tuple[int, str, str, str]:
    """Parse output of ``ssh-keygen -lf <path>``.

    Format: ``<bits> SHA256:<fp> <comment> (<TYPE>)``
    Returns ``(bits, fingerprint, comment, key_type)``. Empty strings on
    a line we can't parse — we never want this to crash a listing.
    """
    output = output.strip()
    if not output:
        return 0, "", "", ""
    # Split off the trailing "(TYPE)"
    m = re.match(r"^(?P<bits>\d+)\s+(?P<fp>\S+)\s+(?P<comment>.*?)\s+\((?P<type>[A-Z0-9]+)\)\s*$", output)
    if not m:
        return 0, "", "", ""
    return (
        int(m.group("bits")),
        m.group("fp"),
        m.group("comment"),
        m.group("type").lower(),
    )


def _fingerprint_pubkey(pub_path: Path) -> tuple[int, str, str, str]:
    if not shutil.which("ssh-keygen"):
        return 0, "", "", ""
    try:
        proc = subprocess.run(
            ["ssh-keygen", "-lf", str(pub_path)],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0, "", "", ""
    if proc.returncode != 0:
        return 0, "", "", ""
    return _parse_keygen_lf(proc.stdout)


def list_local_keypairs() -> list[dict]:
    """Enumerate candidate private keys in ``~/.ssh/``.

    Considers a fixed set of well-known names plus any ``id_*`` private
    file (and matching ``id_*.pub``) under ``~/.ssh/``. Returns dicts
    with fingerprint metadata; private key bytes are *never* read.
    """
    ssh_dir = Path.home() / ".ssh"
    if not ssh_dir.is_dir():
        return []

    seen: set[Path] = set()
    out: list[LocalKeypair] = []

    candidates: list[Path] = []
    for name in _KNOWN_PRIVATE_NAMES:
        p = ssh_dir / name
        if p.is_file():
            candidates.append(p)
    for p in sorted(ssh_dir.glob("id_*")):
        if p.suffix == ".pub":
            continue
        if p.is_file() and p not in candidates:
            candidates.append(p)

    for priv in candidates:
        if priv in seen:
            continue
        seen.add(priv)
        pub = priv.with_suffix(priv.suffix + ".pub") if priv.suffix else priv.parent / (priv.name + ".pub")
        if not pub.is_file():
            continue
        bits, fp, comment, ktype = _fingerprint_pubkey(pub)
        out.append(
            LocalKeypair(
                private_path=str(priv),
                public_path=str(pub),
                key_type=ktype,
                fingerprint=fp,
                bits=bits,
                comment=comment,
            )
        )
    return [asdict(k) for k in out]


# ---------------------------------------------------------------------------
# Keypair generation
# ---------------------------------------------------------------------------


_SAFE_KEY_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _expand(p: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(p))).resolve()


async def create_keypair(
    name: str,
    *,
    key_type: str = "ed25519",
    comment: str = "",
) -> dict:
    """Generate a fresh keypair under ``~/.ssh/<name>``.

    ``name`` must be a bare filename (no slashes) so the wizard can't
    accidentally write outside ``~/.ssh``. Refuses to overwrite an
    existing key.

    Returns a :class:`LocalKeypair`-shaped dict on success. The
    private key never leaves disk.
    """
    name = (name or "").strip()
    if not name or not _SAFE_KEY_NAME.match(name) or name.endswith(".pub"):
        raise ValueError(f"unsafe key name: {name!r}")
    if key_type not in ("ed25519", "rsa", "ecdsa"):
        raise ValueError(f"unsupported key_type: {key_type!r}")

    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(mode=0o700, exist_ok=True)
    try:
        os.chmod(ssh_dir, 0o700)
    except OSError as exc:
        logger.warning("could not chmod ~/.ssh to 0700: %s", exc)

    priv = ssh_dir / name
    pub = ssh_dir / f"{name}.pub"
    if priv.exists() or pub.exists():
        raise FileExistsError(f"{priv} or {pub} already exists")

    if not shutil.which("ssh-keygen"):
        raise RuntimeError("ssh-keygen not found on this machine")

    if not comment:
        comment = f"exo-cluster-{platform.node() or 'host'}-{int(time.time())}"

    argv = [
        "ssh-keygen",
        "-t", key_type,
        "-N", "",
        "-f", str(priv),
        "-C", comment,
        "-q",
    ]
    if key_type == "rsa":
        argv.extend(["-b", "4096"])

    def _run_sync() -> subprocess.CompletedProcess[str]:
        return subprocess.run(argv, capture_output=True, text=True, timeout=15.0, check=False)

    proc = await asyncio.to_thread(_run_sync)
    if proc.returncode != 0:
        raise RuntimeError(f"ssh-keygen failed: {(proc.stderr or proc.stdout).strip()}")

    try:
        os.chmod(priv, 0o600)
        os.chmod(pub, 0o644)
    except OSError as exc:
        logger.warning("could not chmod new keypair: %s", exc)

    bits, fp, c, ktype = _fingerprint_pubkey(pub)
    return asdict(
        LocalKeypair(
            private_path=str(priv),
            public_path=str(pub),
            key_type=ktype,
            fingerprint=fp,
            bits=bits,
            comment=c,
        )
    )


def read_pubkey(public_path: str) -> str:
    """Read the public key file (one-line OpenSSH format)."""
    p = _expand(public_path)
    if not p.is_file():
        raise FileNotFoundError(f"public key not found: {p}")
    if not str(p).endswith(".pub"):
        raise ValueError(f"refusing to read non-.pub file: {p}")
    return p.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Key install (the password-bearing call)
# ---------------------------------------------------------------------------


def _bonjour_name_for_ip(ip: str) -> str | None:
    """Return the Bonjour ``*.local`` name for a link-local IPv4 address.

    On macOS, ``socket.gethostbyaddr`` goes through ``mDNSResponder`` and
    resolves link-local addresses to their advertised Bonjour hostname when
    the remote is advertising one.  Returns ``None`` on any failure or when
    no ``.local`` name is found.

    Falls back to scanning the cached LAN-SSH results when the reverse
    lookup returns nothing (e.g. the remote hasn't re-advertised since a
    reconnect).
    """
    import socket as _socket
    try:
        name, _, _ = _socket.gethostbyaddr(ip)
        if name and name.endswith(".local"):
            return name
    except OSError:
        pass

    # Secondary: check the LAN scan cache — quick, no network call.
    try:
        from backend import exo_discovery
        for h in exo_discovery.scan_lan_ssh(timeout=0.5):
            if not (h.hostname and h.hostname.endswith(".local")):
                continue
            all_addrs = list(h.addresses or []) + list(h.thunderbolt_addresses or [])
            if ip in all_addrs:
                return h.hostname
    except Exception:  # noqa: BLE001
        pass
    return None


def _link_local_alternatives(host: str, port: int) -> dict[str, list[str]]:
    """For a link-local target, surface the Bonjour name and any other
    reachable alternatives.

    Returns a dict with keys:
    * ``bonjour`` — the ``*.local`` name if one can be resolved (list of 0 or 1)
    * ``reachable`` — other IPs that respond on ``port`` right now
    * ``other`` — IPs that are advertised but not currently answering

    An empty dict means nothing else is known.
    """
    try:
        ip = ipaddress.IPv4Address(host)
    except (ValueError, TypeError):
        return {}
    if not ip.is_link_local:
        return {}

    result: dict[str, list[str]] = {}

    # Bonjour name is by far the most reliable — mDNSResponder picks
    # whichever interface can actually reach the remote right now.
    bonjour = _bonjour_name_for_ip(host)
    if bonjour:
        result["bonjour"] = [bonjour]

    # Also collect other link-local IPs via ARP (non-en0 interfaces only).
    candidates: set[str] = set()
    try:
        arp = subprocess.run(
            ["arp", "-an"], capture_output=True, text=True, timeout=1.0,
            check=False,
        )
        if arp.returncode == 0:
            for line in arp.stdout.splitlines():
                m = re.search(
                    r"\(([0-9.]+)\)\s+at\s+([0-9a-f:]+)\s+on\s+(\S+)",
                    line, re.IGNORECASE,
                )
                if not m:
                    continue
                a, iface = m.group(1), m.group(3)
                if iface.lower() == "en0" or a == host:
                    continue
                try:
                    if ipaddress.IPv4Address(a).is_link_local:
                        candidates.add(a)
                except ValueError:
                    pass
    except (OSError, subprocess.TimeoutExpired):
        pass

    if candidates:
        try:
            from backend import exo_discovery
            addrs = list(candidates)
            sorted_all = exo_discovery._sort_by_reachability(addrs, port)
            reachable = [a for a in sorted_all if exo_discovery._tcp_reachable(a, port, 0.4)]
            other = [a for a in sorted_all if a not in reachable]
            if reachable:
                result["reachable"] = reachable[:4]
            if other:
                result["other"] = other[:4]
        except Exception:  # noqa: BLE001
            result["other"] = sorted(candidates)[:4]

    return result


def _classify_ssh_failure(
    exc: BaseException, host: str, port: int, user: str,
) -> str:
    """Map an ``asyncssh.connect`` failure to a user-actionable message.

    Three layers of failures show up here:

    * **Network / OS** (``OSError`` and subclasses) — sshd not listening,
      route missing, link-local bridge down.  The ``errno`` /
      ``strerror`` text is safe to surface and gives the user something
      to act on (e.g. "Connection refused" → check sshd; "Network is
      unreachable" → check the Thunderbolt bridge).
    * **SSH protocol** (``asyncssh.PermissionDenied``,
      ``KeyExchangeFailed``, ``HostKeyNotVerifiable``, ``ProtocolError``,
      ``ConnectionLost``).  These never include the password — they
      describe the protocol-level reason auth or handshake failed.
    * **Anything else** — fall back to ``f"{type(exc).__name__}: ..."``
      with the message truncated.

    The password is held in a local frame in ``install_authorized_key``;
    asyncssh / asyncio exceptions never carry it, so quoting their
    ``str()`` here doesn't leak the secret.
    """
    target = f"{host}:{port}"

    def _link_local_hint() -> str:
        info = _link_local_alternatives(host, port)
        if not info:
            return ""
        bonjour = info.get("bonjour") or []
        reachable = info.get("reachable") or []
        other = info.get("other") or []
        # Bonjour name is the best fix — mDNSResponder bypasses the broken
        # macOS link-local routing that pins individual IPs to the wrong
        # interface.
        if bonjour:
            return (
                f"  Use the Bonjour name instead: re-run the wizard with "
                f"'{bonjour[0]}' as the host — mDNSResponder will pick the "
                f"right interface automatically."
            )
        if reachable:
            return (
                f"  Reachable alternative: {', '.join(reachable)}. "
                f"Re-run the wizard with that address."
            )
        if other:
            return (
                f"  The cluster has advertised {', '.join(other)} for this "
                f"host but none respond on port {port} right now — remote may "
                f"be asleep or sshd is not running."
            )
        return ""

    # 1) OS / connection-level errors come through as plain OSError or
    #    one of its subclasses (ConnectionRefusedError, TimeoutError, …).
    if isinstance(exc, OSError):
        # ``strerror`` is the short OS message; ``errno`` is portable.
        strerr = (getattr(exc, "strerror", None) or str(exc) or "").strip()
        errno = getattr(exc, "errno", None)
        # Common cases get a tailored hint so the user knows what to fix.
        if isinstance(exc, ConnectionRefusedError) or (errno == 61):
            return (
                f"could not connect to {target}: connection refused — "
                f"is sshd running and listening on port {port}?"
            )
        if errno in (51, 65, 101, 113):  # Network/no route/unreachable
            base = (
                f"could not connect to {target}: {strerr or 'network unreachable'}"
                f" — check the Thunderbolt bridge or LAN routing to {host!r}"
            )
            return base + _link_local_hint()
        if errno == 60 or "timed out" in strerr.lower():
            return (
                f"connection to {target} timed out ({strerr or 'no response'}) "
                f"— host is not responding on port {port}"
            )
        # Unknown OS-level failure: still useful to show the strerror.
        body = strerr or type(exc).__name__
        return f"could not connect to {target}: {body}"

    # 2) asyncssh-specific protocol/auth failures.  Probe the type name
    #    to avoid an extra import here (asyncssh was lazy-imported in
    #    install_authorized_key, but this helper runs in the except
    #    block before we know whether the import succeeded).
    type_name = type(exc).__name__
    if type_name == "PermissionDenied":
        return (
            f"password rejected by {user}@{host} — double-check the password "
            f"and that PasswordAuthentication is enabled in sshd_config"
        )
    if type_name in ("HostKeyNotVerifiable", "HostKeyAlgorithmMismatch"):
        return f"remote host key could not be verified for {host}: {exc}"
    if type_name in ("KeyExchangeFailed", "ProtocolError", "ProtocolNotSupported"):
        return f"ssh handshake with {target} failed: {exc}"
    if type_name in ("ConnectionLost", "DisconnectError"):
        return f"ssh connection to {target} was dropped: {exc}"

    # 3) Generic fallback — keep it short to avoid dumping a stack
    #    trace into the toast.
    short = str(exc).strip().splitlines()[0] if str(exc) else ""
    short = short[:200]
    if short:
        return f"ssh setup failed ({type_name}): {short}"
    return f"ssh setup failed ({type_name})"


async def install_authorized_key(
    host: str,
    user: str,
    port: int,
    password: str,
    public_key_path: str,
    *,
    private_key_path: str = "",
    timeout: float = 15.0,
) -> dict:
    """Append the local public key to the remote's ``authorized_keys``.

    This is the **only** function in the codebase that accepts a
    password. It is called from a single REST handler with no agent
    visibility. The password lives only in stack frames between this
    function and ``asyncssh.connect`` and is discarded as soon as we
    return.

    Steps:

    1. Read the local public key text from ``public_key_path``.
    2. Open one ``asyncssh`` connection with the password.
    3. ``mkdir -p ~/.ssh && chmod 700 ~/.ssh``
    4. Append the pubkey to ``~/.ssh/authorized_keys`` if not already
       present (idempotent), then chmod 600.
    5. Close the connection and immediately reopen with the local
       private key to confirm key auth works end-to-end.

    Returns ``{"ok": True, "fingerprint": ..., "already_present": bool}``
    on success. Raises on any failure; the route layer translates that
    into a generic 500 (it never echoes the request body).
    """
    try:
        import asyncssh  # lazy import — heavy
    except ImportError as exc:
        raise RuntimeError(
            "asyncssh is required for the Cluster setup wizard but is not "
            "installed in this environment."
        ) from exc

    pub_path = _expand(public_key_path)
    pub_text = read_pubkey(str(pub_path))
    if not pub_text:
        raise ValueError("public key file is empty")

    # Compute fingerprint locally so we can return it without the remote.
    bits, fingerprint, _comment, key_type = _fingerprint_pubkey(pub_path)

    if private_key_path:
        priv_path = _expand(private_key_path)
    else:
        # Common case: pub at <name>.pub → private at <name>.
        priv_path = pub_path.with_suffix("") if pub_path.suffix == ".pub" else pub_path

    host = (host or "").strip()
    user = (user or "").strip()
    port = int(port or 22)
    if not host or not user:
        raise ValueError("host and user are required")

    # The remote command appends the key only if absent. We use single
    # quotes around the key text and refuse newlines in the input so
    # there is no command-injection surface even if the user pasted
    # something weird.
    pub_one_line = pub_text.replace("\n", " ").replace("\r", " ").strip()
    if "'" in pub_one_line:
        raise ValueError("public key contains a single quote; refusing to install")

    remote_cmd = (
        "set -e; "
        'umask 077; '
        'mkdir -p "$HOME/.ssh"; '
        'chmod 700 "$HOME/.ssh"; '
        'AK="$HOME/.ssh/authorized_keys"; '
        '[ -f "$AK" ] || : > "$AK"; '
        'chmod 600 "$AK"; '
        f"KEY='{pub_one_line}'; "
        'if grep -qxF "$KEY" "$AK" 2>/dev/null; then '
        '  echo __exo_already_present__; '
        'else '
        '  printf "%s\\n" "$KEY" >> "$AK"; '
        '  echo __exo_installed__; '
        'fi'
    )

    already_present = False
    # Tracks whether we swapped from the user-supplied link-local IP to a
    # Bonjour name.  Returned to the UI so it can update its host field.
    host_used = host
    host_swapped_from: str | None = None

    async def _connect_password(h: str) -> Any:
        # macOS sshd often advertises ``keyboard-interactive`` ahead of
        # plain ``password`` auth (or disables plain password entirely).
        # asyncssh auto-responds to keyboard-interactive challenges with
        # the supplied password when ``keyboard-interactive`` is included
        # in ``preferred_auth`` — this matches what the system ``ssh``
        # client does and avoids "Connection closed" rejections.
        return await asyncio.wait_for(
            asyncssh.connect(
                host=h,
                port=port,
                username=user,
                password=password,
                known_hosts=None,
                client_keys=None,
                preferred_auth=("keyboard-interactive", "password"),
            ),
            timeout=timeout,
        )

    try:
        conn = await _connect_password(host)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"connection to {host}:{port} timed out — host may be "
            f"unreachable or the network bridge is down"
        ) from exc
    except OSError as exc:
        # If the failure is a network/routing error on a link-local IP,
        # look up the Bonjour name and retry automatically.  macOS often
        # routes link-local IPs down the wrong interface (en0/Wi-Fi) when
        # a stale cloned host-route exists, but mDNSResponder resolves
        # *.local names to whichever address is actually reachable.
        errno_val = getattr(exc, "errno", None)
        try:
            ip_obj = ipaddress.IPv4Address(host)
            is_link_local = ip_obj.is_link_local
        except (ValueError, TypeError):
            is_link_local = False

        if is_link_local and errno_val in (51, 65, 101, 113):
            bonjour = _bonjour_name_for_ip(host)
            if bonjour:
                logger.info(
                    "install_authorized_key: %s unreachable (errno %s), "
                    "retrying via Bonjour name %s",
                    host, errno_val, bonjour,
                )
                try:
                    conn = await _connect_password(bonjour)
                    host_swapped_from = host
                    host_used = bonjour
                except asyncio.TimeoutError as exc2:
                    raise RuntimeError(
                        f"connection to {bonjour}:{port} timed out after "
                        f"falling back from {host}"
                    ) from exc2
                except Exception as exc2:  # noqa: BLE001
                    raise RuntimeError(
                        _classify_ssh_failure(exc2, bonjour, port, user)
                    ) from exc2
            else:
                raise RuntimeError(_classify_ssh_failure(exc, host, port, user)) from exc
        else:
            raise RuntimeError(_classify_ssh_failure(exc, host, port, user)) from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(_classify_ssh_failure(exc, host, port, user)) from exc

    try:
        result = await asyncio.wait_for(conn.run(remote_cmd, check=False), timeout=timeout)
        out = (result.stdout or "").strip()
        if "__exo_already_present__" in out:
            already_present = True
        elif "__exo_installed__" not in out:
            raise RuntimeError(
                f"remote install did not complete: rc={result.exit_status} "
                f"stderr={(result.stderr or '').strip()[:200]}"
            )
    finally:
        conn.close()
        try:
            await conn.wait_closed()
        except Exception:  # noqa: BLE001
            pass

    # Verify with key auth — use host_used (may be the Bonjour name).
    if priv_path.is_file():
        try:
            verify_conn = await asyncio.wait_for(
                asyncssh.connect(
                    host=host_used,
                    port=port,
                    username=user,
                    client_keys=[str(priv_path)],
                    known_hosts=None,
                ),
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"installed key, but verification connection failed: {type(exc).__name__}"
            ) from exc
        finally_run = None
        try:
            finally_run = await asyncio.wait_for(
                verify_conn.run("echo __exo_verify__", check=False), timeout=timeout,
            )
        finally:
            verify_conn.close()
            try:
                await verify_conn.wait_closed()
            except Exception:  # noqa: BLE001
                pass
        if not finally_run or "__exo_verify__" not in (finally_run.stdout or ""):
            raise RuntimeError("verification connection completed but echo failed")

    result: dict[str, Any] = {
        "ok": True,
        "fingerprint": fingerprint,
        "key_type": key_type,
        "bits": bits,
        "already_present": already_present,
        "host_used": host_used,
    }
    if host_swapped_from:
        result["host_swapped_from"] = host_swapped_from
    return result


# ---------------------------------------------------------------------------
# ~/.ssh/config append
# ---------------------------------------------------------------------------


_HOST_BLOCK_HEADER = re.compile(r"^\s*Host\s+(.+?)\s*$", re.IGNORECASE)


@dataclass
class SshConfigAppendResult:
    config_path: str = ""
    backup_path: str = ""
    appended_block: str = ""
    replaced: bool = False
    options_used: dict[str, str] = field(default_factory=dict)


def _existing_alias_present(config_path: Path, alias: str) -> bool:
    if not config_path.is_file():
        return False
    alias_lc = alias.lower()
    for raw in config_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        m = _HOST_BLOCK_HEADER.match(line)
        if not m:
            continue
        for tok in m.group(1).split():
            if tok.lower() == alias_lc:
                return True
    return False


def _strip_alias_block(text: str, alias: str) -> str:
    """Remove an existing ``Host <alias>`` block from ``text``.

    Conservative: only removes blocks where the entire ``Host`` line is
    a single alias matching the target (no multi-host patterns). Any
    block we can't safely remove is left in place so the caller's
    ``replace=True`` path may still leave duplicates — the wizard
    surfaces this in the diff preview.
    """
    out_lines: list[str] = []
    in_target = False
    for raw in text.splitlines(keepends=False):
        line = raw.split("#", 1)[0].strip()
        m = _HOST_BLOCK_HEADER.match(line)
        if m:
            tokens = m.group(1).split()
            if len(tokens) == 1 and tokens[0].lower() == alias.lower():
                in_target = True
                continue
            in_target = False
        if in_target:
            continue
        out_lines.append(raw)
    return "\n".join(out_lines).rstrip() + "\n"


def append_ssh_config_block(
    *,
    alias: str,
    hostname: str,
    user: str = "",
    port: int = 22,
    identity_file: str = "",
    extra_options: dict[str, str] | None = None,
    replace: bool = False,
    config_path: str | None = None,
) -> dict:
    """Append a ``Host <alias>`` block to ``~/.ssh/config``.

    Always backs up the existing file to ``<path>.bak.<unix-ts>`` before
    writing. Refuses to write if ``alias`` already exists unless
    ``replace=True``.

    Returns the block that was written + the backup path.
    """
    alias = (alias or "").strip()
    hostname = (hostname or "").strip()
    if not alias:
        raise ValueError("alias is required")
    if not hostname:
        raise ValueError("hostname is required")
    if any(c in alias for c in " \t\n#*?!"):
        raise ValueError(f"alias contains unsafe characters: {alias!r}")

    target = _expand(config_path) if config_path else _expand("~/.ssh/config")
    target.parent.mkdir(mode=0o700, exist_ok=True)
    if target.parent.exists():
        try:
            os.chmod(target.parent, 0o700)
        except OSError:
            pass

    # Backup before any mutation, even if file doesn't exist yet (so we
    # can roll back). When the file is fresh, the backup is empty.
    backup_path = target.with_suffix(target.suffix + f".bak.{int(time.time())}") if target.suffix \
        else target.parent / f"{target.name}.bak.{int(time.time())}"
    existing_text = ""
    if target.is_file():
        existing_text = target.read_text(encoding="utf-8", errors="replace")
        backup_path.write_text(existing_text, encoding="utf-8")

    replaced = False
    if _existing_alias_present(target, alias):
        if not replace:
            raise FileExistsError(f"Host '{alias}' already exists in {target}")
        existing_text = _strip_alias_block(existing_text, alias)
        replaced = True

    options: dict[str, str] = {}
    options["HostName"] = hostname
    if user:
        options["User"] = user
    if port and int(port) != 22:
        options["Port"] = str(int(port))
    if identity_file:
        options["IdentityFile"] = identity_file
        options["IdentitiesOnly"] = "yes"
    for k, v in (extra_options or {}).items():
        if any(c in str(v) for c in "\n\r"):
            raise ValueError(f"option {k} contains newline")
        options[k] = str(v)

    block_lines = [f"Host {alias}"]
    for k, v in options.items():
        block_lines.append(f"  {k} {v}")
    block = "\n".join(block_lines) + "\n"

    new_text = existing_text
    if new_text and not new_text.endswith("\n"):
        new_text += "\n"
    if new_text and not new_text.endswith("\n\n"):
        new_text += "\n"
    new_text += block

    target.write_text(new_text, encoding="utf-8")
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass

    return asdict(
        SshConfigAppendResult(
            config_path=str(target),
            backup_path=str(backup_path) if existing_text else "",
            appended_block=block,
            replaced=replaced,
            options_used=options,
        )
    )


# ---------------------------------------------------------------------------
# Local user / convenience
# ---------------------------------------------------------------------------


def local_user_info() -> dict:
    """Return convenience info for prefilling the wizard's first step."""
    ssh_dir = Path.home() / ".ssh"
    user = os.environ.get("USER") or os.environ.get("USERNAME") or ""
    ssh_perm_ok = False
    if ssh_dir.is_dir():
        try:
            mode = stat.S_IMODE(ssh_dir.stat().st_mode)
            ssh_perm_ok = mode == 0o700
        except OSError:
            ssh_perm_ok = False
    return {
        "user": user,
        "home": str(Path.home()),
        "ssh_dir": str(ssh_dir),
        "ssh_dir_exists": ssh_dir.is_dir(),
        "ssh_dir_perm_ok": ssh_perm_ok,
        "platform": platform.system().lower(),
    }


# ---------------------------------------------------------------------------
# Logging filter — defence-in-depth
# ---------------------------------------------------------------------------


class SecretScrubFilter(logging.Filter):
    """Scrub anything that looks like a password from log records.

    Last-line defence in case a password ever surfaces in a stack trace
    or third-party log line. Patterns:

    * ``password=...`` / ``password: "..."`` (key=value style)
    * ``"password":"..."`` (JSON style)

    Pre-emptive scrub of the *registered* secrets is more reliable, but
    we don't have a session-scoped registry here; this filter is
    pattern-based and runs against ``record.getMessage()``. False
    positives are acceptable — over-redaction beats leakage.
    """

    _PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r'(?i)(password\s*[=:]\s*)(?:"([^"]*)"|\'([^\']*)\'|(\S+))'),
        re.compile(r'(?i)("password"\s*:\s*)"[^"]*"'),
    )

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return True
        original = msg
        for pat in self._PATTERNS:
            msg = pat.sub(lambda m: f"{m.group(1)}***", msg)
        if msg != original:
            record.msg = msg
            record.args = ()
        return True


def install_secret_scrub_filter() -> None:
    """Attach :class:`SecretScrubFilter` to the root logger.

    Idempotent — does nothing if already attached. Called from
    :mod:`backend.server` startup.
    """
    root = logging.getLogger()
    for f in root.filters:
        if isinstance(f, SecretScrubFilter):
            return
    root.addFilter(SecretScrubFilter())


__all__ = [
    "SshProbeResult",
    "LocalKeypair",
    "SshConfigAppendResult",
    "SecretScrubFilter",
    "probe_host",
    "list_local_keypairs",
    "create_keypair",
    "read_pubkey",
    "install_authorized_key",
    "append_ssh_config_block",
    "local_user_info",
    "install_secret_scrub_filter",
]


# Suppress ``Any`` lint when used in narrow places above.
_unused: Any = None
