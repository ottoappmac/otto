"""Helpers for discovering candidate exo secondary nodes.

Two independent sources, both surfaced to the Settings UI's "Add remote"
form:

* :func:`parse_ssh_config` — walks the user's ``~/.ssh/config`` (and any
  ``Include`` files) and returns the concrete host aliases it declares.
  This is the primary source: an SSH alias is the credential the user has
  already declared "I can log into this machine non-interactively", so
  these are the ones we know we can ``scp``/``ssh`` to.

* :func:`scan_lan_ssh` — runs a short Bonjour/mDNS browse for
  ``_ssh._tcp`` services on the local network. macOS uses the system
  ``dns-sd`` tool; Linux uses ``avahi-browse``. Hosts surfaced this way
  may or may not be reachable for *this* user — the UI cross-references
  the result with ``parse_ssh_config`` to flag which ones already have a
  known alias.

Both functions are stdlib + system-tool only; no third-party deps. They
are called from :mod:`backend.routes.exo` (REST) only — the deep-agent
tools intentionally do not get this surface (it's a human action).
"""

from __future__ import annotations

import ipaddress
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Union

_IPNetwork = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]

logger = logging.getLogger(__name__)


@dataclass
class SshConfigHost:
    """One concrete host entry parsed from ``~/.ssh/config``.

    ``alias`` is the literal pattern (e.g. ``mini1``) — wildcards
    (``*``, ``?``) are filtered out before this dataclass is built.
    """

    alias: str
    hostname: str = ""
    user: str = ""
    port: int = 22
    identity_file: str = ""
    source_file: str = ""


@dataclass
class LanSshHost:
    """One ``_ssh._tcp`` service announcement seen on the LAN.

    ``addresses`` is the full set of IPs we resolved for the host
    (IPv4 first, then IPv6). ``thunderbolt_addresses`` is the subset
    that falls inside *this* machine's local Thunderbolt-bridge
    subnet — when both Macs are connected via a TB cable, picking one
    of these IPs sends SSH traffic over the TB link rather than Wi-Fi.
    """

    name: str
    hostname: str = ""
    port: int = 22
    addresses: list[str] = field(default_factory=list)
    thunderbolt_addresses: list[str] = field(default_factory=list)
    matches_alias: str = ""


# ---------------------------------------------------------------------------
# ~/.ssh/config parsing
# ---------------------------------------------------------------------------


def _expand(p: str | os.PathLike[str]) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(str(p)))).resolve()


def _resolve_includes(directive_value: str, *, relative_to: Path) -> list[Path]:
    """Expand a single ``Include`` directive value into concrete file paths.

    Per ssh_config(5), include patterns are evaluated relative to
    ``~/.ssh`` (or the directory of the parent file); both absolute and
    glob patterns are accepted.
    """
    raw = directive_value.strip().strip('"').strip("'")
    if not raw:
        return []
    if not os.path.isabs(raw) and not raw.startswith("~"):
        raw = str(relative_to / raw)
    raw = os.path.expanduser(os.path.expandvars(raw))
    matches = sorted(Path("/").glob(raw.lstrip("/")))
    return [m for m in matches if m.is_file()]


_KEY_VALUE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9]*)\s*[=\s]\s*(.+?)\s*$")


def _parse_one(path: Path, *, _seen: set[Path] | None = None) -> Iterable[SshConfigHost]:
    """Parse a single ssh_config file, yielding non-wildcard host entries.

    Recurses into ``Include`` files. ``_seen`` prevents infinite loops on
    cyclic includes.
    """
    seen = _seen if _seen is not None else set()
    if path in seen or not path.is_file():
        return
    seen.add(path)

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("ssh-config: cannot read %s: %s", path, exc)
        return

    current_aliases: list[str] = []
    current_kv: dict[str, str] = {}

    def flush() -> Iterable[SshConfigHost]:
        for alias in current_aliases:
            if any(c in alias for c in "*?!"):
                continue
            if alias.lower() == "localhost":
                continue
            yield SshConfigHost(
                alias=alias,
                hostname=current_kv.get("hostname", "") or alias,
                user=current_kv.get("user", ""),
                port=int(current_kv.get("port", "") or 22),
                identity_file=current_kv.get("identityfile", ""),
                source_file=str(path),
            )

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        m = _KEY_VALUE_RE.match(line)
        if not m:
            continue
        key, value = m.group(1).lower(), m.group(2).strip()

        if key == "host":
            yield from flush()
            current_aliases = value.split()
            current_kv = {}
        elif key == "include":
            yield from flush()
            current_aliases = []
            current_kv = {}
            for inc in _resolve_includes(value, relative_to=path.parent):
                yield from _parse_one(inc, _seen=seen)
        else:
            current_kv[key] = value

    yield from flush()


def parse_ssh_config(path: str | os.PathLike[str] | None = None) -> list[SshConfigHost]:
    """Return concrete host blocks from the user's ``~/.ssh/config``.

    Wildcards (``Host *``) are silently dropped — they can't be used as a
    bootstrap target. Duplicates (same alias from multiple Include files)
    are de-duplicated keeping the first occurrence.
    """
    root = _expand(path) if path else _expand("~/.ssh/config")
    if not root.is_file():
        return []

    seen_aliases: set[str] = set()
    out: list[SshConfigHost] = []
    for entry in _parse_one(root):
        if entry.alias in seen_aliases:
            continue
        seen_aliases.add(entry.alias)
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# LAN mDNS / Bonjour SSH discovery
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Thunderbolt-Bridge subnet detection (macOS)
#
# When two Macs are connected by a Thunderbolt cable, macOS auto-creates a
# ``bridge0`` interface (the "Thunderbolt Bridge") and assigns each side a
# 169.254.x.x/16 IPv4LL address. mDNS advertises the host on every active
# interface, so a ``foo.local`` lookup from the master returns *all* of the
# secondary's IPs — Wi-Fi, Ethernet, and TB. To prefer the TB-side IP for
# SSH, we need to identify which addresses fall inside this machine's local
# TB-bridge subnet.
# ---------------------------------------------------------------------------


_NETSETUP_TB_PORT = re.compile(
    r"^Hardware Port:\s*Thunderbolt Bridge\s*$\s*Device:\s*(\S+)\s*$",
    re.MULTILINE,
)
_IFCONFIG_INET = re.compile(
    r"^\s*inet\s+(?P<addr>\d+\.\d+\.\d+\.\d+)\s+netmask\s+0x(?P<mask>[0-9a-fA-F]+)",
    re.MULTILINE,
)
_IFCONFIG_INET6 = re.compile(
    r"^\s*inet6\s+(?P<addr>[0-9a-fA-F:]+)(?:%\S+)?\s+prefixlen\s+(?P<plen>\d+)",
    re.MULTILINE,
)


def _thunderbolt_bridge_interfaces() -> list[str]:
    """Return the list of interface names that ``networksetup`` labels as
    "Thunderbolt Bridge" on this Mac (typically just ``bridge0``).

    Returns ``[]`` on non-macOS, or when ``networksetup`` is unavailable
    or refuses to authorize (e.g. running as a sandboxed user). The
    result is intentionally cheap — we don't yet check whether the
    interface is up or has an inet address; that's the caller's job.
    """
    if platform.system() != "Darwin" or not shutil.which("networksetup"):
        return []
    try:
        result = subprocess.run(
            ["networksetup", "-listallhardwareports"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if result.returncode != 0:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in _NETSETUP_TB_PORT.finditer(result.stdout):
        dev = m.group(1).strip()
        if dev and dev not in seen:
            seen.add(dev)
            out.append(dev)
    return out


def _ifconfig_subnets(iface: str) -> list[_IPNetwork]:
    """Parse ``ifconfig <iface>`` and return the IPv4 + IPv6 subnets it
    has assigned. Skips link-local IPv6 (``fe80::``) since callers can't
    use it without a scope ID (and matching by subnet there is
    meaningless — every interface has its own ``fe80::/64``).
    """
    if not shutil.which("ifconfig"):
        return []
    try:
        result = subprocess.run(
            ["ifconfig", iface],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if result.returncode != 0:
        return []

    nets: list[_IPNetwork] = []
    for m in _IFCONFIG_INET.finditer(result.stdout):
        addr = m.group("addr")
        try:
            mask_int = int(m.group("mask"), 16)
            prefix = bin(mask_int).count("1")
            net = ipaddress.IPv4Network(f"{addr}/{prefix}", strict=False)
        except (ValueError, ipaddress.AddressValueError):
            continue
        nets.append(net)
    for m in _IFCONFIG_INET6.finditer(result.stdout):
        addr = m.group("addr")
        if addr.lower().startswith("fe80"):
            continue
        try:
            net = ipaddress.IPv6Network(f"{addr}/{m.group('plen')}", strict=False)
        except (ValueError, ipaddress.AddressValueError):
            continue
        nets.append(net)
    return nets


def _local_thunderbolt_subnets() -> list[_IPNetwork]:
    """All IPv4/IPv6 subnets currently bound to a TB-Bridge interface.

    Returns ``[]`` when no TB cable is connected (the bridge exists but
    has no ``inet``), or when not on macOS.
    """
    nets: list[_IPNetwork] = []
    for iface in _thunderbolt_bridge_interfaces():
        nets.extend(_ifconfig_subnets(iface))
    return nets


def _local_thunderbolt_addresses() -> set[str]:
    """All IPv4/IPv6 addresses currently bound to a TB-Bridge interface.

    Used to filter "self" out of ARP-derived peer candidate lists —
    macOS ``arp -an`` happily lists this machine's own bridge0 address
    alongside any actual peers, which would otherwise let the local
    SSH listener masquerade as a reachable remote.
    """
    out: set[str] = set()
    if not shutil.which("ifconfig"):
        return out
    for iface in _thunderbolt_bridge_interfaces():
        try:
            result = subprocess.run(
                ["ifconfig", iface],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        if result.returncode != 0:
            continue
        for m in _IFCONFIG_INET.finditer(result.stdout):
            out.add(m.group("addr"))
        for m in _IFCONFIG_INET6.finditer(result.stdout):
            addr = m.group("addr")
            if not addr.lower().startswith("fe80"):
                out.add(addr)
    return out


def _filter_thunderbolt(
    addresses: Iterable[str], subnets: Iterable[_IPNetwork],
) -> list[str]:
    """Return the subset of ``addresses`` that lie inside any of ``subnets``.

    Order is preserved; non-IP strings are silently skipped.
    """
    nets = list(subnets)
    if not nets:
        return []
    out: list[str] = []
    for a in addresses:
        try:
            ip = ipaddress.ip_address(a)
        except ValueError:
            continue
        for n in nets:
            if ip.version != n.version:
                continue
            if ip in n:
                out.append(a)
                break
    return out


def _tcp_reachable(addr: str, port: int = 22, timeout: float = 0.5) -> bool:
    """Return True if a TCP connection to ``addr:port`` succeeds within ``timeout``.

    Used to probe multiple thunderbolt-candidate IPs and prefer the reachable
    one when a remote Mac advertises more than one link-local address (e.g.
    after a Thunderbolt cable reconnect that assigns a new 169.254.x.x address
    while the old one is still advertised in the mDNS cache).
    """
    try:
        with socket.create_connection((addr, port), timeout=timeout):
            return True
    except OSError:
        return False


def _sort_by_reachability(addresses: list[str], port: int = 22) -> list[str]:
    """Reorder ``addresses`` so that reachable ones float to the top.

    Probes are run in parallel threads so total latency is one probe timeout
    (0.5 s) rather than N × 0.5 s. The relative order within each group
    (reachable / unreachable) is preserved.
    """
    if len(addresses) <= 1:
        return addresses
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(addresses)) as ex:
        results = list(ex.map(lambda a: _tcp_reachable(a, port), addresses))
    reachable = [a for a, ok in zip(addresses, results) if ok]
    unreachable = [a for a, ok in zip(addresses, results) if not ok]
    return reachable + unreachable


def _resolve_addresses(host: str) -> list[str]:
    """Resolve ``host`` to a de-duplicated list of IPv4/IPv6 addresses.

    Uses :func:`socket.getaddrinfo`, which on macOS goes through
    ``mDNSResponder`` and therefore handles ``.local`` names. Returns
    an empty list on any failure — callers treat addresses as advisory
    metadata, never as the only way to reach a host.

    IPv4 addresses are returned first (most useful for SSH); IPv6
    follows. Loopback / link-local-only results are filtered out so we
    don't surface ``::1`` or ``fe80::…`` to the UI.
    """
    if not host:
        return []
    out: list[str] = []
    seen: set[str] = set()
    try:
        infos = socket.getaddrinfo(
            host, None, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP,
        )
    except OSError:
        return []
    # Prefer v4 first.
    for family in (socket.AF_INET, socket.AF_INET6):
        for info in infos:
            if info[0] != family:
                continue
            addr = info[4][0]
            if not addr or addr in seen:
                continue
            if addr.startswith("127.") or addr == "::1":
                continue
            if family == socket.AF_INET6 and addr.lower().startswith("fe80"):
                continue
            seen.add(addr)
            out.append(addr)
    return out


def _scan_macos(timeout: float) -> list[LanSshHost]:
    """``dns-sd -B`` then ``dns-sd -L`` per service to pull host:port.

    ``dns-sd`` streams forever; we Popen + kill after the timeout. The
    ``-B`` output looks like::

        Browsing for _ssh._tcp
        DATE: ---Tue 28 Apr 2026---
         9:00:00.000  Add        2   4 local.               _ssh._tcp.           mini1

    After resolving each name's ``host:port``, we additionally call
    :func:`_resolve_addresses` so the UI can prefer a concrete IP over
    the flaky ``foo.local`` form when picking a host.
    """
    if not shutil.which("dns-sd"):
        return []

    # Phase 1: browse names.
    proc = subprocess.Popen(
        ["dns-sd", "-B", "_ssh._tcp", "local."],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    names: list[str] = []
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
    finally:
        if proc.stdout is not None:
            for line in proc.stdout.read().splitlines():
                # Columns: TIME  Add/Rmv FLAGS IFACE DOMAIN  TYPE  NAME
                parts = line.split(None, 6)
                if len(parts) < 7 or parts[1] != "Add":
                    continue
                name = parts[6].strip()
                if name and name not in names:
                    names.append(name)

    if not names:
        return []

    # Phase 2: resolve each name to host:port.
    out: list[LanSshHost] = []
    for name in names:
        proc = subprocess.Popen(
            ["dns-sd", "-L", name, "_ssh._tcp", "local."],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            proc.wait(timeout=min(timeout, 1.5))
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                proc.kill()
        finally:
            text = proc.stdout.read() if proc.stdout is not None else ""
        m = re.search(
            r"can be reached at\s+(?P<host>\S+):(?P<port>\d+)",
            text,
        )
        if m:
            host = m.group("host").rstrip(".")
            out.append(LanSshHost(
                name=name,
                hostname=host,
                port=int(m.group("port")),
                addresses=_resolve_addresses(host),
            ))
        else:
            out.append(LanSshHost(name=name))
    return out


def _scan_linux(timeout: float) -> list[LanSshHost]:
    """``avahi-browse -tr _ssh._tcp`` resolves names + addresses in one shot."""
    if not shutil.which("avahi-browse"):
        return []
    try:
        result = subprocess.run(
            ["avahi-browse", "-tr", "-p", "--no-db-lookup", "_ssh._tcp"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return []

    out: dict[str, LanSshHost] = {}
    for line in result.stdout.splitlines():
        if not line.startswith("="):
            continue
        # =;eth0;IPv4;mini1;_ssh._tcp;local;mini1.local;192.168.1.20;22;
        cols = line.split(";")
        if len(cols) < 9:
            continue
        name = cols[3]
        host = cols[6]
        addr = cols[7]
        try:
            port = int(cols[8])
        except ValueError:
            continue
        bucket = out.setdefault(name, LanSshHost(name=name, hostname=host, port=port))
        if addr and addr not in bucket.addresses:
            bucket.addresses.append(addr)
    return list(out.values())


def _local_host_identities() -> set[str]:
    """All names this machine answers to (lowercased), so we can filter it
    out of the LAN scan results.

    Bonjour will list us as a candidate node, which is rarely useful and
    leads to ``scp self → self`` confusion. We collect every identity we
    can cheaply discover: the kernel hostname, the FQDN, the macOS
    LocalHostName, and any address-resolved aliases.
    """
    out: set[str] = set()

    def _add(name: str | None) -> None:
        if not name:
            return
        n = name.strip().lower().rstrip(".")
        if not n:
            return
        out.add(n)
        # Also record the "short" form (before first dot) and the
        # ``.local`` variant — Bonjour names commonly carry that suffix.
        short = n.split(".", 1)[0]
        if short:
            out.add(short)
            out.add(f"{short}.local")

    try:
        _add(socket.gethostname())
    except Exception:
        pass
    try:
        _add(socket.getfqdn())
    except Exception:
        pass
    if platform.system() == "Darwin" and shutil.which("scutil"):
        for key in ("LocalHostName", "ComputerName", "HostName"):
            try:
                r = subprocess.run(
                    ["scutil", "--get", key],
                    capture_output=True,
                    text=True,
                    timeout=1.0,
                    check=False,
                )
                if r.returncode == 0:
                    _add(r.stdout.strip())
            except Exception:
                pass
    return out


def scan_lan_ssh(timeout: float = 3.0) -> list[LanSshHost]:
    """Browse the LAN for ``_ssh._tcp`` services and return what's seen.

    Cross-references the result with :func:`parse_ssh_config` so the UI
    can render a "(also in your ~/.ssh/config as <alias>)" hint, and
    filters out *this* machine (no point provisioning ourselves into
    ourselves).

    Returns an empty list (not an error) on platforms without an mDNS
    browse tool, or when the browse times out without seeing anyone.
    """
    sysname = platform.system()
    timeout = max(0.5, min(15.0, float(timeout)))
    try:
        if sysname == "Darwin":
            hosts = _scan_macos(timeout)
        elif sysname == "Linux":
            hosts = _scan_linux(timeout)
        else:
            hosts = []
    except Exception as exc:
        logger.warning("scan_lan_ssh failed: %s", exc)
        hosts = []

    if not hosts:
        return hosts

    tb_subnets = _local_thunderbolt_subnets()

    selves = _local_host_identities()
    if selves:
        def _is_self(h: LanSshHost) -> bool:
            for cand in (h.name, h.hostname, h.hostname.rstrip("."),
                         h.hostname.split(".", 1)[0] if h.hostname else ""):
                if cand and cand.lower() in selves:
                    return True
            return False
        hosts = [h for h in hosts if not _is_self(h)]

    if hosts:
        ssh_cfg = parse_ssh_config()
        by_hostname = {h.hostname.lower(): h for h in ssh_cfg if h.hostname}
        by_alias = {h.alias.lower(): h for h in ssh_cfg}
        for lan in hosts:
            candidates = (
                lan.hostname.lower(),
                lan.hostname.lower().rstrip("."),
                lan.hostname.lower().split(".local", 1)[0],
                lan.name.lower(),
            )
            for c in candidates:
                if not c:
                    continue
                hit = by_hostname.get(c) or by_alias.get(c)
                if hit:
                    lan.matches_alias = hit.alias
                    break
            # Final IP enrichment: if a backend path didn't populate
            # ``addresses`` (or it came back empty), try one more
            # resolution pass through ``mDNSResponder``/``nss-mdns`` so
            # the UI can offer the concrete IP as an alternative to
            # the brittle ``.local`` form.
            if not lan.addresses and lan.hostname:
                lan.addresses = _resolve_addresses(lan.hostname)
            # Identify TB-bridge addresses and float them to the top of
            # ``addresses`` so the UI's "first IPv4" heuristic naturally
            # prefers the TB link when one is up.
            if tb_subnets and lan.addresses:
                lan.thunderbolt_addresses = _filter_thunderbolt(
                    lan.addresses, tb_subnets,
                )
                if lan.thunderbolt_addresses:
                    # When multiple link-local addresses are present (e.g. a
                    # stale mDNS-cached IP alongside the current one after a
                    # cable reconnect), probe each candidate and prefer
                    # reachable ones so the UI picks the live address.
                    if len(lan.thunderbolt_addresses) > 1:
                        lan.thunderbolt_addresses = _sort_by_reachability(
                            lan.thunderbolt_addresses
                        )
                    rest = [a for a in lan.addresses
                            if a not in lan.thunderbolt_addresses]
                    lan.addresses = lan.thunderbolt_addresses + rest
    return hosts


# ---------------------------------------------------------------------------
# Convenience for routes.exo
# ---------------------------------------------------------------------------


def ssh_config_to_dicts() -> list[dict]:
    return [asdict(h) for h in parse_ssh_config()]


def lan_scan_to_dicts(timeout: float = 3.0) -> list[dict]:
    return [asdict(h) for h in scan_lan_ssh(timeout)]


# ---------------------------------------------------------------------------
# Connectivity probe
# ---------------------------------------------------------------------------


def test_ssh(alias: str, *, timeout: float = 6.0) -> dict:
    """Run a non-interactive SSH probe against ``alias`` and report.

    Mirrors the flags ``backend.exo_cli.run_remote`` will use, so a
    ``"ok": true`` here means ``Provision & start`` will at least be able
    to reach the host with key auth.

    Returns ``{ok, return_code, stdout, stderr, hint}`` where ``hint`` is
    a human-readable interpretation of common failure modes
    (``rc=255`` → unreachable / auth, ``rc=127`` → no remote shell, etc.).
    """
    alias = (alias or "").strip()
    if not alias:
        return {"ok": False, "return_code": -1, "stdout": "", "stderr": "",
                "hint": "Empty alias."}
    timeout = max(2.0, min(30.0, float(timeout)))
    try:
        proc = subprocess.run(
            [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", f"ConnectTimeout={int(min(timeout, 10))}",
                "-o", "StrictHostKeyChecking=accept-new",
                alias,
                "echo __exo_ok__",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "return_code": -1,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "hint": (
                f"ssh {alias!r} timed out after {timeout:.0f}s. The host is "
                "unreachable, behind a firewall, or refusing connections."
            ),
        }
    except FileNotFoundError:
        return {"ok": False, "return_code": -1, "stdout": "", "stderr": "",
                "hint": "`ssh` not found on this machine — install OpenSSH client."}

    ok = proc.returncode == 0 and "__exo_ok__" in proc.stdout
    hint = ""
    if not ok:
        rc = proc.returncode
        stderr = (proc.stderr or "").lower()
        if rc == 255:
            if "could not resolve hostname" in stderr or "name or service not known" in stderr:
                hint = (
                    f"SSH cannot resolve {alias!r}. If you picked it from the "
                    "LAN scan, add a `Host` block to ~/.ssh/config that points "
                    "at the resolvable hostname (e.g. `HostName mac.local`)."
                )
            elif "permission denied" in stderr:
                hint = (
                    f"Permission denied — {alias!r} is reachable but key auth "
                    "isn't set up. Add your master's public key to "
                    f"`~/.ssh/authorized_keys` on the remote and retry."
                )
            elif "connection refused" in stderr:
                hint = (
                    f"Connection refused — Remote Login (sshd) is not running "
                    f"on {alias!r}. On macOS: System Settings → General → "
                    "Sharing → Remote Login."
                )
            elif "host key" in stderr or "verification failed" in stderr:
                hint = (
                    f"Host key changed for {alias!r}. Remove the stale entry "
                    "with `ssh-keygen -R <host>` and try again."
                )
            else:
                hint = (
                    f"ssh exited 255 (generic connection failure). "
                    "See stderr for details."
                )
        elif rc == 127:
            hint = "Connected, but the remote shell could not run `echo`."
        else:
            hint = f"ssh exited {rc}."
    return {
        "ok": ok,
        "return_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "hint": hint,
    }


# ---------------------------------------------------------------------------
# Live Thunderbolt-Bridge link snapshot (for the Cluster setup wizard)
# ---------------------------------------------------------------------------


def live_thunderbolt_link(*, mdns_timeout: float = 1.0) -> dict:
    """Return a snapshot of any active Thunderbolt-Bridge link.

    The Cluster setup wizard polls this so it can offer a one-click hint
    like *"Thunderbolt cable detected — peer reachable at 169.254.10.20"*
    when the user is plugging in a fresh secondary Mac.

    Returns ``{"connected": False}`` when no TB-Bridge is up. Otherwise:

    * ``connected: True``
    * ``interface``: the bridge interface name (typically ``"bridge0"``)
    * ``local_subnets``: list of ``"169.254.x.0/16"``-style strings on
      this side of the link.
    * ``peer_candidates``: probable peer IPv4 addresses on the link,
      ordered by live TCP reachability so the working address comes
      first.  Sources merged here:

        - **ARP entries** with a resolved MAC, restricted to a local
          TB subnet.
        - **mDNS-advertised** addresses (``thunderbolt_addresses``
          from a fast LAN scan) — these reflect what the remote
          *thinks* it's reachable on, even when the local ARP cache
          has gone stale (e.g. after a cable bounce or sleep/wake
          cycle).

    * ``reachable_peer``: the first candidate that accepts a TCP
      connection on port 22, or ``None``.

    Best-effort and fast — total runtime is bounded by ``mdns_timeout``
    plus the per-candidate TCP probe (≈0.5 s × candidate count,
    parallelised).  Linux returns a minimal stub (``connected: False``)
    since TB-Bridge networking is macOS-specific.
    """
    subnets = _local_thunderbolt_subnets()
    if not subnets:
        return {"connected": False}

    ifaces = _thunderbolt_bridge_interfaces()
    iface = ifaces[0] if ifaces else ""

    local_subnets = [str(n) for n in subnets]
    # Exclude this machine's own bridge0 addresses from ``peer_candidates``
    # — otherwise the local SSH listener masquerades as a reachable remote.
    local_addrs = _local_thunderbolt_addresses()

    seen: set[str] = set()
    candidates: list[str] = []

    def _accept(ip_s: str) -> None:
        if ip_s in local_addrs or ip_s in seen:
            return
        try:
            ip_v = ipaddress.IPv4Address(ip_s)
        except ValueError:
            return
        if not any(
            isinstance(n, ipaddress.IPv4Network) and ip_v in n for n in subnets
        ):
            return
        seen.add(ip_s)
        candidates.append(ip_s)

    # ── 1) ARP entries with resolved MACs ─────────────────────────
    try:
        arp = subprocess.run(
            ["arp", "-an"],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        arp = None
    if arp and arp.returncode == 0:
        for line in arp.stdout.splitlines():
            m = re.search(
                r"\(([0-9.]+)\)\s+at\s+([0-9a-f:]+)", line, re.IGNORECASE,
            )
            if m:
                _accept(m.group(1))

    # ── 2) mDNS-advertised TB addresses ───────────────────────────
    # When ARP is empty / stale (e.g. just after a bridge flap), the
    # remote may still be advertising over mDNS with the addresses it
    # genuinely believes it's reachable on.  We do a short browse with
    # a tight timeout so we don't slow the wizard down significantly
    # — total live-link snapshot stays under ≈2 seconds.
    if mdns_timeout > 0:
        try:
            for h in scan_lan_ssh(timeout=mdns_timeout):
                for addr in (h.thunderbolt_addresses or []):
                    _accept(addr)
                # Some remotes don't have ``thunderbolt_addresses``
                # populated when running outside the TB-bridge code
                # path — fall back to walking the full address list
                # and let ``_accept``'s subnet check filter for us.
                for addr in (h.addresses or []):
                    _accept(addr)
        except Exception as exc:  # noqa: BLE001
            logger.debug("live_thunderbolt_link mDNS enrichment failed: %s", exc)

    # ── 3) Probe and order ────────────────────────────────────────
    # ``_sort_by_reachability`` does a parallel TCP probe and floats
    # responsive addresses to the front, so the wizard's "first
    # candidate" heuristic naturally lands on a working IP.
    ordered = _sort_by_reachability(candidates) if candidates else []
    reachable = next((a for a in ordered if _tcp_reachable(a)), None)

    return {
        "connected": True,
        "interface": iface,
        "local_subnets": local_subnets,
        "peer_candidates": ordered,
        "reachable_peer": reachable,
    }
