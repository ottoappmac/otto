"""Sign and verify agent-generated MCP artifacts.

Threat model
------------

The risk we care about: an attacker (malicious tool description,
prompt-injection chain, hostile MCP that escaped its sandbox) tampers
with a generated server's ``server.py`` or ``permissions.json`` after
:mod:`backend.mcp_builder` has audited and persisted them.  Next time
the user connects the MCP, the tampered code runs with the user's
credentials in its environment.

Mitigation: every generated MCP is signed at build time with a key
that only Otto's backend knows.  At spawn time we recompute the
signature and refuse to start the subprocess if it doesn't match.

Why HMAC-SHA256 (not Ed25519)
-----------------------------

Otto signs and Otto verifies -- there is no third party who needs to
audit the signature without access to the key.  In that model
symmetric signing is sufficient and lets us stay on the Python
standard library (``hmac`` + ``hashlib``) instead of pulling in the
heavy ``cryptography`` dependency.  The signing key never leaves the
OS keychain, never touches LLM context, and is rotated only via
:func:`rotate_signing_key`.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key material -- stored in OS keychain, never in source / logs
# ---------------------------------------------------------------------------


_KEY_SERVICE = "otto.trust"
_KEY_NAME = "mcp_signing_key_v1"
_KEY_BYTES = 32  # 256-bit -- matches HMAC-SHA256 output width


class TrustKeyError(RuntimeError):
    """Raised when the signing key cannot be read or written."""


def _kr():
    """Lazy import of ``keyring``.

    Mirrors the pattern in :mod:`backend.credential_vault`.  We don't
    reuse the vault's own service prefix because the trust key isn't a
    user credential -- conflating them would let a "clear all
    credentials" flow accidentally wipe the signing key and orphan
    every existing signature.
    """
    try:
        import keyring  # type: ignore
        return keyring
    except Exception as exc:  # noqa: BLE001
        raise TrustKeyError(
            f"keyring is not available ({exc}). "
            "MCP signature verification requires the OS keychain."
        ) from exc


def _load_or_create_key() -> bytes:
    kr = _kr()
    raw = kr.get_password(_KEY_SERVICE, _KEY_NAME)
    if raw:
        try:
            decoded = base64.b64decode(raw)
        except (ValueError, TypeError) as exc:
            raise TrustKeyError(
                f"trust key is malformed in keychain: {exc}"
            ) from exc
        if len(decoded) == _KEY_BYTES:
            return decoded
        # Corrupt entry -- fall through to regenerate.
        logger.warning("mcp_signer: discarding malformed key in keychain")

    new_key = secrets.token_bytes(_KEY_BYTES)
    kr.set_password(_KEY_SERVICE, _KEY_NAME, base64.b64encode(new_key).decode("ascii"))
    logger.info("mcp_signer: generated new trust key")
    return new_key


def rotate_signing_key() -> str:
    """Generate a fresh trust key, invalidating every existing signature.

    Returns the base64-encoded fingerprint of the new key (a SHA-256
    hash truncated to 16 hex chars) for surfacing in the UI.  Callers
    are expected to subsequently re-sign every generated MCP -- the
    REST route does this for the user.
    """
    kr = _kr()
    new_key = secrets.token_bytes(_KEY_BYTES)
    kr.set_password(_KEY_SERVICE, _KEY_NAME, base64.b64encode(new_key).decode("ascii"))
    fp = key_fingerprint(new_key)
    logger.warning("mcp_signer: trust key rotated; fingerprint=%s", fp)
    return fp


def key_fingerprint(key: bytes | None = None) -> str:
    """Return a stable 16-character fingerprint of the trust key.

    Safe to surface in logs and the UI -- it leaks ~64 bits of the
    key's identity, not any of the key material itself.  Callers that
    just want "is the key the same as last time?" use this without
    needing to handle key bytes at all.
    """
    if key is None:
        key = _load_or_create_key()
    return hashlib.sha256(key).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Canonical artifact hashing
# ---------------------------------------------------------------------------


def hash_file(path: Path) -> str:
    """SHA-256 of a file's bytes, hex-encoded.

    Returns an empty string if the file doesn't exist so callers can
    surface "missing" distinct from "corrupt" without an extra stat.
    """
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(payload: dict[str, Any]) -> bytes:
    """Canonical JSON bytes for signing.

    Sorted keys, no whitespace, UTF-8 -- the standard pattern.  This
    must match between sign and verify or signatures break silently.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# Signed-bundle shape
# ---------------------------------------------------------------------------


def build_bundle(
    *,
    server_id: str,
    server_path: Path,
    manifest_path: Path,
    permissions_path: Path | None,
) -> dict[str, Any]:
    """Build the dict that gets signed.

    ``permissions_path`` is optional so existing generated servers
    (created before the permissions concept) still produce a valid
    bundle; the field is set to an empty string when the file is
    missing and the verifier treats both representations as equivalent.
    """
    return {
        "v": 1,
        "server_id": server_id,
        "server_sha256": hash_file(server_path),
        "manifest_sha256": hash_file(manifest_path),
        "permissions_sha256": (
            hash_file(permissions_path) if permissions_path is not None else ""
        ),
        "signed_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }


def sign_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Compute the HMAC signature for *bundle* and return the full envelope.

    Output shape::

        {
          "bundle":     {...the dict passed in...},
          "signature":  "<hex MAC>",
          "key_fingerprint": "<16-char>",
          "alg":        "HMAC-SHA256",
        }

    Both halves get persisted to ``signature.json`` next to ``server.py``.
    """
    key = _load_or_create_key()
    mac = hmac.new(key, _canonical_json(bundle), hashlib.sha256).hexdigest()
    return {
        "bundle": bundle,
        "signature": mac,
        "key_fingerprint": key_fingerprint(key),
        "alg": "HMAC-SHA256",
    }


def verify_envelope(envelope: dict[str, Any]) -> tuple[bool, str]:
    """Verify a signed envelope.

    Returns ``(ok, reason)`` -- ``reason`` is empty on success and
    carries a human-readable diagnostic on failure.  Failures never
    raise (callers want to log + show the reason, not catch an
    exception just to ignore it).
    """
    if not isinstance(envelope, dict):
        return False, "envelope is not a dict"
    if envelope.get("alg") != "HMAC-SHA256":
        return False, f"unsupported alg: {envelope.get('alg')!r}"
    bundle = envelope.get("bundle")
    sig = envelope.get("signature")
    if not isinstance(bundle, dict) or not isinstance(sig, str):
        return False, "envelope missing bundle or signature"
    try:
        key = _load_or_create_key()
    except TrustKeyError as exc:
        return False, f"trust key unavailable: {exc}"
    expected = hmac.new(key, _canonical_json(bundle), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return False, "signature mismatch"
    fp = envelope.get("key_fingerprint")
    if fp and fp != key_fingerprint(key):
        return False, "signed with a different trust key (likely rotated)"
    return True, ""


# ---------------------------------------------------------------------------
# High-level: per-MCP sign / verify
# ---------------------------------------------------------------------------


def signature_path(server_dir: Path) -> Path:
    return server_dir / "signature.json"


def write_signature(
    *,
    server_id: str,
    server_dir: Path,
    server_file: Path,
    manifest_file: Path,
    permissions_file: Path | None,
) -> dict[str, Any]:
    """Sign the artifacts in *server_dir* and persist the envelope."""
    bundle = build_bundle(
        server_id=server_id,
        server_path=server_file,
        manifest_path=manifest_file,
        permissions_path=permissions_file,
    )
    envelope = sign_bundle(bundle)
    sig = signature_path(server_dir)
    sig.write_text(json.dumps(envelope, indent=2, sort_keys=True), encoding="utf-8")
    sig.chmod(0o600)
    return envelope


def verify_directory(
    *,
    server_id: str,
    server_dir: Path,
    server_file: Path,
    manifest_file: Path,
    permissions_file: Path | None,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Recompute hashes and check the persisted signature.

    Returns ``(ok, reason, envelope)``.  ``envelope`` is the parsed
    contents of ``signature.json`` when present so callers can log
    when the signature was issued + by which key fingerprint, even on
    failure.
    """
    sig = signature_path(server_dir)
    if not sig.is_file():
        return False, "signature.json is missing", None
    try:
        envelope = json.loads(sig.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"signature.json unreadable: {exc}", None

    bundle = envelope.get("bundle") or {}
    if bundle.get("server_id") != server_id:
        return False, "server_id in signature does not match", envelope

    # Cheap recompute before HMAC -- catches tampering even if the
    # signature itself was carried over verbatim (which the HMAC step
    # would also catch, but doing the hash check first gives a clearer
    # error message about *which* file changed).
    current_server = hash_file(server_file)
    current_manifest = hash_file(manifest_file)
    current_permissions = (
        hash_file(permissions_file) if permissions_file is not None else ""
    )
    if current_server != bundle.get("server_sha256", ""):
        return False, "server.py has changed since it was signed", envelope
    if current_manifest != bundle.get("manifest_sha256", ""):
        return False, "manifest.json has changed since it was signed", envelope
    expected_perm = bundle.get("permissions_sha256", "")
    if current_permissions != expected_perm:
        # Allow the case where neither side has permissions.json -- the
        # old generation path doesn't write one.
        if not (current_permissions == "" and expected_perm == ""):
            return False, "permissions.json has changed since it was signed", envelope

    ok, reason = verify_envelope(envelope)
    return ok, reason, envelope


# ---------------------------------------------------------------------------
# Bulk helpers used by the routes layer
# ---------------------------------------------------------------------------


def signed_servers(server_dirs: Iterable[Path]) -> list[dict[str, Any]]:
    """Summarise every signed MCP for the trust panel.

    Each entry has ``id``, ``signature_present``, ``key_fingerprint``,
    ``signed_at``.  Verification is not performed here (it touches the
    keychain on every call); use :func:`verify_directory` per server
    when the UI actually needs the bool.
    """
    out: list[dict[str, Any]] = []
    for sd in server_dirs:
        sig = signature_path(sd)
        entry: dict[str, Any] = {"id": sd.name, "signature_present": sig.is_file()}
        if sig.is_file():
            try:
                env = json.loads(sig.read_text(encoding="utf-8"))
                bundle = env.get("bundle") or {}
                entry["key_fingerprint"] = env.get("key_fingerprint", "")
                entry["signed_at"] = bundle.get("signed_at", "")
            except (OSError, json.JSONDecodeError):
                entry["error"] = "unreadable signature.json"
        out.append(entry)
    return out


__all__ = [
    "TrustKeyError",
    "key_fingerprint",
    "rotate_signing_key",
    "hash_file",
    "hash_text",
    "build_bundle",
    "sign_bundle",
    "verify_envelope",
    "signature_path",
    "write_signature",
    "verify_directory",
    "signed_servers",
]
