"""Backend-only credential store backed by the OS keychain.

This module is the **single chokepoint** through which secrets enter and
leave the application.  Design rules — read these before adding any new
function:

* **Backend-only.**  Nothing in this module is callable from a LangChain
  tool, an MCP server, or the LLM.  The only outbound paths for a stored
  value are :func:`CredentialVault.get` (called from
  :mod:`backend.mcp_manager` at subprocess spawn time) and the env-vars
  injected into that subprocess.
* **Never log values.**  Logging at INFO level should reference names
  only.  When debugging, use ``logger.debug("vault: get name=%s found=%s",
  name, value is not None)`` — never the value itself.
* **No ``get`` REST endpoint.**  See :mod:`backend.routes.vault` — the
  routes only expose ``set``, ``delete``, and ``list`` (names).  Adding
  a route that returns a value would defeat the whole point of the vault.
* **Service prefix.**  All entries share the ``otto.mcp`` namespace so
  they're easy to audit and bulk-revoke (Keychain Access → search).

Storage layout
--------------

Each secret is keyed by ``(service, account)`` where::

    service = "otto.mcp.{server_id}"
    account = "{secret_name}"

For example, a Stripe MCP's secret key lives under
``service="otto.mcp.stripe", account="secret_key"``.  The secret name
is what the generated MCP code references via ``os.environ[name]`` when
the subprocess starts — :class:`backend.config.MCPServerConfig` stores
the name in ``required_secrets`` so the manager knows which entries to
hydrate.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


_SERVICE_PREFIX = "otto.mcp"
_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")

# Reserved name used to store the structured auth bundle for non-static
# providers (OAuth device, OAuth auth-code, browser-capture).  The
# bundle is JSON-serialised under this single keychain entry rather
# than spread across multiple per-field rows so token rotation is an
# atomic write.  Hidden from ``list_names`` because it's not a
# user-facing credential — the user never sees / sets / deletes it
# directly; the auth providers in :mod:`backend.auth` own it.
_BUNDLE_NAME = "__auth_bundle__"


class CredentialVaultError(RuntimeError):
    """Raised when a vault operation fails (backend unavailable, etc.)."""


def _service_for(server_id: str) -> str:
    if not _VALID_NAME_RE.match(server_id):
        raise CredentialVaultError(
            f"Invalid server_id {server_id!r} — must match {_VALID_NAME_RE.pattern}"
        )
    return f"{_SERVICE_PREFIX}.{server_id}"


def _validate_secret_name(name: str) -> None:
    if not _VALID_NAME_RE.match(name):
        raise CredentialVaultError(
            f"Invalid secret name {name!r} — must match {_VALID_NAME_RE.pattern}"
        )


class CredentialVault:
    """Thin wrapper around :mod:`keyring`.

    Lazily imports ``keyring`` so the backend can boot even when the
    library isn't installed (the vault simply refuses every operation
    in that case, surfaced as a clear error).
    """

    def __init__(self) -> None:
        self._keyring = None
        self._import_error: Optional[str] = None

    def _kr(self):
        if self._keyring is not None:
            return self._keyring
        if self._import_error is not None:
            raise CredentialVaultError(self._import_error)
        try:
            import keyring  # type: ignore
            self._keyring = keyring
            return keyring
        except Exception as exc:
            self._import_error = (
                f"keyring is not available: {exc}. "
                "Install it with `uv pip install keyring`."
            )
            raise CredentialVaultError(self._import_error) from exc

    def available(self) -> bool:
        """Return True when the keyring backend is usable on this host."""
        try:
            self._kr()
            return True
        except CredentialVaultError:
            return False

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def set(self, server_id: str, name: str, value: str) -> None:
        """Store ``value`` under ``(server_id, name)`` in the OS keychain.

        Empty values are rejected — call :meth:`delete` to clear an entry.
        """
        if not value:
            raise CredentialVaultError(
                "Refusing to store empty value — use delete() to clear."
            )
        _validate_secret_name(name)
        kr = self._kr()
        kr.set_password(_service_for(server_id), name, value)
        logger.info("vault: set server=%s name=%s", server_id, name)

    def delete(self, server_id: str, name: str) -> bool:
        """Remove a single entry. Returns True if it existed."""
        _validate_secret_name(name)
        kr = self._kr()
        try:
            kr.delete_password(_service_for(server_id), name)
            logger.info("vault: delete server=%s name=%s", server_id, name)
            return True
        except Exception:
            return False

    def delete_all(self, server_id: str) -> int:
        """Remove every entry for *server_id*.  Returns count removed.

        Used when the user revokes a generated MCP server — the registry
        wipes config, files, and credentials in one go.  Includes the
        reserved auth bundle if present so OAuth tokens get cleaned up
        alongside paste-in API keys.
        """
        names = self.list_names(server_id)
        removed = 0
        for n in names:
            if self.delete(server_id, n):
                removed += 1
        if self.delete_bundle(server_id):
            removed += 1
        return removed

    # ------------------------------------------------------------------
    # Read paths — only callable from backend code (not exposed via REST)
    # ------------------------------------------------------------------

    def get(self, server_id: str, name: str) -> Optional[str]:
        """Return the stored value, or ``None`` if missing.

        **Never expose this method's return value to the LLM.**  The only
        legitimate caller is :mod:`backend.mcp_manager` building the
        environment dict for an MCP subprocess.
        """
        _validate_secret_name(name)
        kr = self._kr()
        try:
            return kr.get_password(_service_for(server_id), name)
        except Exception as exc:
            logger.warning(
                "vault: read failed server=%s name=%s err=%s",
                server_id, name, exc,
            )
            return None

    def has(self, server_id: str, name: str) -> bool:
        """Safe to expose to the LLM — boolean only, no value leak."""
        return self.get(server_id, name) is not None

    def list_names(self, server_id: str) -> list[str]:
        """List the secret names registered for *server_id*.

        ``keyring`` does not have a portable "list" API — the macOS,
        Linux, and Windows backends all differ.  We track names
        ourselves in a sidecar JSON file under the app data dir.  Names
        are safe to expose to the LLM.

        Filters out the reserved auth-bundle slot — that's an internal
        storage detail, not a user-facing credential.
        """
        return [n for n in _SidecarIndex.load().names_for(server_id)
                if n != _BUNDLE_NAME]

    def list_servers(self) -> list[str]:
        """Return server_ids that have at least one stored secret.

        Includes servers that only have an auth bundle (no per-name
        rows) so the credentials view can still surface them.
        """
        return _SidecarIndex.load().servers()

    # ------------------------------------------------------------------
    # Auth bundles — structured tokens for non-static auth providers
    #
    # Stored as a single JSON blob under the reserved ``__auth_bundle__``
    # name.  All field-level access goes through
    # :mod:`backend.auth.utils.project_env` in the parent process; the
    # MCP subprocess and the LLM never see the bundle directly.
    # ------------------------------------------------------------------

    def set_bundle(self, server_id: str, bundle: dict[str, Any]) -> None:
        """Persist an auth bundle (token, refresh, expiry, …) for *server_id*.

        Empty / falsy bundles are rejected — call :meth:`delete_bundle`
        to clear, or :meth:`delete_all` to wipe the whole MCP.  This
        mirrors the static :meth:`set` contract (no silent clearing).
        """
        if not bundle:
            raise CredentialVaultError(
                "Refusing to store empty bundle — use delete_bundle()."
            )
        kr = self._kr()
        payload = json.dumps(bundle, separators=(",", ":"))
        kr.set_password(_service_for(server_id), _BUNDLE_NAME, payload)
        _SidecarIndex.load().add(server_id, _BUNDLE_NAME)
        logger.info(
            "vault: set_bundle server=%s fields=%s",
            server_id, sorted(bundle),
        )

    def get_bundle(self, server_id: str) -> Optional[dict[str, Any]]:
        """Read the persisted auth bundle, or ``None`` if not set / corrupt.

        Corrupt bundles (anything that isn't valid JSON) are logged and
        treated as missing so a one-off serialisation bug can't lock
        the user out of an MCP — they just get re-prompted to log in.
        """
        kr = self._kr()
        try:
            raw = kr.get_password(_service_for(server_id), _BUNDLE_NAME)
        except Exception as exc:
            logger.warning(
                "vault: get_bundle failed server=%s err=%s", server_id, exc,
            )
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "vault: get_bundle parse failed server=%s err=%s",
                server_id, exc,
            )
            return None
        if not isinstance(data, dict):
            return None
        return data

    def has_bundle(self, server_id: str) -> bool:
        """Boolean wrapper safe to expose to the UI / LLM."""
        return self.get_bundle(server_id) is not None

    def delete_bundle(self, server_id: str) -> bool:
        """Drop the auth bundle without touching per-name rows.  True if existed."""
        kr = self._kr()
        try:
            kr.delete_password(_service_for(server_id), _BUNDLE_NAME)
            _SidecarIndex.load().remove(server_id, _BUNDLE_NAME)
            logger.info("vault: delete_bundle server=%s", server_id)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Name-only sidecar index
#
# ``keyring`` doesn't enumerate stored entries portably, so we maintain
# a lightweight JSON sidecar of {server_id: [names]}.  Values never
# touch the sidecar — only names — so the file is safe to back up,
# diff, and read from any process.  The actual values stay in the OS
# keychain, where they belong.
# ---------------------------------------------------------------------------


class _SidecarIndex:
    """JSON file under app data dir tracking which secret names exist.

    Format::

        {
          "stripe":   ["secret_key", "publishable_key"],
          "slack":    ["access_token", "refresh_token"]
        }
    """

    @classmethod
    def _path(cls):
        from backend.config import get_app_data_dir
        return get_app_data_dir() / "vault_index.json"

    def __init__(self, data: dict[str, list[str]]) -> None:
        self._data = data

    @classmethod
    def load(cls) -> "_SidecarIndex":
        import json
        p = cls._path()
        if not p.exists():
            return cls({})
        try:
            return cls(json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("vault: sidecar parse failed (%s) — starting fresh", exc)
            return cls({})

    def save(self) -> None:
        import json
        p = self._path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")

    def add(self, server_id: str, name: str) -> None:
        names = self._data.setdefault(server_id, [])
        if name not in names:
            names.append(name)
            names.sort()
            self.save()

    def remove(self, server_id: str, name: str) -> None:
        names = self._data.get(server_id) or []
        if name in names:
            names.remove(name)
            if names:
                self._data[server_id] = names
            else:
                self._data.pop(server_id, None)
            self.save()

    def names_for(self, server_id: str) -> list[str]:
        return list(self._data.get(server_id) or [])

    def servers(self) -> list[str]:
        return sorted(self._data.keys())


# Hook the sidecar into mutators so list_names actually returns
# something.  We do this here rather than inside CredentialVault so the
# sidecar stays an internal detail.

_orig_set = CredentialVault.set
_orig_delete = CredentialVault.delete


def _set_with_index(self: CredentialVault, server_id: str, name: str, value: str) -> None:
    _orig_set(self, server_id, name, value)
    _SidecarIndex.load().add(server_id, name)


def _delete_with_index(self: CredentialVault, server_id: str, name: str) -> bool:
    ok = _orig_delete(self, server_id, name)
    if ok:
        _SidecarIndex.load().remove(server_id, name)
    return ok


CredentialVault.set = _set_with_index  # type: ignore[assignment]
CredentialVault.delete = _delete_with_index  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# App-level secret store
#
# The :class:`CredentialVault` above is namespaced per MCP ``server_id``
# (``otto.mcp.{id}``).  Otto's *own* credentials — LLM/cloud API keys,
# AWS keys, HF/LangSmith/oMLX keys — are not tied to an MCP server, so
# they live in a sibling namespace ``otto.app`` with a flat
# ``(service="otto.app", account="{name}")`` layout.
#
# The set of app secrets is statically known (see
# :data:`backend.config._SECRET_FIELDS`), so unlike the MCP vault we do
# **not** keep a sidecar name index.  Same hard rules apply: backend-only,
# never log values, no REST endpoint that returns a value.
# ---------------------------------------------------------------------------


_APP_SERVICE = "otto.app"

# Reserved account used to store every app secret as a single JSON blob.
# macOS Keychain authorises access per item, so writing each secret to its
# own row makes first-run setup pop one authorisation prompt *per secret*
# (~10 prompts).  Consolidating into one entry collapses that to a single
# read + write — at most one prompt.  Mirrors the per-MCP ``__auth_bundle__``
# pattern in :class:`CredentialVault`.
_APP_BUNDLE_NAME = "__app_secrets__"


class AppSecretVault:
    """Flat keychain store for Otto's own credentials (service ``otto.app``).

    Lazily imports ``keyring`` so the backend still boots when the
    library or an OS backend is unavailable; in that case
    :meth:`available` returns ``False`` and callers fall back to
    plaintext config (see :meth:`backend.config.AppConfig.save`).
    """

    def __init__(self) -> None:
        self._keyring = None
        self._import_error: Optional[str] = None

    def _kr(self):
        if self._keyring is not None:
            return self._keyring
        if self._import_error is not None:
            raise CredentialVaultError(self._import_error)
        try:
            import keyring  # type: ignore
            self._keyring = keyring
            return keyring
        except Exception as exc:
            self._import_error = (
                f"keyring is not available: {exc}. "
                "Install it with `uv pip install keyring`."
            )
            raise CredentialVaultError(self._import_error) from exc

    def available(self) -> bool:
        """Return True when the keyring backend is usable on this host."""
        try:
            self._kr()
            return True
        except CredentialVaultError:
            return False

    def set(self, name: str, value: str) -> None:
        """Store ``value`` under ``name``.  Empty values are rejected —
        call :meth:`delete` to clear an entry."""
        if not value:
            raise CredentialVaultError(
                "Refusing to store empty value — use delete() to clear."
            )
        _validate_secret_name(name)
        self._kr().set_password(_APP_SERVICE, name, value)
        logger.info("app_vault: set name=%s", name)

    def get(self, name: str) -> Optional[str]:
        """Return the stored value, or ``None`` if missing.

        **Never expose this method's return value to the LLM.**  The only
        legitimate callers are :mod:`backend.config` (hydrating the
        in-memory ``AppConfig``) and code that already had access to the
        plaintext config.
        """
        _validate_secret_name(name)
        try:
            return self._kr().get_password(_APP_SERVICE, name)
        except CredentialVaultError:
            raise
        except Exception as exc:
            logger.warning("app_vault: read failed name=%s err=%s", name, exc)
            return None

    def has(self, name: str) -> bool:
        """Boolean wrapper safe to expose to the UI / LLM."""
        return self.get(name) is not None

    def delete(self, name: str) -> bool:
        """Remove a single entry.  Returns True if it existed."""
        _validate_secret_name(name)
        try:
            self._kr().delete_password(_APP_SERVICE, name)
            logger.info("app_vault: delete name=%s", name)
            return True
        except CredentialVaultError:
            raise
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Bundle store — all app secrets in a single keychain entry
    #
    # Stored as one JSON blob under the reserved ``__app_secrets__``
    # account so save/load touch a single keychain item.  Same hard
    # rules apply: backend-only, never log values, no REST endpoint that
    # returns a value.
    # ------------------------------------------------------------------

    def set_bundle(self, bundle: dict[str, str]) -> None:
        """Persist *bundle* (``{account: value}``) as one keychain entry.

        Empty / falsy bundles are rejected — call :meth:`delete_bundle`
        to clear, mirroring the :meth:`set` contract (no silent clearing).
        """
        if not bundle:
            raise CredentialVaultError(
                "Refusing to store empty bundle — use delete_bundle()."
            )
        payload = json.dumps(bundle, separators=(",", ":"))
        self._kr().set_password(_APP_SERVICE, _APP_BUNDLE_NAME, payload)
        logger.info("app_vault: set_bundle fields=%s", sorted(bundle))

    def get_bundle(self) -> Optional[dict[str, str]]:
        """Read the consolidated secrets blob, or ``None`` if unset / corrupt.

        Corrupt payloads (not valid JSON, or not a dict) are logged and
        treated as missing so a one-off serialisation bug can't lock the
        user out — they fall back to legacy per-field entries / re-entry.
        """
        try:
            raw = self._kr().get_password(_APP_SERVICE, _APP_BUNDLE_NAME)
        except CredentialVaultError:
            raise
        except Exception as exc:
            logger.warning("app_vault: get_bundle failed err=%s", exc)
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("app_vault: get_bundle parse failed err=%s", exc)
            return None
        if not isinstance(data, dict):
            return None
        return data

    def delete_bundle(self) -> bool:
        """Remove the consolidated secrets blob.  Returns True if it existed."""
        try:
            self._kr().delete_password(_APP_SERVICE, _APP_BUNDLE_NAME)
            logger.info("app_vault: delete_bundle")
            return True
        except CredentialVaultError:
            raise
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Module-level singletons — the rest of the backend imports these directly.
# ---------------------------------------------------------------------------

vault = CredentialVault()
app_vault = AppSecretVault()


__all__ = [
    "CredentialVault",
    "AppSecretVault",
    "CredentialVaultError",
    "vault",
    "app_vault",
]
