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

One keychain item for everything
--------------------------------

macOS Keychain authorises access **per item** (each ``service + account``
pair prompts separately and carries its own "Always Allow" ACL).  Storing
one secret per item therefore meant the user was re-prompted once per app
secret *and* once per MCP server (Settings, Slack, Discord, OneDrive, …).

To collapse that to a single authorisation prompt, **all** secrets now
live in one keychain item — ``service="otto.vault", account="__otto_vault__"``
— holding a single JSON document::

    {
      "v": 1,
      "app": { "anthropic_api_key": "…", "openai_api_key": "…" },
      "mcp": {
        "slack":              { "SLACK_BOT_TOKEN": "…" },
        "discord":            { "DISCORD_BOT_TOKEN": "…" },
        "microsoft-onedrive": { "MS365_CLIENT_ID": "…",
                                "__auth_bundle__": { … } }
      }
    }

The document is read once and cached in memory (see
:class:`_ConsolidatedStore`); only *mutations* touch the keychain again, so
the Tools-page credential polling no longer re-reads (and never re-prompts).

:class:`CredentialVault` (per-MCP) and :class:`AppSecretVault` (Otto's own
API keys) are thin facades over the same shared store — their public API is
unchanged, so callers elsewhere in the backend need no edits.

The MCP signing key (:mod:`backend.mcp_signer`, ``otto.trust``) is
deliberately **not** folded in here: it has a distinct lifecycle, is only
touched when spawning agent-generated MCPs, and must survive a
"clear all credentials" flow.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")

# The single consolidated keychain item that backs every namespace.
_VAULT_SERVICE = "otto.vault"
_VAULT_ACCOUNT = "__otto_vault__"
_DOC_VERSION = 1

# Reserved account name used to store a per-MCP auth bundle (OAuth device,
# OAuth auth-code, browser-capture) as a nested object inside the ``mcp``
# section.  Hidden from ``list_names`` because it's not a user-facing
# credential — the auth providers in :mod:`backend.auth` own it.
_BUNDLE_NAME = "__auth_bundle__"

# Legacy namespaces read **once** during migration (never written again).
_LEGACY_MCP_PREFIX = "otto.mcp"
_LEGACY_APP_SERVICE = "otto.app"
_LEGACY_APP_BUNDLE_NAME = "__app_secrets__"


class CredentialVaultError(RuntimeError):
    """Raised when a vault operation fails (backend unavailable, etc.)."""


def _service_for(server_id: str) -> str:
    """Validate *server_id* (kept for parity with the old per-service layout)."""
    if not _VALID_NAME_RE.match(server_id):
        raise CredentialVaultError(
            f"Invalid server_id {server_id!r} — must match {_VALID_NAME_RE.pattern}"
        )
    return f"{_LEGACY_MCP_PREFIX}.{server_id}"


def _validate_secret_name(name: str) -> None:
    if not _VALID_NAME_RE.match(name):
        raise CredentialVaultError(
            f"Invalid secret name {name!r} — must match {_VALID_NAME_RE.pattern}"
        )


# ---------------------------------------------------------------------------
# Consolidated store — owns the single keychain item + in-memory cache
# ---------------------------------------------------------------------------


class _ConsolidatedStore:
    """Owns the one keychain item that backs every secret namespace.

    Lazily imports ``keyring`` so the backend still boots when the
    library or an OS backend is unavailable; in that case
    :meth:`available` returns ``False`` and callers fall back to
    plaintext config.

    Reads are served from an in-memory cache after the first load, so a
    running process authorises keychain access at most once.  All writes
    are serialised through :meth:`mutate` under a re-entrant lock so the
    two facades (:class:`CredentialVault` and :class:`AppSecretVault`)
    can safely update different sections of the same document.
    """

    def __init__(self) -> None:
        self._keyring = None
        self._import_error: Optional[str] = None
        self._cache: Optional[dict[str, Any]] = None
        self._loaded = False
        self._lock = threading.RLock()

    # -- keyring plumbing ------------------------------------------------

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
        try:
            self._kr()
            return True
        except CredentialVaultError:
            return False

    def reset_cache(self) -> None:
        """Drop the in-memory cache.

        Called after the keyring backend is swapped (tests) so the next
        access re-reads/re-migrates against the new backend.
        """
        with self._lock:
            self._cache = None
            self._loaded = False

    # -- document I/O ----------------------------------------------------

    @staticmethod
    def _empty_doc() -> dict[str, Any]:
        return {"v": _DOC_VERSION, "app": {}, "mcp": {}}

    def _read_item(self) -> Optional[dict[str, Any]]:
        """Read + parse the consolidated item.

        Returns ``None`` only when the item is genuinely absent or its
        payload is unparseable — **not** when the keychain read itself
        fails (denied / locked).  A backend failure propagates so callers
        never overwrite a good item with an empty one.
        """
        raw = self._kr().get_password(_VAULT_SERVICE, _VAULT_ACCOUNT)
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("vault: consolidated payload corrupt (%s) — rebuilding", exc)
            return None
        if not isinstance(data, dict):
            return None
        for section in ("app", "mcp"):
            if not isinstance(data.get(section), dict):
                data[section] = {}
        data.setdefault("v", _DOC_VERSION)
        return data

    def _write_item(self, doc: dict[str, Any]) -> None:
        payload = json.dumps(doc, separators=(",", ":"))
        try:
            self._kr().set_password(_VAULT_SERVICE, _VAULT_ACCOUNT, payload)
        except CredentialVaultError:
            raise
        except Exception as exc:
            raise CredentialVaultError(f"keychain write failed: {exc}") from exc

    def load(self) -> dict[str, Any]:
        """Return the cached document, reading (or migrating) on first use.

        The returned object is the live cache — read-only consumers must
        not mutate it; use the facade accessors which copy on the way out.
        """
        with self._lock:
            if self._loaded and self._cache is not None:
                return self._cache
            try:
                doc = self._read_item()
            except CredentialVaultError:
                raise
            except Exception as exc:
                raise CredentialVaultError(f"keychain read failed: {exc}") from exc
            if doc is None:
                doc = self._migrate_locked()
            self._cache = doc
            self._loaded = True
            return doc

    def mutate(self, fn: Callable[[dict[str, Any]], None]) -> None:
        """Apply *fn* to a copy of the document and persist it atomically.

        The cache is only updated after a successful write, so a failed
        keychain write leaves the in-memory state untouched.
        """
        with self._lock:
            base = self.load()
            doc = copy.deepcopy(base)
            fn(doc)
            self._write_item(doc)  # raises on failure; cache stays on `base`
            self._cache = doc
            self._loaded = True

    # -- one-time migration from the old per-item layout -----------------

    def _migrate_locked(self) -> dict[str, Any]:
        """Fold legacy per-item entries into a single document (once).

        Reads (never deletes) the old ``otto.app`` bundle, legacy
        per-field ``otto.app`` rows, and per-server ``otto.mcp.*`` items,
        then writes them back as one consolidated item.  The old entries
        are left in place — deleting them would re-trigger the very
        per-item prompts this consolidation removes, and they're harmless
        dead rows once unused.

        Nothing is written when there was nothing to migrate, so a fresh
        install doesn't create (and prompt for) an empty item until the
        user actually stores a secret.
        """
        doc = self._empty_doc()
        try:
            kr = self._kr()
        except CredentialVaultError:
            return doc

        migrated_any = False

        # --- app secrets: consolidated bundle written by recent versions
        try:
            raw = kr.get_password(_LEGACY_APP_SERVICE, _LEGACY_APP_BUNDLE_NAME)
            if raw:
                data = json.loads(raw)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, str) and v:
                            doc["app"][k] = v
                            migrated_any = True
        except Exception as exc:  # noqa: BLE001 — best-effort migration
            logger.debug("vault migrate: app bundle read failed (%s)", exc)

        # --- app secrets: legacy per-field rows (older than the bundle)
        try:
            from backend.config import _SECRET_FIELDS  # lazy: avoid import cycle

            for _path, account in _SECRET_FIELDS:
                if account in doc["app"]:
                    continue
                try:
                    v = kr.get_password(_LEGACY_APP_SERVICE, account)
                except Exception:  # noqa: BLE001
                    v = None
                if v:
                    doc["app"][account] = v
                    migrated_any = True
        except Exception as exc:  # noqa: BLE001
            logger.debug("vault migrate: per-field app rows skipped (%s)", exc)

        # --- per-MCP credentials tracked by the sidecar name index
        try:
            index = _SidecarIndex.load()
            for server_id in index.servers():
                for name in index.names_for(server_id):
                    try:
                        v = kr.get_password(
                            f"{_LEGACY_MCP_PREFIX}.{server_id}", name
                        )
                    except Exception:  # noqa: BLE001
                        v = None
                    if v is None:
                        continue
                    section = doc["mcp"].setdefault(server_id, {})
                    if name == _BUNDLE_NAME:
                        try:
                            parsed = json.loads(v)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(parsed, dict) and parsed:
                            section[name] = parsed
                            migrated_any = True
                    elif v:
                        section[name] = v
                        migrated_any = True
                if server_id in doc["mcp"] and not doc["mcp"][server_id]:
                    doc["mcp"].pop(server_id, None)
        except Exception as exc:  # noqa: BLE001
            logger.debug("vault migrate: mcp rows skipped (%s)", exc)

        if migrated_any:
            try:
                self._write_item(doc)
                logger.info(
                    "vault: migrated legacy keychain entries into consolidated item"
                )
            except CredentialVaultError as exc:
                logger.warning(
                    "vault: consolidated write failed during migration (%s)", exc
                )
        return doc


# The one shared store instance — both facades below delegate to it so
# they read from and write to the same cached document.
_store = _ConsolidatedStore()


class _StoreBackedVault:
    """Mixin exposing the keyring plumbing on the shared store.

    The ``_keyring`` / ``_import_error`` properties let existing tests
    (and :mod:`backend.routes.vault`) keep patching / inspecting the
    backend on the facade object while the real state lives on
    :data:`_store`.  Swapping the backend resets the cache so the next
    access re-reads against it.
    """

    def __init__(self, store: _ConsolidatedStore) -> None:
        self._store = store

    @property
    def _keyring(self):
        return self._store._keyring

    @_keyring.setter
    def _keyring(self, value) -> None:
        self._store._keyring = value
        self._store.reset_cache()

    @property
    def _import_error(self):
        return self._store._import_error

    @_import_error.setter
    def _import_error(self, value) -> None:
        self._store._import_error = value
        self._store.reset_cache()

    def _kr(self):
        return self._store._kr()

    def available(self) -> bool:
        """Return True when the keyring backend is usable on this host."""
        return self._store.available()


class CredentialVault(_StoreBackedVault):
    """Per-MCP credential facade over the shared consolidated store.

    Secrets are keyed by ``(server_id, name)`` and live under the
    ``mcp`` section of the single keychain document.  The public API is
    unchanged from the old per-item implementation.
    """

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def set(self, server_id: str, name: str, value: str) -> None:
        """Store ``value`` under ``(server_id, name)``.

        Empty values are rejected — call :meth:`delete` to clear an entry.
        """
        if not value:
            raise CredentialVaultError(
                "Refusing to store empty value — use delete() to clear."
            )
        _validate_secret_name(name)
        _service_for(server_id)  # validate id

        def _apply(doc: dict[str, Any]) -> None:
            doc["mcp"].setdefault(server_id, {})[name] = value

        self._store.mutate(_apply)
        logger.info("vault: set server=%s name=%s", server_id, name)

    def delete(self, server_id: str, name: str) -> bool:
        """Remove a single entry. Returns True if it existed."""
        _validate_secret_name(name)
        _service_for(server_id)
        existed = False

        def _apply(doc: dict[str, Any]) -> None:
            nonlocal existed
            section = doc["mcp"].get(server_id)
            if section and name in section:
                del section[name]
                existed = True
                if not section:
                    doc["mcp"].pop(server_id, None)

        try:
            self._store.mutate(_apply)
        except CredentialVaultError:
            return False
        if existed:
            logger.info("vault: delete server=%s name=%s", server_id, name)
        return existed

    def delete_all(self, server_id: str) -> int:
        """Remove every entry for *server_id* (static rows + auth bundle).

        Returns the number of entries removed.  Used when the user
        revokes a generated MCP server — config, files, and credentials
        are wiped in one go.
        """
        _service_for(server_id)
        removed = 0

        def _apply(doc: dict[str, Any]) -> None:
            nonlocal removed
            section = doc["mcp"].pop(server_id, None)
            if section:
                removed = len(section)

        try:
            self._store.mutate(_apply)
        except CredentialVaultError:
            return 0
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
        _service_for(server_id)
        try:
            doc = self._store.load()
        except CredentialVaultError as exc:
            logger.warning(
                "vault: read failed server=%s name=%s err=%s", server_id, name, exc
            )
            return None
        value = (doc["mcp"].get(server_id) or {}).get(name)
        return value if isinstance(value, str) else None

    def has(self, server_id: str, name: str) -> bool:
        """Safe to expose to the LLM — boolean only, no value leak."""
        return self.get(server_id, name) is not None

    def list_names(self, server_id: str) -> list[str]:
        """List the secret names registered for *server_id*.

        Derived directly from the consolidated document.  Filters out the
        reserved auth-bundle slot — that's an internal storage detail,
        not a user-facing credential.  Names are safe to expose to the LLM.
        """
        try:
            doc = self._store.load()
        except CredentialVaultError:
            return []
        section = doc["mcp"].get(server_id) or {}
        return sorted(n for n in section if n != _BUNDLE_NAME)

    def list_servers(self) -> list[str]:
        """Return server_ids that have at least one stored secret.

        Includes servers that only have an auth bundle (no per-name rows)
        so the credentials view can still surface them.
        """
        try:
            doc = self._store.load()
        except CredentialVaultError:
            return []
        return sorted(sid for sid, section in doc["mcp"].items() if section)

    # ------------------------------------------------------------------
    # Auth bundles — structured tokens for non-static auth providers
    #
    # Stored as a nested object under the reserved ``__auth_bundle__``
    # key inside the server's ``mcp`` section.  All field-level access
    # goes through :mod:`backend.auth.utils.project_env` in the parent
    # process; the MCP subprocess and the LLM never see the bundle.
    # ------------------------------------------------------------------

    def set_bundle(self, server_id: str, bundle: dict[str, Any]) -> None:
        """Persist an auth bundle (token, refresh, expiry, …) for *server_id*.

        Empty / falsy bundles are rejected — call :meth:`delete_bundle`
        to clear, or :meth:`delete_all` to wipe the whole MCP.
        """
        if not bundle:
            raise CredentialVaultError(
                "Refusing to store empty bundle — use delete_bundle()."
            )
        _service_for(server_id)
        stored = copy.deepcopy(bundle)

        def _apply(doc: dict[str, Any]) -> None:
            doc["mcp"].setdefault(server_id, {})[_BUNDLE_NAME] = stored

        self._store.mutate(_apply)
        logger.info("vault: set_bundle server=%s fields=%s", server_id, sorted(bundle))

    def get_bundle(self, server_id: str) -> Optional[dict[str, Any]]:
        """Read the persisted auth bundle, or ``None`` if not set / corrupt."""
        _service_for(server_id)
        try:
            doc = self._store.load()
        except CredentialVaultError as exc:
            logger.warning(
                "vault: get_bundle failed server=%s err=%s", server_id, exc
            )
            return None
        value = (doc["mcp"].get(server_id) or {}).get(_BUNDLE_NAME)
        if not isinstance(value, dict):
            return None
        return copy.deepcopy(value)

    def has_bundle(self, server_id: str) -> bool:
        """Boolean wrapper safe to expose to the UI / LLM."""
        return self.get_bundle(server_id) is not None

    def delete_bundle(self, server_id: str) -> bool:
        """Drop the auth bundle without touching per-name rows.  True if existed."""
        _service_for(server_id)
        existed = False

        def _apply(doc: dict[str, Any]) -> None:
            nonlocal existed
            section = doc["mcp"].get(server_id)
            if section and _BUNDLE_NAME in section:
                del section[_BUNDLE_NAME]
                existed = True
                if not section:
                    doc["mcp"].pop(server_id, None)

        try:
            self._store.mutate(_apply)
        except CredentialVaultError:
            return False
        if existed:
            logger.info("vault: delete_bundle server=%s", server_id)
        return existed


class AppSecretVault(_StoreBackedVault):
    """Otto's own credentials (LLM/cloud API keys) over the shared store.

    Lives under the ``app`` section of the single keychain document.  The
    set of app secrets is statically known (see
    :data:`backend.config._SECRET_FIELDS`).  Same hard rules apply:
    backend-only, never log values, no REST endpoint that returns a value.
    """

    def set(self, name: str, value: str) -> None:
        """Store ``value`` under ``name``.  Empty values are rejected —
        call :meth:`delete` to clear an entry."""
        if not value:
            raise CredentialVaultError(
                "Refusing to store empty value — use delete() to clear."
            )
        _validate_secret_name(name)

        def _apply(doc: dict[str, Any]) -> None:
            doc["app"][name] = value

        self._store.mutate(_apply)
        logger.info("app_vault: set name=%s", name)

    def get(self, name: str) -> Optional[str]:
        """Return the stored value, or ``None`` if missing.

        **Never expose this method's return value to the LLM.**
        """
        _validate_secret_name(name)
        value = self._store.load()["app"].get(name)
        return value if isinstance(value, str) else None

    def has(self, name: str) -> bool:
        """Boolean wrapper safe to expose to the UI / LLM."""
        return self.get(name) is not None

    def delete(self, name: str) -> bool:
        """Remove a single entry.  Returns True if it existed."""
        _validate_secret_name(name)
        existed = False

        def _apply(doc: dict[str, Any]) -> None:
            nonlocal existed
            if name in doc["app"]:
                del doc["app"][name]
                existed = True

        self._store.mutate(_apply)
        if existed:
            logger.info("app_vault: delete name=%s", name)
        return existed

    # ------------------------------------------------------------------
    # Bundle store — all app secrets in one shot
    #
    # ``set_bundle`` replaces the entire ``app`` section: callers
    # (:meth:`backend.config.AppConfig._route_secrets_to_vault`) always
    # pass the complete authoritative set of app secrets.
    # ------------------------------------------------------------------

    def set_bundle(self, bundle: dict[str, str]) -> None:
        """Persist *bundle* (``{account: value}``) as the whole app section.

        Empty / falsy bundles are rejected — call :meth:`delete_bundle`
        to clear, mirroring the :meth:`set` contract (no silent clearing).
        """
        if not bundle:
            raise CredentialVaultError(
                "Refusing to store empty bundle — use delete_bundle()."
            )
        stored = {k: v for k, v in bundle.items() if isinstance(v, str) and v}

        def _apply(doc: dict[str, Any]) -> None:
            doc["app"] = dict(stored)

        self._store.mutate(_apply)
        logger.info("app_vault: set_bundle fields=%s", sorted(stored))

    def get_bundle(self) -> Optional[dict[str, str]]:
        """Read the app secrets, or ``None`` when none are stored."""
        app = self._store.load()["app"]
        if not app:
            return None
        return {k: v for k, v in app.items() if isinstance(v, str)}

    def delete_bundle(self) -> bool:
        """Clear every app secret.  Returns True if any existed."""
        existed = False

        def _apply(doc: dict[str, Any]) -> None:
            nonlocal existed
            if doc["app"]:
                existed = True
            doc["app"] = {}

        self._store.mutate(_apply)
        if existed:
            logger.info("app_vault: delete_bundle")
        return existed


# ---------------------------------------------------------------------------
# Name-only sidecar index (legacy)
#
# Older versions couldn't enumerate keychain entries portably, so they
# tracked which per-MCP secret names existed in a JSON sidecar under the
# app data dir.  The consolidated document now derives names directly, so
# the sidecar is only read during :meth:`_ConsolidatedStore._migrate_locked`
# to discover which legacy ``otto.mcp.*`` items to fold in.
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
        p = cls._path()
        if not p.exists():
            return cls({})
        try:
            return cls(json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("vault: sidecar parse failed (%s) — starting fresh", exc)
            return cls({})

    def names_for(self, server_id: str) -> list[str]:
        return list(self._data.get(server_id) or [])

    def servers(self) -> list[str]:
        return sorted(self._data.keys())


# ---------------------------------------------------------------------------
# Module-level singletons — the rest of the backend imports these directly.
# ---------------------------------------------------------------------------

vault = CredentialVault(_store)
app_vault = AppSecretVault(_store)


__all__ = [
    "CredentialVault",
    "AppSecretVault",
    "CredentialVaultError",
    "vault",
    "app_vault",
]
