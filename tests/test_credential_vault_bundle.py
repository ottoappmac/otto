"""Unit tests for the auth-bundle storage path on :class:`CredentialVault`.

The bundle API stores OAuth / browser-captured tokens as a single
JSON-serialised entry under the reserved ``__auth_bundle__`` keychain
account, distinct from the per-name rows used by static credentials.
The tests stub out ``keyring`` so they run without an OS keychain
backend and verify:

* ``set_bundle`` / ``get_bundle`` / ``has_bundle`` / ``delete_bundle``
  round-trip a structured dict losslessly.
* ``list_names`` does NOT surface the reserved bundle slot — that's an
  internal detail, not a user-facing credential.
* ``delete_all`` wipes BOTH per-name rows AND the bundle so a server
  uninstall leaves no residue.
* Corrupt bundles are treated as missing rather than crashing the
  manager (defence in depth — a one-off serialisation bug shouldn't
  lock the user out of an MCP).
"""

from __future__ import annotations

import json
from typing import Optional

import pytest

import backend.credential_vault as vault_mod
from backend.credential_vault import vault


class _FakeKeyring:
    """In-memory keyring stand-in (``set_password`` / ``get_password`` /
    ``delete_password``).  Indexed by ``(service, account)`` tuples
    just like the real macOS / Linux backends."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, account: str, value: str) -> None:
        self.store[(service, account)] = value

    def get_password(self, service: str, account: str) -> Optional[str]:
        return self.store.get((service, account))

    def delete_password(self, service: str, account: str) -> None:
        if (service, account) in self.store:
            del self.store[(service, account)]
        else:
            raise KeyError((service, account))


@pytest.fixture
def fake_keyring(monkeypatch, tmp_path):
    """Replace the real keyring backend + sidecar path for the test."""
    fake = _FakeKeyring()
    vault._keyring = fake  # type: ignore[attr-defined]
    vault._import_error = None  # type: ignore[attr-defined]

    # Sidecar lives under the app data dir — point it at tmp_path so
    # tests don't pollute the real ``~/Library/Application Support/Otto``.
    monkeypatch.setattr(vault_mod, "_SidecarIndex",
                        _make_sidecar_index_class(tmp_path))
    yield fake
    vault._keyring = None  # type: ignore[attr-defined]


def _make_sidecar_index_class(tmp_path):
    """Build a fresh sidecar class bound to *tmp_path* so each test
    gets its own clean index file."""
    OriginalSidecar = vault_mod._SidecarIndex

    class _ScopedSidecar(OriginalSidecar):
        @classmethod
        def _path(cls):
            return tmp_path / "vault_index.json"

    return _ScopedSidecar


# ---------------------------------------------------------------------------
# set / get / has / delete round-trip
# ---------------------------------------------------------------------------


def test_set_bundle_round_trips(fake_keyring):
    bundle = {
        "access_token": "abc",
        "refresh_token": "xyz",
        "token_type": "Bearer",
        "expiry_iso": "2030-01-01T00:00:00+00:00",
        "extra": {"vendor_user_id": "u-99"},
    }
    vault.set_bundle("github", bundle)

    loaded = vault.get_bundle("github")
    assert loaded == bundle
    assert vault.has_bundle("github") is True


def test_get_bundle_returns_none_when_missing(fake_keyring):
    assert vault.get_bundle("never-set") is None
    assert vault.has_bundle("never-set") is False


def test_set_bundle_rejects_empty(fake_keyring):
    from backend.credential_vault import CredentialVaultError

    with pytest.raises(CredentialVaultError):
        vault.set_bundle("github", {})


def test_delete_bundle_returns_existed(fake_keyring):
    vault.set_bundle("github", {"access_token": "x"})
    assert vault.delete_bundle("github") is True
    assert vault.delete_bundle("github") is False
    assert vault.get_bundle("github") is None


# ---------------------------------------------------------------------------
# Name listing must hide the reserved slot
# ---------------------------------------------------------------------------


def test_list_names_omits_bundle_slot(fake_keyring):
    vault.set("github", "GITHUB_USER_AGENT", "otto/1.0")
    vault.set_bundle("github", {"access_token": "tok"})
    names = vault.list_names("github")
    assert "GITHUB_USER_AGENT" in names
    assert "__auth_bundle__" not in names


# ---------------------------------------------------------------------------
# delete_all wipes both halves
# ---------------------------------------------------------------------------


def test_delete_all_wipes_static_and_bundle(fake_keyring):
    vault.set("github", "GITHUB_USER_AGENT", "otto/1.0")
    vault.set_bundle("github", {"access_token": "tok"})
    n = vault.delete_all("github")
    assert n == 2
    assert vault.list_names("github") == []
    assert vault.get_bundle("github") is None


# ---------------------------------------------------------------------------
# Corrupt-bundle resilience
# ---------------------------------------------------------------------------


def test_get_bundle_treats_corrupt_payload_as_missing(fake_keyring):
    # Simulate a bug elsewhere having stored non-JSON in the consolidated
    # item — the whole document is unreadable, so every lookup is "missing".
    fake_keyring.set_password("otto.vault", "__otto_vault__", "{not json")
    vault._store.reset_cache()  # type: ignore[attr-defined]
    assert vault.get_bundle("github") is None


def test_get_bundle_ignores_non_object_payload(fake_keyring):
    # A well-formed document whose bundle slot isn't an object is treated
    # as missing rather than crashing the manager.
    fake_keyring.set_password(
        "otto.vault",
        "__otto_vault__",
        json.dumps({"v": 1, "app": {}, "mcp": {"github": {"__auth_bundle__": "a string"}}}),
    )
    vault._store.reset_cache()  # type: ignore[attr-defined]
    assert vault.get_bundle("github") is None
