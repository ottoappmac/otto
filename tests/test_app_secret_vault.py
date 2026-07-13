"""Tests for vault-backed app credentials.

Otto's own secrets (LLM/cloud API keys, AWS keys, HF/LangSmith/oMLX
keys) are stored in the OS keychain under the ``otto.app`` namespace
instead of plaintext in ``config.json``.  These tests stub out
``keyring`` so they run without a real backend and verify:

* :class:`backend.credential_vault.AppSecretVault` set/get/has/delete.
* :meth:`AppConfig.save` scrubs secrets from disk and routes them to
  the vault; the in-memory model is left untouched.
* :meth:`AppConfig.load` re-hydrates secrets from the vault.
* Legacy plaintext secrets on disk are migrated to the vault and the
  file is scrubbed.
* ``GET /api/settings`` masks every secret; ``PUT /api/settings`` with
  the placeholder preserves the stored value.
* When the keychain is unavailable, secrets fall back to plaintext.
"""

from __future__ import annotations

import json
from typing import Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.config as config_mod
from backend.config import AppConfig
from backend.credential_vault import CredentialVaultError, app_vault


class _FakeKeyring:
    """In-memory keyring stand-in indexed by ``(service, account)``."""

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
def app_data(tmp_path, monkeypatch):
    """Point the app data dir (config.json lives here) at a tmp dir."""
    monkeypatch.setattr(config_mod, "get_app_data_dir", lambda: tmp_path)
    monkeypatch.setattr(config_mod, "_vault_unavailable_warned", False)
    return tmp_path


@pytest.fixture
def fake_app_vault(tmp_path, monkeypatch):
    """Replace the shared vault keyring backend with an in-memory fake."""
    # The one-time migration reads a sidecar under the app data dir — point
    # it at a tmp dir so tests never touch the real keychain index.
    monkeypatch.setattr(config_mod, "get_app_data_dir", lambda: tmp_path)
    fake = _FakeKeyring()
    monkeypatch.setattr(app_vault, "_keyring", fake)
    monkeypatch.setattr(app_vault, "_import_error", None)
    yield fake
    app_vault._keyring = None  # type: ignore[attr-defined]
    app_vault._import_error = None  # type: ignore[attr-defined]


@pytest.fixture
def unavailable_app_vault(monkeypatch):
    """Force app_vault.available() to return False."""
    monkeypatch.setattr(app_vault, "_keyring", None)
    monkeypatch.setattr(app_vault, "_import_error", "keyring is not available")
    yield
    app_vault._keyring = None  # type: ignore[attr-defined]
    app_vault._import_error = None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# AppSecretVault primitives
# ---------------------------------------------------------------------------


def test_app_vault_set_get_has_delete(fake_app_vault):
    assert app_vault.get("anthropic_api_key") is None
    assert app_vault.has("anthropic_api_key") is False

    app_vault.set("anthropic_api_key", "sk-secret")
    assert app_vault.get("anthropic_api_key") == "sk-secret"
    assert app_vault.has("anthropic_api_key") is True
    # Stored inside the single consolidated keychain item.
    raw = fake_app_vault.store[("otto.vault", "__otto_vault__")]
    assert json.loads(raw)["app"]["anthropic_api_key"] == "sk-secret"

    assert app_vault.delete("anthropic_api_key") is True
    assert app_vault.get("anthropic_api_key") is None
    # Deleting a missing entry reports False.
    assert app_vault.delete("anthropic_api_key") is False


def test_app_vault_rejects_empty_value(fake_app_vault):
    with pytest.raises(CredentialVaultError):
        app_vault.set("anthropic_api_key", "")


# ---------------------------------------------------------------------------
# scrub-on-save
# ---------------------------------------------------------------------------


def test_save_scrubs_disk_and_routes_to_vault(app_data, fake_app_vault):
    cfg = AppConfig()
    cfg.llm.anthropic.api_key = "sk-anthropic"
    cfg.llm.openai.api_key = "sk-openai"
    cfg.llm.mlx.hf_token = "hf-token"
    cfg.observability.langsmith.api_key = "ls-key"
    cfg.save()

    on_disk = json.loads((app_data / "config.json").read_text())
    assert on_disk["llm"]["anthropic"]["api_key"] == ""
    assert on_disk["llm"]["openai"]["api_key"] == ""
    assert on_disk["llm"]["mlx"]["hf_token"] == ""
    assert on_disk["observability"]["langsmith"]["api_key"] == ""

    # All secrets are consolidated into a single keychain bundle entry.
    bundle = app_vault.get_bundle()
    assert bundle == {
        "anthropic_api_key": "sk-anthropic",
        "openai_api_key": "sk-openai",
        "hf_token": "hf-token",
        "langsmith_api_key": "ls-key",
    }

    # In-memory model keeps the real values for apply_to_environ().
    assert cfg.llm.anthropic.api_key == "sk-anthropic"


def test_save_writes_single_keychain_item(app_data, fake_app_vault):
    # Regression guard for the "~10 keychain prompts on first install" bug:
    # every app secret must land in exactly one keychain entry so macOS
    # prompts at most once.
    cfg = AppConfig()
    cfg.llm.anthropic.api_key = "sk-anthropic"
    cfg.llm.openai.api_key = "sk-openai"
    cfg.llm.mlx.hf_token = "hf-token"
    cfg.observability.langsmith.api_key = "ls-key"
    cfg.save()

    assert len(fake_app_vault.store) == 1
    assert ("otto.vault", "__otto_vault__") in fake_app_vault.store


def test_save_empty_secret_deletes_vault_entry(app_data, fake_app_vault):
    cfg = AppConfig()
    cfg.llm.anthropic.api_key = "sk-anthropic"
    cfg.save()
    assert app_vault.get_bundle() == {"anthropic_api_key": "sk-anthropic"}

    # Clearing the only secret and re-saving empties the app section.
    cfg.llm.anthropic.api_key = ""
    cfg.save()
    assert app_vault.get_bundle() is None
    raw = fake_app_vault.store[("otto.vault", "__otto_vault__")]
    assert json.loads(raw)["app"] == {}


# ---------------------------------------------------------------------------
# hydrate-on-load
# ---------------------------------------------------------------------------


def test_load_hydrates_secrets_from_vault(app_data, fake_app_vault):
    cfg = AppConfig()
    cfg.llm.anthropic.api_key = "sk-anthropic"
    cfg.omlx.admin_api_key = "omlx-admin"
    cfg.save()

    loaded = AppConfig.load()
    assert loaded.llm.anthropic.api_key == "sk-anthropic"
    assert loaded.omlx.admin_api_key == "omlx-admin"
    # The plaintext file is still scrubbed after a load+resave cycle.
    on_disk = json.loads((app_data / "config.json").read_text())
    assert on_disk["llm"]["anthropic"]["api_key"] == ""


# ---------------------------------------------------------------------------
# legacy plaintext migration
# ---------------------------------------------------------------------------


def test_load_migrates_legacy_plaintext_secret(app_data, fake_app_vault):
    # Simulate an old config written before the vault existed: a plaintext
    # api_key sitting directly in config.json with nothing in the vault.
    data = AppConfig().model_dump(mode="json")
    data["llm"]["anthropic"]["api_key"] = "sk-legacy"
    (app_data / "config.json").write_text(json.dumps(data, indent=2))
    assert app_vault.get_bundle() is None

    loaded = AppConfig.load()
    # Value is now available in-memory...
    assert loaded.llm.anthropic.api_key == "sk-legacy"
    # ...stored in the consolidated vault bundle...
    assert app_vault.get_bundle() == {"anthropic_api_key": "sk-legacy"}
    # ...and scrubbed from disk.
    on_disk = json.loads((app_data / "config.json").read_text())
    assert on_disk["llm"]["anthropic"]["api_key"] == ""


def test_load_migrates_legacy_per_field_keychain_rows(app_data, fake_app_vault):
    # Simulate a version that stored one keychain row per secret (the
    # source of the ~10 first-run prompts). On load these are folded into
    # the consolidated item WITHOUT deleting the old rows (deleting them
    # would re-trigger the per-item keychain prompt we're avoiding).
    fake_app_vault.set_password("otto.app", "anthropic_api_key", "sk-old")
    fake_app_vault.set_password("otto.app", "openai_api_key", "sk-old-openai")
    (app_data / "config.json").write_text(
        json.dumps(AppConfig().model_dump(mode="json"), indent=2)
    )

    loaded = AppConfig.load()
    assert loaded.llm.anthropic.api_key == "sk-old"
    assert loaded.llm.openai.api_key == "sk-old-openai"

    # Consolidated into the single vault item...
    assert app_vault.get_bundle() == {
        "anthropic_api_key": "sk-old",
        "openai_api_key": "sk-old-openai",
    }
    # ...and the legacy per-field rows are left in place (not deleted).
    assert fake_app_vault.store[("otto.app", "anthropic_api_key")] == "sk-old"
    assert fake_app_vault.store[("otto.app", "openai_api_key")] == "sk-old-openai"


# ---------------------------------------------------------------------------
# keychain-unavailable fallback
# ---------------------------------------------------------------------------


def test_save_falls_back_to_plaintext_when_unavailable(app_data, unavailable_app_vault):
    cfg = AppConfig()
    cfg.llm.anthropic.api_key = "sk-plaintext"
    cfg.save()

    on_disk = json.loads((app_data / "config.json").read_text())
    # No keychain — value stays in the file so the app keeps working.
    assert on_disk["llm"]["anthropic"]["api_key"] == "sk-plaintext"

    loaded = AppConfig.load()
    assert loaded.llm.anthropic.api_key == "sk-plaintext"


# ---------------------------------------------------------------------------
# Settings API redaction + placeholder preserve
# ---------------------------------------------------------------------------


@pytest.fixture
def settings_client(app_data, fake_app_vault):
    from backend.routes.settings import router as settings_router

    app = FastAPI()
    app.include_router(settings_router)
    return TestClient(app)


def test_get_settings_masks_all_secrets(app_data, fake_app_vault, settings_client):
    cfg = AppConfig()
    cfg.llm.anthropic.api_key = "sk-anthropic"
    cfg.llm.openai.api_key = "sk-openai"
    cfg.save()

    resp = settings_client.get("/api/settings")
    assert resp.status_code == 200
    body = resp.json()
    assert body["llm"]["anthropic"]["api_key"] == "••••"
    assert body["llm"]["openai"]["api_key"] == "••••"
    # Unset secret comes back blank, not masked.
    assert body["llm"]["openai"]["azure_api_key"] == ""


def test_put_settings_placeholder_preserves_secret(app_data, fake_app_vault, settings_client):
    cfg = AppConfig()
    cfg.llm.anthropic.api_key = "sk-anthropic"
    cfg.save()

    # Round-trip the masked payload back unchanged (placeholders for secrets).
    masked = settings_client.get("/api/settings").json()
    assert masked["llm"]["anthropic"]["api_key"] == "••••"

    resp = settings_client.put("/api/settings", json=masked)
    assert resp.status_code == 200

    # The placeholder must not have overwritten the real stored value.
    assert app_vault.get_bundle()["anthropic_api_key"] == "sk-anthropic"
    assert AppConfig.load().llm.anthropic.api_key == "sk-anthropic"


def test_put_settings_updates_secret_value(app_data, fake_app_vault, settings_client):
    cfg = AppConfig()
    cfg.llm.anthropic.api_key = "sk-old"
    cfg.save()

    masked = settings_client.get("/api/settings").json()
    masked["llm"]["anthropic"]["api_key"] = "sk-new"

    resp = settings_client.put("/api/settings", json=masked)
    assert resp.status_code == 200

    assert app_vault.get_bundle()["anthropic_api_key"] == "sk-new"
    assert AppConfig.load().llm.anthropic.api_key == "sk-new"
