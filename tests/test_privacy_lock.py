"""Tests for the verifiable airplane-mode privacy lock.

These cover the cross-platform parts of :mod:`backend.privacy_lock`:

* engage / disengage state transitions
* provider allowlist enforcement
* pf template rendering and host:port parsing
* audit log append + tail
* config wiring (PrivacyConfig defaults, allow_loopback / allow_mdns)

macOS-specific ``pf_status`` shells out to ``pfctl`` so it's only
checked for "returns a dict with the right keys" without asserting
content -- the kernel rules are out of scope for an offline test.
"""

from __future__ import annotations

import pathlib

import pytest

from backend.config import AppConfig, PrivacyConfig
from backend import privacy_lock


@pytest.fixture()
def isolated_app_data(tmp_path, monkeypatch):
    """Redirect ``get_app_data_dir`` so audit logs land under tmp_path."""
    monkeypatch.setattr(
        "backend.privacy_lock.get_app_data_dir",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "backend.config.get_app_data_dir",
        lambda: tmp_path,
    )
    return tmp_path


# ── Defaults ----------------------------------------------------------------


def test_privacy_config_defaults_are_locked_down():
    cfg = PrivacyConfig()
    assert cfg.enabled is False
    assert cfg.allow_loopback is True
    assert cfg.allow_mdns is True
    assert cfg.allowed_hosts == []
    assert cfg.local_only_providers == ["mlx", "omlx", "exo"]


def test_appconfig_includes_privacy_field():
    app = AppConfig()
    assert isinstance(app.privacy, PrivacyConfig)


# ── Engagement state -------------------------------------------------------


def test_engage_flips_state_and_stamps_audit_token(isolated_app_data):
    cfg = AppConfig()
    result = privacy_lock.engage(cfg)
    assert result["engaged"] is True
    assert cfg.privacy.enabled is True
    assert cfg.privacy.engaged_at != ""
    assert len(cfg.privacy.audit_token) == 32  # 16 hex bytes


def test_engage_is_idempotent(isolated_app_data):
    cfg = AppConfig()
    first = privacy_lock.engage(cfg)
    second = privacy_lock.engage(cfg)
    assert first["audit_token"] == second["audit_token"]


def test_disengage_clears_flag_but_keeps_history(isolated_app_data):
    cfg = AppConfig()
    privacy_lock.engage(cfg)
    token_before = cfg.privacy.audit_token
    ts_before = cfg.privacy.engaged_at
    res = privacy_lock.disengage(cfg)
    assert cfg.privacy.enabled is False
    assert res["previously_engaged_at"] == ts_before
    assert cfg.privacy.audit_token == token_before


# ── Provider guard ---------------------------------------------------------


def test_enforce_allows_local_when_engaged(isolated_app_data):
    cfg = AppConfig()
    privacy_lock.engage(cfg)
    for prov in ("mlx", "omlx", "exo"):
        privacy_lock.enforce_provider_allowed(prov, cfg)


def test_enforce_blocks_cloud_when_engaged(isolated_app_data):
    cfg = AppConfig()
    privacy_lock.engage(cfg)
    for prov in ("anthropic", "openai", "cohere"):
        with pytest.raises(privacy_lock.PrivacyLockActive):
            privacy_lock.enforce_provider_allowed(prov, cfg)


def test_enforce_noop_when_disengaged(isolated_app_data):
    cfg = AppConfig()
    privacy_lock.enforce_provider_allowed("anthropic", cfg)


def test_blank_allowlist_falls_back_to_defaults(isolated_app_data):
    cfg = AppConfig()
    cfg.privacy.local_only_providers = []
    privacy_lock.engage(cfg)
    privacy_lock.enforce_provider_allowed("mlx", cfg)
    with pytest.raises(privacy_lock.PrivacyLockActive):
        privacy_lock.enforce_provider_allowed("anthropic", cfg)


def test_unknown_provider_is_refused(isolated_app_data):
    cfg = AppConfig()
    privacy_lock.engage(cfg)
    with pytest.raises(privacy_lock.PrivacyLockActive):
        privacy_lock.enforce_provider_allowed("totally-made-up", cfg)


# ── Audit log --------------------------------------------------------------


def test_audit_log_appends_jsonl(isolated_app_data):
    cfg = AppConfig()
    privacy_lock.engage(cfg)
    privacy_lock.disengage(cfg)
    events = privacy_lock.tail_audit(10)
    assert len(events) >= 2
    # tail returns newest first
    assert events[0]["event"] == "disengage"
    assert events[-1]["event"] == "engage"


def test_audit_records_refusals(isolated_app_data):
    cfg = AppConfig()
    privacy_lock.engage(cfg)
    with pytest.raises(privacy_lock.PrivacyLockActive):
        privacy_lock.enforce_provider_allowed("anthropic", cfg)
    events = privacy_lock.tail_audit(10)
    assert any(e["event"] == "refuse_provider" for e in events)


def test_audit_log_failure_is_non_fatal(isolated_app_data, monkeypatch):
    # If the log file is unwritable we must still update state.
    def _broken_open(*a, **kw):
        raise OSError("disk full")

    cfg = AppConfig()
    monkeypatch.setattr(pathlib.Path, "open", _broken_open)
    privacy_lock.engage(cfg)  # should not raise


# ── pf template ------------------------------------------------------------


def test_pf_template_renders_default_policy():
    cfg = AppConfig()
    text = privacy_lock.render_pf_template(cfg)
    assert "block out all" in text
    assert "lo0" in text
    assert "5353" in text


def test_pf_template_includes_allowed_hosts():
    cfg = AppConfig()
    cfg.privacy.allowed_hosts = ["10.0.0.5:52415", "exo-node-2"]
    text = privacy_lock.render_pf_template(cfg)
    assert "10.0.0.5" in text
    assert "exo-node-2" in text
    assert "port 52415" in text


def test_pf_template_omits_loopback_when_disabled():
    cfg = AppConfig()
    cfg.privacy.allow_loopback = False
    text = privacy_lock.render_pf_template(cfg)
    assert "lo0" not in text


def test_pf_install_command_uses_anchor():
    cfg = AppConfig()
    cfg.privacy.pf_anchor = "otto.test"
    cmd = privacy_lock.pf_install_command(cfg)
    assert cmd == "sudo pfctl -a otto.test -f -"


def test_parse_host_port_ipv6():
    host, port = privacy_lock._parse_host_port("[::1]:1234")
    assert host == "::1"
    assert port == 1234


def test_parse_host_port_no_port():
    host, port = privacy_lock._parse_host_port("example.com")
    assert host == "example.com"
    assert port is None


def test_pf_status_returns_dict():
    cfg = AppConfig()
    status = privacy_lock.pf_status(cfg)
    assert "available" in status


# ── Allowlist join helper --------------------------------------------------


def test_join_allowlists_dedupes_and_strips():
    out = privacy_lock.join_allowlists(["a", " b "], ["b", "c"], ["", "  "])
    assert out == ["a", "b", "c"]
