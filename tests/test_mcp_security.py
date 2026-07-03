"""Tests for the MCP signing + sandbox layer.

The signer (:mod:`backend.mcp_signer`) uses the OS keychain for the
trust key, which isn't necessarily available on CI.  We monkeypatch
the lazy ``_kr()`` accessor to a tiny in-memory implementation so the
HMAC round-trip is exercised without ever calling the real keyring.

The sandbox profile renderer (:mod:`backend.mcp_sandbox`) is pure
text generation -- we assert that the rules we want appear (and that
the rules we don't want, don't).
"""

from __future__ import annotations

import pytest


# ── In-memory keyring stub for the signer ----------------------------------


class _MemoryKeyring:
    """Tiny stand-in for the ``keyring`` module."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, name: str) -> str | None:
        return self.store.get((service, name))

    def set_password(self, service: str, name: str, value: str) -> None:
        self.store[(service, name)] = value

    def delete_password(self, service: str, name: str) -> None:
        self.store.pop((service, name), None)


@pytest.fixture()
def mem_keyring(monkeypatch):
    kr = _MemoryKeyring()
    monkeypatch.setattr("backend.mcp_signer._kr", lambda: kr)
    return kr


# ── Signer round-trip ------------------------------------------------------


def test_sign_and_verify_roundtrip(mem_keyring, tmp_path):
    from backend import mcp_signer

    server_dir = tmp_path / "srv"
    server_dir.mkdir()
    server_file = server_dir / "server.py"
    server_file.write_text("print('hi')\n")
    manifest_file = server_dir / "manifest.json"
    manifest_file.write_text("{}")
    perm_file = server_dir / "permissions.json"
    perm_file.write_text("{}")

    envelope = mcp_signer.write_signature(
        server_id="srv",
        server_dir=server_dir,
        server_file=server_file,
        manifest_file=manifest_file,
        permissions_file=perm_file,
    )
    assert envelope["alg"] == "HMAC-SHA256"
    assert envelope["signature"]

    ok, reason, env = mcp_signer.verify_directory(
        server_id="srv",
        server_dir=server_dir,
        server_file=server_file,
        manifest_file=manifest_file,
        permissions_file=perm_file,
    )
    assert ok, reason
    assert env == envelope


def test_verify_detects_server_tamper(mem_keyring, tmp_path):
    from backend import mcp_signer

    sd = tmp_path / "srv"
    sd.mkdir()
    server_file = sd / "server.py"
    server_file.write_text("a")
    manifest_file = sd / "manifest.json"
    manifest_file.write_text("{}")
    mcp_signer.write_signature(
        server_id="srv",
        server_dir=sd,
        server_file=server_file,
        manifest_file=manifest_file,
        permissions_file=None,
    )
    # Tamper after signing
    server_file.write_text("evil_code")

    ok, reason, _ = mcp_signer.verify_directory(
        server_id="srv",
        server_dir=sd,
        server_file=server_file,
        manifest_file=manifest_file,
        permissions_file=None,
    )
    assert not ok
    assert "server.py" in reason


def test_verify_detects_manifest_tamper(mem_keyring, tmp_path):
    from backend import mcp_signer

    sd = tmp_path / "srv"
    sd.mkdir()
    s = sd / "server.py"
    s.write_text("a")
    m = sd / "manifest.json"
    m.write_text("{\"tools\": []}")
    mcp_signer.write_signature(
        server_id="srv",
        server_dir=sd,
        server_file=s,
        manifest_file=m,
        permissions_file=None,
    )
    m.write_text("{\"tools\": [{\"name\": \"new\"}]}")
    ok, reason, _ = mcp_signer.verify_directory(
        server_id="srv",
        server_dir=sd,
        server_file=s,
        manifest_file=m,
        permissions_file=None,
    )
    assert not ok
    assert "manifest.json" in reason


def test_verify_missing_signature(mem_keyring, tmp_path):
    from backend import mcp_signer

    sd = tmp_path / "srv"
    sd.mkdir()
    server_file = sd / "server.py"
    server_file.write_text("x")
    manifest_file = sd / "manifest.json"
    manifest_file.write_text("{}")
    ok, reason, env = mcp_signer.verify_directory(
        server_id="srv",
        server_dir=sd,
        server_file=server_file,
        manifest_file=manifest_file,
        permissions_file=None,
    )
    assert not ok
    assert "signature" in reason.lower()
    assert env is None


def test_rotate_invalidates_old_signatures(mem_keyring, tmp_path):
    from backend import mcp_signer

    sd = tmp_path / "srv"
    sd.mkdir()
    s = sd / "server.py"
    s.write_text("x")
    m = sd / "manifest.json"
    m.write_text("{}")
    mcp_signer.write_signature(
        server_id="srv",
        server_dir=sd,
        server_file=s,
        manifest_file=m,
        permissions_file=None,
    )

    mcp_signer.rotate_signing_key()
    ok, reason, _ = mcp_signer.verify_directory(
        server_id="srv",
        server_dir=sd,
        server_file=s,
        manifest_file=m,
        permissions_file=None,
    )
    assert not ok
    assert "signature" in reason.lower() or "trust key" in reason.lower()


def test_signed_servers_summary(mem_keyring, tmp_path):
    from backend import mcp_signer

    sd = tmp_path / "alpha"
    sd.mkdir()
    s = sd / "server.py"
    s.write_text("x")
    m = sd / "manifest.json"
    m.write_text("{}")
    mcp_signer.write_signature(
        server_id="alpha",
        server_dir=sd,
        server_file=s,
        manifest_file=m,
        permissions_file=None,
    )
    unsigned = tmp_path / "beta"
    unsigned.mkdir()

    summary = {entry["id"]: entry for entry in mcp_signer.signed_servers([sd, unsigned])}
    assert summary["alpha"]["signature_present"] is True
    assert summary["beta"]["signature_present"] is False
    assert "key_fingerprint" in summary["alpha"]


# ── Permission manifest round-trip -----------------------------------------


def test_permission_manifest_dict_roundtrip():
    from backend.mcp_sandbox import PermissionManifest

    src = PermissionManifest(
        fs_read=["/a"],
        fs_write=["/b"],
        network_hosts=["x.example.com:443"],
        allow_network_all=False,
        env_read=["FOO_TOKEN"],
        sandbox_enabled=True,
    )
    restored = PermissionManifest.from_dict(src.to_dict())
    assert restored == src


def test_permission_manifest_from_dict_handles_garbage():
    from backend.mcp_sandbox import PermissionManifest

    # Missing fields should produce a fully-default manifest.
    pm = PermissionManifest.from_dict({})
    assert pm.fs_read == []
    assert pm.sandbox_enabled is True


def test_permission_manifest_from_dict_handles_non_dict():
    from backend.mcp_sandbox import PermissionManifest

    pm = PermissionManifest.from_dict("not a dict")  # type: ignore[arg-type]
    assert pm == PermissionManifest()


# ── Sandbox profile renderer ----------------------------------------------


def test_sandbox_profile_contains_required_sections(tmp_path):
    from backend.mcp_sandbox import PermissionManifest, render_sandbox_profile

    venv_py = tmp_path / ".venv" / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("")
    profile = render_sandbox_profile(
        mcp_id="testmcp",
        server_dir=tmp_path,
        venv_python=venv_py,
        manifest=PermissionManifest(
            fs_read=["/etc/myconfig"],
            fs_write=["/private/var/log/myapp"],
            network_hosts=["api.example.com:443", "other:8080"],
        ),
    )
    assert "(version 1)" in profile
    assert "(deny default)" in profile
    assert "/etc/myconfig" in profile
    assert "/private/var/log/myapp" in profile
    assert "api.example.com:443" in profile
    assert "other:8080" in profile
    assert "testmcp" in profile


def test_sandbox_profile_allow_network_all(tmp_path):
    from backend.mcp_sandbox import PermissionManifest, render_sandbox_profile

    venv_py = tmp_path / ".venv" / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("")
    profile = render_sandbox_profile(
        mcp_id="m",
        server_dir=tmp_path,
        venv_python=venv_py,
        manifest=PermissionManifest(allow_network_all=True),
    )
    assert "(allow network-outbound)" in profile


def test_sandbox_profile_no_network_when_empty(tmp_path):
    from backend.mcp_sandbox import PermissionManifest, render_sandbox_profile

    venv_py = tmp_path / ".venv" / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("")
    profile = render_sandbox_profile(
        mcp_id="m",
        server_dir=tmp_path,
        venv_python=venv_py,
        manifest=PermissionManifest(),
    )
    assert "(allow network-outbound)" not in profile
    assert "denied by default" in profile


def test_wrap_command_disabled_returns_unchanged(tmp_path):
    from backend.mcp_sandbox import PermissionManifest, wrap_command

    cmd = ["python", "server.py"]
    out = wrap_command(
        command=cmd,
        server_dir=tmp_path,
        manifest=PermissionManifest(sandbox_enabled=False),
    )
    assert out == cmd


def test_wrap_command_returns_unchanged_when_profile_missing(tmp_path, monkeypatch):
    from backend import mcp_sandbox
    from backend.mcp_sandbox import PermissionManifest

    monkeypatch.setattr(mcp_sandbox, "is_supported", lambda: True)
    monkeypatch.setattr(mcp_sandbox, "resolve_sandbox_exec", lambda: "/usr/bin/sandbox-exec")

    cmd = ["python", "server.py"]
    out = mcp_sandbox.wrap_command(
        command=cmd,
        server_dir=tmp_path,  # no sandbox.sb present
        manifest=PermissionManifest(),
    )
    assert out == cmd


def test_wrap_command_wraps_when_profile_present(tmp_path, monkeypatch):
    from backend import mcp_sandbox
    from backend.mcp_sandbox import PermissionManifest

    (tmp_path / "sandbox.sb").write_text("(version 1)")
    monkeypatch.setattr(mcp_sandbox, "is_supported", lambda: True)
    monkeypatch.setattr(mcp_sandbox, "resolve_sandbox_exec", lambda: "/usr/bin/sandbox-exec")

    cmd = ["python", "server.py"]
    out = mcp_sandbox.wrap_command(
        command=cmd,
        server_dir=tmp_path,
        manifest=PermissionManifest(),
    )
    assert out[0] == "/usr/bin/sandbox-exec"
    assert "-f" in out
    assert out[-2:] == ["python", "server.py"]
