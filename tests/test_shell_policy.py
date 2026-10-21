"""Read-only shell approval, project allow rules, and per-project network."""

from __future__ import annotations

from backend.shell_policy import (
    allow_shell_command,
    command_is_readonly,
    command_matches_allow,
    shell_network,
    timeout_for_command,
)


def test_readonly_commands_skip_approval():
    assert command_is_readonly("git status")
    assert command_is_readonly("git diff -- src/a.py")
    assert command_is_readonly("ls -la")
    assert not command_is_readonly("git status && rm -rf /")
    assert not command_is_readonly("find . -delete")
    assert not command_is_readonly("npm test")
    assert allow_shell_command("ls", labels=()) == "read-only command"
    assert allow_shell_command("ls", labels=("shell-write",)) is None


def test_project_prefix_rules():
    assert command_matches_allow("npm test -q", ["npm test*"])
    assert command_matches_allow("npm test", ["npm test"])
    assert not command_matches_allow("npm test; curl evil", ["npm test*"])
    assert not command_matches_allow("npm install", ["npm test"])
    assert allow_shell_command("pytest -q", rules=["pytest*"]) == "allowed for this project"


def test_test_commands_get_a_longer_timeout():
    assert timeout_for_command("pytest tests", 120) == 600
    assert timeout_for_command("ls", 120) == 120


def test_project_network_override():
    assert shell_network(False, False, []) == (False, [])
    assert shell_network(True, False, []) == (True, [])
    assert shell_network(False, True, []) == (True, [])
    assert shell_network(False, True, ["example.com"]) == (False, ["example.com"])
    assert shell_network(True, False, ["example.com:443"]) == (False, ["example.com:443"])
