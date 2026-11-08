"""LiveOutputMiddleware's tee wrapper must keep the command's own exit status."""

import subprocess

import pytest

from backend.safety_middleware import tee_wrap


@pytest.mark.parametrize(
    "command",
    [
        "true",
        "false",
        "ls /definitely/not/here",
        "exit 3",
        "echo hi | grep zz | tail -1",
        "python3 -c 'raise SystemExit(5)'  # trailing comment",
        "cat <<EOF\nheredoc\nEOF",
    ],
)
def test_wrapped_status_matches_unwrapped(tmp_path, command):
    log = tmp_path / "out.log"
    wrapped = tee_wrap(command, str(log), str(log) + ".rc")
    got = subprocess.run(["/bin/sh", "-c", wrapped], capture_output=True, text=True)
    want = subprocess.run(["/bin/sh", "-c", command], capture_output=True, text=True)
    assert got.returncode == want.returncode
    assert got.stdout == log.read_text()


def test_output_reaches_log(tmp_path):
    log = tmp_path / "out.log"
    subprocess.run(["/bin/sh", "-c", tee_wrap("echo a; echo b >&2", str(log), str(log) + ".rc")])
    assert log.read_text().splitlines() == ["a", "b"]
