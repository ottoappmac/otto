"""Patch extraction for the SWE-bench runner drops tests and scratch files."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "swebench"))
import env as swe_env  # noqa: E402


def _git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   cwd=cwd, check=True, capture_output=True)


def test_extract_patch_keeps_only_source_changes(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "core.py").write_text("x = 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_core.py").write_text("def test(): pass\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path,
                          capture_output=True, text=True).stdout.strip()

    (tmp_path / "pkg" / "core.py").write_text("x = 2\n")                # the fix
    (tmp_path / "pkg" / "new_mod.py").write_text("y = 1\n")             # legit new source
    (tmp_path / "tests" / "test_core.py").write_text("def test(): 1\n")  # test edit
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "reproduce.py").write_text("print(1)\n")      # scratch
    (tmp_path / ".execute.sb").write_text("(version 1)\n")               # sandbox leftover

    patch = swe_env.extract_patch(tmp_path, base)
    assert "pkg/core.py" in patch and "pkg/new_mod.py" in patch
    assert "test_core.py" not in patch
    assert "reproduce.py" not in patch
    assert ".execute.sb" not in patch
