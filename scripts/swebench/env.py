"""Per-instance checkout and Python environment for the SWE-bench runner.

The repository is exported at ``base_commit`` into a fresh git repo with a
single commit, so the agent cannot read the upstream fix from history.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

_TEST_PATH = re.compile(
    r"(^|/)(tests?|testing)/|(^|/)test_[^/]*$|_tests?\.py$|(^|/)conftest\.py$"
)


def is_test_path(path: str) -> bool:
    return bool(_TEST_PATH.search(path))


def sh(cmd: str, cwd: Path | None = None, env: dict | None = None,
       timeout: int = 1800, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd, shell=True, cwd=cwd, env=env, timeout=timeout,
        capture_output=True, text=True, executable="/bin/bash",
    )
    if check and proc.returncode != 0:
        tail = (proc.stdout + proc.stderr)[-1500:]
        raise RuntimeError(f"command failed ({proc.returncode}): {cmd}\n{tail}")
    return proc


class EnvUnavailable(RuntimeError):
    """The instance's Python version (or similar) cannot be set up on this machine."""


@dataclass
class Prepared:
    repo_dir: Path
    base_sha: str
    test_cmd: str
    notes: list[str]
    mode: str = "uv"                 # "uv" (host virtualenv) or "docker" (see docker_env.py)
    container: str | None = None
    image: str | None = None
    pulled: bool = False             # we pulled the image (so it is ours to remove); False if it was already local


def _spec(instance: dict) -> dict:
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    try:
        return MAP_REPO_VERSION_TO_SPECS[instance["repo"]][str(instance["version"])]
    except KeyError:
        return {}


# Repos with compiled extensions. A host virtualenv has to compile them, which needs system
# libraries and tools the Mac may not have (one matplotlib run lost its whole 30 minutes
# to a failed build), so "auto" gives these the prebuilt Docker environment instead.
NATIVE_BUILD_REPOS = {"matplotlib/matplotlib", "scikit-learn/scikit-learn", "astropy/astropy"}

# Top-level import name when it differs from the repo name.
IMPORT_NAMES = {"scikit-learn/scikit-learn": "sklearn", "pytest-dev/pytest": "pytest"}


def import_name(repo: str) -> str:
    return IMPORT_NAMES.get(repo, repo.split("/")[1].replace("-", "_"))


_UV_PYTHONS: dict[str, bool] = {}


def uv_has_python(version: str) -> bool:
    """Whether uv can give this Mac a native build of the Python version.

    Python 3.5-3.7 have no arm64 build, so those instances need Docker.
    """
    if version not in _UV_PYTHONS:
        arch = {"arm64": "aarch64"}.get(platform.machine(), platform.machine())
        out = subprocess.run(["uv", "python", "list", version], capture_output=True, text=True).stdout
        _UV_PYTHONS[version] = any(f"macos-{arch}" in line for line in out.splitlines())
    return _UV_PYTHONS[version]


def wants_docker(env_mode: str, repo: str, python: str) -> bool:
    """Whether to skip the host virtualenv and use the Docker environment from the start."""
    if env_mode == "docker":
        return True
    return env_mode == "auto" and (not uv_has_python(python) or repo in NATIVE_BUILD_REPOS)


def prepare(instance: dict, work: Path, env_mode: str = "auto") -> Prepared:
    """Create ``work/checkouts/<id>`` with the repo at base_commit and its environment.

    env_mode: "uv" builds a host virtualenv, "docker" runs the environment in the
    prebuilt instance image, "auto" uses uv unless uv has no build of the Python version,
    the repo has compiled extensions (NATIVE_BUILD_REPOS), or the host environment turns
    out not to import the project.
    """
    iid = instance["instance_id"]
    spec0 = _spec(instance)
    if wants_docker(env_mode, instance["repo"], spec0.get("python", "3.9")):
        from docker_env import prepare_docker
        return prepare_docker(instance, work, spec0.get("test_cmd", "pytest -rA"), [])
    repo = instance["repo"]
    dest = work / "checkouts" / iid
    mirror = work / "mirrors" / repo.replace("/", "__")
    notes: list[str] = []

    if dest.exists():
        sh(f"rm -rf {shlex.quote(str(dest))}")
    if not mirror.exists():
        mirror.parent.mkdir(parents=True, exist_ok=True)
        sh(f"git clone --bare https://github.com/{repo}.git {shlex.quote(str(mirror))}",
           timeout=3600)
    probe = sh(f"git -C {shlex.quote(str(mirror))} cat-file -e {instance['base_commit']}",
               check=False)
    if probe.returncode != 0:
        sh(f"git -C {shlex.quote(str(mirror))} fetch origin", timeout=3600)

    dest.mkdir(parents=True)
    sh(f"git -C {shlex.quote(str(mirror))} archive {instance['base_commit']} "
       f"| tar -x -C {shlex.quote(str(dest))}", timeout=900)
    sh("git init -q && git add -A && "
       "git -c user.name=bench -c user.email=bench@example.com commit -q -m base",
       cwd=dest)
    (dest / ".git" / "info" / "exclude").write_text(".venv/\n")

    spec = _spec(instance)
    py = spec.get("python", "3.9")
    env = dict(os.environ)
    env["UV_PYTHON_INSTALL_DIR"] = str(work / "pythons")
    try:
        sh(f"uv venv --seed --python {py} .venv", cwd=dest, env=env)
    except RuntimeError as exc:
        raise EnvUnavailable(f"python {py} unavailable: {str(exc)[-200:]}") from exc
    venv_env = dict(env)
    venv_env["VIRTUAL_ENV"] = str(dest / ".venv")
    venv_env["PATH"] = f"{dest / '.venv' / 'bin'}:{env['PATH']}"

    for cmd in spec.get("pre_install") or []:
        if re.match(r"\s*(apt|apt-get|conda|sudo)\b", cmd):
            notes.append("pre_install skipped (system package): " + cmd)
            continue
        cmd = re.sub(r"\bsed -i (?!'')", "sed -i '' ", cmd)  # BSD sed on macOS
        r = sh(cmd, cwd=dest, env=venv_env, check=False)
        if r.returncode:
            notes.append(f"pre_install failed: {cmd}: " + (r.stdout + r.stderr)[-300:])
    pkgs = spec.get("pip_packages") or []
    if pkgs:
        r = sh("python -m pip install " + " ".join(shlex.quote(p) for p in pkgs),
               cwd=dest, env=venv_env, check=False)
        if r.returncode:
            notes.append("pip_packages failed: " + (r.stdout + r.stderr)[-400:])
    if spec.get("packages") == "requirements.txt" and (dest / "requirements.txt").exists():
        r = sh("python -m pip install -r requirements.txt", cwd=dest, env=venv_env, check=False)
        if r.returncode:
            notes.append("requirements.txt failed: " + (r.stdout + r.stderr)[-400:])
    install = spec.get("install")
    if install:
        r = sh(install, cwd=dest, env=venv_env, check=False, timeout=3600)
        if r.returncode:
            notes.append("install failed: " + (r.stdout + r.stderr)[-600:])
    else:
        notes.append("no install spec found for this repo/version")
    # A broken build leaves an environment the agent cannot use. Find out now, not after it
    # has spent its 30 minutes trying to repair it.
    pkg = import_name(repo)
    check = sh(f"python -c 'import {pkg}'", cwd=dest, env=venv_env, check=False)
    if check.returncode != 0:
        why = (check.stderr or check.stdout)[-300:].strip()
        if env_mode == "auto":
            from docker_env import prepare_docker
            return prepare_docker(instance, work, spec0.get("test_cmd", "pytest -rA"),
                                  notes + [f"host environment cannot import {pkg} ({why}); using Docker"])
        notes.append(f"host environment cannot import {pkg}: {why}")
    # Anything the install touched in the tree is part of the baseline, not the patch.
    sh("git add -A && git -c user.name=bench -c user.email=bench@example.com "
       "commit -q --allow-empty -m env", cwd=dest)
    base_sha = sh("git rev-parse HEAD", cwd=dest).stdout.strip()
    return Prepared(dest, base_sha, spec.get("test_cmd", "pytest -rA"), notes)


_HARNESS_FILES = {".execute.sb"}


def _is_scratch(path: str, status: str) -> bool:
    """Files the sandbox or the agent's scratch work leave in the checkout."""
    if path in _HARNESS_FILES or path.startswith(".swe/") or path.endswith(".pyc"):
        return True
    if "/__pycache__/" in "/" + path:
        return True
    return status.startswith("A") and path.startswith("output/")


def extract_patch(repo_dir: Path, base_sha: str) -> str:
    """Diff of source files against the baseline, including new files.

    Test files, sandbox leftovers and new files under ``output/`` are dropped.
    """
    sh("git add -A -N", cwd=repo_dir, check=False)
    rows = sh(f"git diff --name-status {base_sha}", cwd=repo_dir).stdout.splitlines()
    # Binary files (e.g. a .fits the agent's repro script wrote at the repo root) are
    # never part of a source fix and can't be applied as a text patch anyway.
    binary = {line.split("\t", 2)[2] for line in
              sh(f"git diff --numstat {base_sha}", cwd=repo_dir).stdout.splitlines()
              if line.startswith("-\t-\t")}
    keep = []
    for row in rows:
        status, _, path = row.partition("\t")
        if path not in binary and not is_test_path(path) and not _is_scratch(path, status):
            keep.append(path)
    if not keep:
        return ""
    args = " ".join(shlex.quote(n) for n in keep)
    return sh(f"git diff --no-color {base_sha} -- {args}", cwd=repo_dir).stdout


def drop_failed_preds(out: Path) -> list[str]:
    """Forget predictions whose run was skipped or crashed (the harness's fault, not the
    agent's), so the next run does them again. preds.jsonl is backed up first."""
    preds = out / "preds.jsonl"
    kept, dropped = [], []
    for line in preds.read_text().splitlines():
        if not line.strip():
            continue
        iid = json.loads(line)["instance_id"]
        traj = out / "trajectories" / f"{iid}.json"
        status = json.loads(traj.read_text()).get("status") if traj.exists() else None
        (dropped if status in ("skipped_env", "harness_error") else kept).append((iid, line))
    if dropped:
        preds.with_suffix(".jsonl.bak").write_text(preds.read_text())
        preds.write_text("".join(line + "\n" for _, line in kept))
    return [iid for iid, _ in dropped]
