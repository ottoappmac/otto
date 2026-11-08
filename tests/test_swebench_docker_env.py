"""Docker-backed agent environments: the broker protocol, patch filter and routing (no Docker)."""

import asyncio
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "swebench"))
import docker_env  # noqa: E402
import env as swe_env  # noqa: E402


def _git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   cwd=cwd, check=True, capture_output=True)


def test_image_name_matches_official_naming():
    assert docker_env.image_name("django__django-13158") == \
        "swebench/sweb.eval.x86_64.django_1776_django-13158:latest"


def test_runner_is_installed_executable_with_exchange_folders(tmp_path):
    docker_env.install_runner(tmp_path)
    run = tmp_path / ".swe" / "run"
    assert run.exists() and run.stat().st_mode & 0o111
    for sub in ("req", "res", "scratch"):
        assert (tmp_path / ".swe" / sub).is_dir()


def test_broker_round_trip_through_the_real_run_script(tmp_path, monkeypatch):
    calls = []

    async def fake_exec(container, cmd, timeout=docker_env.EXEC_TIMEOUT):
        calls.append((container, cmd))
        return f"ran: {cmd}\n", 3

    monkeypatch.setattr(docker_env, "_exec", fake_exec)
    docker_env.install_runner(tmp_path)

    async def scenario():
        stop = asyncio.Event()
        server = asyncio.create_task(docker_env.serve(tmp_path, "ctr-1", stop))
        proc = await asyncio.create_subprocess_exec(
            str(tmp_path / ".swe" / "run"), "python", "-m", "pytest", "-x",
            stdout=asyncio.subprocess.PIPE)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=20)
        stop.set()
        await server
        return out.decode(), proc.returncode

    out, rc = asyncio.run(scenario())
    assert out == "ran: python -m pytest -x\n"
    assert rc == 3                                   # the container's exit code is passed back
    assert calls == [("ctr-1", "python -m pytest -x")]
    assert not list((tmp_path / ".swe" / "res").iterdir())   # answer files are cleaned up


def test_broker_only_runs_in_the_fixed_container(tmp_path, monkeypatch):
    seen = []

    async def fake_exec(container, cmd, timeout=docker_env.EXEC_TIMEOUT):
        seen.append(container)
        return "", 0

    monkeypatch.setattr(docker_env, "_exec", fake_exec)
    docker_env.install_runner(tmp_path)

    async def scenario():
        stop = asyncio.Event()
        server = asyncio.create_task(docker_env.serve(tmp_path, "fixed", stop))
        for cmd in ("echo a", "echo b"):
            p = await asyncio.create_subprocess_exec(str(tmp_path / ".swe" / "run"), cmd)
            await asyncio.wait_for(p.wait(), timeout=20)
        stop.set()
        await server

    asyncio.run(scenario())
    assert seen == ["fixed", "fixed"]


def test_broker_is_a_noop_for_host_environments(tmp_path):
    prepared = swe_env.Prepared(tmp_path, "sha", "pytest", [])      # mode defaults to "uv"

    async def scenario():
        async with docker_env.broker(prepared):
            return "ok"

    assert asyncio.run(scenario()) == "ok"
    assert prepared.mode == "uv" and prepared.container is None


def test_patch_drops_runner_scratch_and_bytecode(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "core.py").write_text("x = 1\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path,
                          capture_output=True, text=True).stdout.strip()
    (tmp_path / "pkg" / "core.py").write_text("x = 2\n")
    docker_env.install_runner(tmp_path)
    (tmp_path / ".swe" / "scratch" / "repro.py").write_text("print(1)\n")
    (tmp_path / "pkg" / "__pycache__").mkdir()
    (tmp_path / "pkg" / "__pycache__" / "core.cpython-36.pyc").write_bytes(b"\0")
    patch = swe_env.extract_patch(tmp_path, base)
    assert "pkg/core.py" in patch
    for junk in (".swe", "repro.py", ".pyc", "__pycache__"):
        assert junk not in patch


def test_auto_mode_uses_docker_only_when_uv_has_no_build(monkeypatch):
    monkeypatch.setattr(swe_env, "_spec", lambda i: {"python": i["py"], "test_cmd": "pytest"})
    monkeypatch.setattr(swe_env, "uv_has_python", lambda v: v not in ("3.5", "3.6", "3.7"))
    chosen = []

    def fake_prepare_docker(instance, work, test_cmd, notes):
        chosen.append(instance["instance_id"])
        return swe_env.Prepared(work, "sha", test_cmd, notes, mode="docker", container="c", image="i")

    monkeypatch.setattr(docker_env, "prepare_docker", fake_prepare_docker)
    got = swe_env.prepare({"instance_id": "a", "py": "3.6", "repo": "r"}, Path("/tmp/x"), "auto")
    assert got.mode == "docker" and chosen == ["a"]
    got2 = swe_env.prepare({"instance_id": "b", "py": "3.6", "repo": "r"}, Path("/tmp/x"), "docker")
    assert got2.mode == "docker" and chosen == ["a", "b"]    # forced, even if uv could do it


def test_retry_failed_drops_only_skipped_and_crashed(tmp_path):
    import json
    rows = [("a", "done"), ("b", "skipped_env"), ("c", "harness_error"), ("d", "error"), ("e", None)]
    (tmp_path / "trajectories").mkdir()
    lines = []
    for iid, status in rows:
        lines.append(json.dumps({"instance_id": iid, "model_patch": ""}))
        if status:
            (tmp_path / "trajectories" / f"{iid}.json").write_text(json.dumps({"status": status}))
    (tmp_path / "preds.jsonl").write_text("\n".join(lines) + "\n")
    assert swe_env.drop_failed_preds(tmp_path) == ["b", "c"]
    left = [json.loads(l)["instance_id"] for l in (tmp_path / "preds.jsonl").read_text().splitlines()]
    assert left == ["a", "d", "e"]                    # a model failure ("error") stays recorded
    assert (tmp_path / "preds.jsonl.bak").exists()


def test_task_text_points_scratch_scripts_at_the_mapped_project_not_output():
    text = docker_env.task_text("./tests/runtests.py", "django__django-13158")
    assert "/links/django__django-13158/.swe/scratch/" in text
    assert "NOT /output/" in text and "./.swe/run" in text


def test_event_log_appends_timestamped_frames_and_survives_no_path(tmp_path):
    import json
    import pytest
    run = pytest.importorskip("run")           # needs httpx and websockets
    path = tmp_path / "x.events.jsonl"
    run.log_event(path, {"type": "tool_call", "content": "execute"})
    run.log_event(path, {"type": "tool_result", "content": "ok"})
    run.log_event(None, {"type": "ignored"})   # no path: silently nothing
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert [r["type"] for r in rows] == ["tool_call", "tool_result"]
    assert all(isinstance(r["ts"], float) for r in rows) and rows[0]["ts"] <= rows[1]["ts"]


def test_compiled_extension_repos_go_to_docker_in_auto_mode(monkeypatch):
    monkeypatch.setattr(swe_env, "uv_has_python", lambda v: True)          # uv could build the Python
    for repo in ("matplotlib/matplotlib", "scikit-learn/scikit-learn", "astropy/astropy"):
        assert swe_env.wants_docker("auto", repo, "3.9") is True
    for repo in ("django/django", "pallets/flask", "sympy/sympy", "pydata/xarray"):
        assert swe_env.wants_docker("auto", repo, "3.9") is False


def test_explicit_modes_override_the_routing(monkeypatch):
    monkeypatch.setattr(swe_env, "uv_has_python", lambda v: False)
    assert swe_env.wants_docker("uv", "matplotlib/matplotlib", "3.6") is False     # forced host
    assert swe_env.wants_docker("docker", "pallets/flask", "3.11") is True          # forced Docker
    assert swe_env.wants_docker("auto", "pallets/flask", "3.6") is True             # no uv build


def test_import_name_handles_repos_whose_package_differs():
    assert swe_env.import_name("scikit-learn/scikit-learn") == "sklearn"
    assert swe_env.import_name("pylint-dev/pylint") == "pylint"
    assert swe_env.import_name("pydata/xarray") == "xarray"
    assert swe_env.import_name("pytest-dev/pytest") == "pytest"


def test_agent_prompt_forbids_editing_the_build_to_make_it_import():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend import agent_library
    prompt = agent_library._SWEBENCH_AGENT_PROMPT
    assert "build or packaging files" in prompt and "setup.py" in prompt
    assert agent_library._BUILTIN_AGENT_VERSIONS["swebench-coding-agent"] >= 3   # so it is re-seeded
