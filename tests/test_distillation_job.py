"""Distillation train job — start / refuse / poll / cancel.  No GPU."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.distillation import job as distill_job
from backend.distillation.job import DistillBusy, DistillStatus
from backend.routes.distillation import router as distillation_router


TRAJ = [
    {
        "session_id": "s1",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok", "tool_calls": []},
        ],
        "tools_used": ["search_messages", "send_message"],
        "steps": 2,
    }
]


class _Dist:
    teacher_model_id = "mlx-community/Qwen3-32B-4bit"
    student_model_id = "mlx-community/Qwen3-8B-4bit"
    min_tool_calls = 2
    filter_sessions_by_teacher = False
    data_dir = ""


class _Cfg:
    distillation = _Dist()


@pytest.fixture(autouse=True)
def _reset_job():
    distill_job.reset()
    distill_job.set_stream_lora_for_tests(None)
    yield
    distill_job.reset()
    distill_job.set_stream_lora_for_tests(None)


@pytest.fixture
def job_env(tmp_path, monkeypatch):
    async def aload(*_args, **_kwargs):
        return _Cfg()

    monkeypatch.setattr("backend.config.AppConfig.aload", aload)
    monkeypatch.setattr(distill_job, "blocking_sessions", lambda: [])
    monkeypatch.setattr(distill_job, "_transcripts_dir", lambda: tmp_path / "t")
    monkeypatch.setattr(distill_job, "_sessions_dir", lambda: tmp_path / "s")
    monkeypatch.setattr(distill_job, "sft_path", lambda **k: tmp_path / "sft.jsonl")
    monkeypatch.setattr(distill_job, "persist_trajectories", lambda *a, **k: tmp_path / "traj.jsonl")
    monkeypatch.setattr(distill_job, "to_sft_jsonl", lambda traj, path, **k: path.write_text("x\n"))
    monkeypatch.setattr(
        distill_job,
        "train_lora",
        lambda **k: {
            "adapter_path": str(tmp_path / "adapters" / "activity"),
            "argv": ["python", "-m", "mlx_lm.lora", "--iters", "2"],
            "warnings": [],
            "catalog_id": "otto-distill/Qwen3-8B-4bit-activity-20260922-112615",
            "display_name": "Qwen3-8B-4bit distilled (activity · 22 Sep 11:26)",
        },
    )

    async def unload():
        return {"evicted_models": 1, "metal_cleared": True}

    monkeypatch.setattr(distill_job, "_unload_mlx", unload)
    return tmp_path


async def _wait_done(timeout: float = 2.0) -> DistillStatus:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        st = distill_job.get_status()
        if st.state != "running":
            return st
        await asyncio.sleep(0.02)
    raise AssertionError(f"job still running: {distill_job.get_status().to_dict()}")


def test_status_to_dict_includes_blocking(monkeypatch):
    monkeypatch.setattr(
        distill_job,
        "blocking_sessions",
        lambda: [{"id": "abc", "title": "Run", "provider": "mlx"}],
    )
    d = distill_job.get_status().to_dict()
    assert d["state"] == "idle"
    assert d["blocking_sessions"][0]["id"] == "abc"
    assert d["log_lines"] == []


@pytest.mark.asyncio
async def test_refuse_when_session_running(job_env, monkeypatch):
    monkeypatch.setattr(
        distill_job,
        "blocking_sessions",
        lambda: [{"id": "s1", "title": "Chat", "provider": "mlx"}],
    )
    with pytest.raises(DistillBusy) as ei:
        await distill_job.start_train_job()
    assert ei.value.code == "sessions_running"
    assert distill_job.get_status().state == "idle"


@pytest.mark.asyncio
async def test_refuse_when_already_busy(job_env, monkeypatch):
    gate = asyncio.Event()

    async def hang(argv, status, cancel):
        await gate.wait()
        return 0

    monkeypatch.setattr(distill_job, "collect_trajectories", lambda **k: TRAJ)
    distill_job.set_stream_lora_for_tests(hang)
    await distill_job.start_train_job(iters=2)
    with pytest.raises(DistillBusy) as ei:
        await distill_job.start_train_job()
    assert ei.value.code == "busy"
    distill_job.request_cancel()
    gate.set()
    await _wait_done()


@pytest.mark.asyncio
async def test_empty_trajectories_errors(job_env, monkeypatch):
    monkeypatch.setattr(distill_job, "collect_trajectories", lambda **k: [])
    await distill_job.start_train_job()
    st = await _wait_done()
    assert st.state == "error"
    assert "No high-quality trajectories" in (st.error or "")


@pytest.mark.asyncio
async def test_success_streams_iters_and_adapter(job_env, monkeypatch):
    monkeypatch.setattr(distill_job, "collect_trajectories", lambda **k: TRAJ)

    async def fake_stream(argv, status, cancel):
        status.append("Iter 1: Train loss 2.0")
        status.append("Iter 2: Train loss 1.1")
        return 0

    distill_job.set_stream_lora_for_tests(fake_stream)
    await distill_job.start_train_job(iters=2, student_id="mlx-community/Qwen3-8B-4bit")
    st = await _wait_done()
    assert st.state == "success"
    assert st.iter_current == 2
    assert st.n_trajectories == 1
    assert st.adapter_path.endswith("activity")
    assert st.catalog_id.startswith("otto-distill/")
    assert "distilled" in st.display_name
    assert any("Iter 2" in ln for ln in st.log_lines)


@pytest.mark.asyncio
async def test_cancel_during_train(job_env, monkeypatch):
    monkeypatch.setattr(distill_job, "collect_trajectories", lambda **k: TRAJ)
    entered = asyncio.Event()

    async def hang(argv, status, cancel):
        entered.set()
        while not cancel.is_set():
            await asyncio.sleep(0.02)
        return -15

    distill_job.set_stream_lora_for_tests(hang)
    await distill_job.start_train_job(iters=2)
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    assert distill_job.request_cancel() == "cancel_requested"
    st = await _wait_done()
    assert st.state == "cancelled"


def test_routes_409_when_busy(job_env, monkeypatch):
    distill_job.get_status().state = "running"
    app = FastAPI()
    app.include_router(distillation_router)
    client = TestClient(app)
    res = client.post("/api/distillation/train", json={"iters": 2})
    assert res.status_code == 409
    assert res.json()["code"] == "busy"


def test_routes_status_ok():
    app = FastAPI()
    app.include_router(distillation_router)
    client = TestClient(app)
    res = client.get("/api/distillation/status")
    assert res.status_code == 200
    assert res.json()["state"] == "idle"


def test_blocking_sessions_reads_active(monkeypatch):
    fake = SimpleNamespace(
        list_active=lambda: [
            SimpleNamespace(id="a", title="One", status="running", llm_provider="mlx"),
            SimpleNamespace(id="b", title="Two", status="idle", llm_provider="mlx"),
        ]
    )
    monkeypatch.setattr("backend.state.session_mgr", fake, raising=False)
    rows = distill_job.blocking_sessions()
    assert [r["id"] for r in rows] == ["a"]


def test_list_adapters_strips_meta(monkeypatch):
    rec = {
        "catalog_id": "otto-distill/ready",
        "display_name": "Ready distilled",
        "adapter_path": "/tmp/ready",
        "base_repo_id": "mlx-community/Qwen3-8B-4bit",
        "teacher_model_id": "mlx-community/Qwen3-32B-4bit",
        "dataset_sha": "abcd1234",
        "trained_at": "2026-09-22T00:00:00Z",
        "size_mb": 12.3,
        "meta": {"secret": True},
    }
    monkeypatch.setattr(
        "backend.distillation.catalog.iter_trained_adapters",
        lambda: [rec],
    )
    app = FastAPI()
    app.include_router(distillation_router)
    client = TestClient(app)
    res = client.get("/api/distillation/adapters")
    assert res.status_code == 200
    adapters = res.json()["adapters"]
    assert len(adapters) == 1
    assert adapters[0]["catalog_id"] == "otto-distill/ready"
    assert adapters[0]["display_name"] == "Ready distilled"
    assert "meta" not in adapters[0]


def test_blocking_sessions_ignores_anthropic_running(monkeypatch):
    fake = SimpleNamespace(
        list_active=lambda: [
            SimpleNamespace(
                id="cloud",
                title="Claude",
                status="running",
                llm_provider="anthropic",
                distill_catalog_id=None,
            ),
            SimpleNamespace(
                id="local",
                title="MLX",
                status="running",
                llm_provider="mlx",
                distill_catalog_id=None,
            ),
        ]
    )
    monkeypatch.setattr("backend.state.session_mgr", fake, raising=False)
    rows = distill_job.blocking_sessions()
    assert [r["id"] for r in rows] == ["local"]


@pytest.mark.asyncio
async def test_defer_train_when_this_mlx_session_running(monkeypatch, job_env):
    monkeypatch.setattr(
        distill_job,
        "blocking_sessions",
        lambda: [{"id": "sess-1", "title": "Me", "provider": "mlx"}],
    )
    monkeypatch.setattr(distill_job, "collect_trajectories", lambda **k: TRAJ)

    async def fake_stream(argv, status, cancel):
        return 0

    distill_job.set_stream_lora_for_tests(fake_stream)

    st = await distill_job.start_train_job(
        iters=2,
        purpose="calendar scheduling",
        defer_session_id="sess-1",
    )
    assert st.state == "queued"
    assert st.purpose == "calendar scheduling"

    monkeypatch.setattr(distill_job, "blocking_sessions", lambda: [])
    started = await distill_job.start_deferred_if_idle()
    assert started is not None
    assert started.state == "running"
    done = await _wait_done()
    assert done.state == "success"


@pytest.mark.asyncio
async def test_anthropic_running_starts_train_immediately(monkeypatch, job_env):
    monkeypatch.setattr(distill_job, "blocking_sessions", lambda: [])
    monkeypatch.setattr(distill_job, "collect_trajectories", lambda **k: TRAJ)

    async def fake_stream(argv, status, cancel):
        return 0

    distill_job.set_stream_lora_for_tests(fake_stream)
    st = await distill_job.start_train_job(iters=2, purpose="inbox")
    assert st.state == "running"
    done = await _wait_done()
    assert done.state == "success"


@pytest.mark.asyncio
async def test_fuse_unknown_catalog_raises():
    with pytest.raises(ValueError, match="Unknown distilled"):
        await distill_job.start_fuse_job(catalog_id="otto-distill/missing")


@pytest.mark.asyncio
async def test_fuse_already_ready_skips_gpu(tmp_path, monkeypatch):
    adapter = tmp_path / "ready-slug"
    adapter.mkdir()
    fused = tmp_path / "fused" / "ready-slug"
    fused.mkdir(parents=True)
    (fused / "config.json").write_text("{}", encoding="utf-8")
    (fused / "model.safetensors").write_bytes(b"w")
    rec = {
        "catalog_id": "otto-distill/ready-slug",
        "display_name": "Ready",
        "adapter_path": str(adapter),
        "base_repo_id": "mlx-community/Qwen3-8B-4bit",
        "teacher_model_id": "",
        "fused_path": str(fused),
        "omlx_model_id": "ready-slug",
    }
    monkeypatch.setattr(
        "backend.distillation.catalog.resolve_catalog_id",
        lambda cid, **k: rec if cid == rec["catalog_id"] else None,
    )
    monkeypatch.setattr(distill_job, "blocking_sessions", lambda: [])

    async def boom():
        raise AssertionError("should not unload")

    monkeypatch.setattr(distill_job, "_unload_mlx", boom)
    st = await distill_job.start_fuse_job(catalog_id="otto-distill/ready-slug")
    assert st.state == "success"
    assert st.omlx_model_id == "ready-slug"
    assert st.fused_path == str(fused)


@pytest.mark.asyncio
async def test_train_from_upload_skips_collect(job_env, monkeypatch):
    def boom(**_k):
        raise AssertionError("session collect should not run for an upload")

    monkeypatch.setattr(distill_job, "collect_trajectories", boom)

    def fake_write(upload_id, dest, **_k):
        dest.write_text("{}\n", encoding="utf-8")
        return {"n_written": 4, "ok": True}

    monkeypatch.setattr(
        "backend.distillation.uploads.write_sft_from_upload",
        fake_write,
    )

    async def fake_stream(argv, status, cancel):
        return 0

    distill_job.set_stream_lora_for_tests(fake_stream)
    await distill_job.start_train_job(iters=2, upload_id="jobs-abc123")
    st = await _wait_done()
    assert st.state == "success"
    assert st.n_sft == 4
    assert st.n_trajectories == 4
