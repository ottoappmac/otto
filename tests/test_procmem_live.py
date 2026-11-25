"""Live procedural memory in the benchmark (`bench.sh up --memory live`): a task can recall the earlier tasks of the
same run, passed or failed, but not its own stored runs unless same-task recall is asked for (2026-11-24)."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path

import pytest

from backend.procedural_memory import enrich, service, tool
from backend.procedural_memory.embed import HashEmbedder
from tests.test_procedural_memory import AUTOMATOR, automator_rows

MA = Path(__file__).resolve().parents[1] / "scripts" / "macarena"
TASK = 'Make a Quick Action named "Hello" that echoes input.'
TWIN = 'Make a Quick Action named "Bye" that echoes input.'


def run(coro):
    return asyncio.run(coro)


def _mod(name):
    if str(MA) not in sys.path:
        sys.path.insert(0, str(MA))
    return importlib.import_module(name)


@pytest.fixture
def live(tmp_path, monkeypatch):
    """A benchmark backend started with --memory live, its write store pointed at the run folder (as run.py does),
    and an LLM that writes one lesson."""
    monkeypatch.setenv(service.ENV_MODE, "live")
    monkeypatch.setenv(service.ENV_FLOOR, "0.0")
    monkeypatch.delenv(service.ENV_SERVE, raising=False)
    monkeypatch.delenv(service.ENV_SAME_TASK, raising=False)
    service.set_embedder(HashEmbedder())
    service.set_write_path(tmp_path / "memory" / "write.db")

    async def fake_default_llm():
        async def call(prompt):
            return json.dumps({"intent": "Automator: Make a Quick Action that runs a shell script",
                               "segments": [],
                               "lessons": [{"title": "Confirm the workflow was saved",
                                            "sentence": "After File > Save, check the Services folder for it.",
                                            "content": "Next time save before running anything that removes it."}]})
        return call

    monkeypatch.setattr(enrich, "default_llm", fake_default_llm)
    yield tmp_path
    service._runtime_override.clear()
    service._stores.clear()
    service.set_embedder(None)


def _save_failed_task():
    return run(service.save_session("task1", rows=automator_rows(), meta={"agent_name": "macarena-desktop-agent"},
                                    track="benchmark", task_id="T1", label="failure", label_source="benchmark"))


def _recall(current):
    return run(tool.run_recall(agent="macos-desktop-agent", session_id="task2", app=AUTOMATOR,
                               task="Automator: Quick Action that runs a shell script", current_task_text=current))


def test_live_serves_the_runs_own_write_store(live):
    rt = service.runtime()
    assert rt.live and rt.benchmark and rt.serve_tool and rt.serve_path is None
    assert service.serve_store(rt) is service.write_store(rt)
    assert not service.serve_store(rt).read_only
    st = service.status()
    assert st["live"] and st["serve_tool"] and st["mode"] == "benchmark"


def test_a_later_similar_task_recalls_an_earlier_failed_one_with_its_lesson(live):
    assert _recall(TWIN)[1]["abstained"]                     # nothing stored yet
    [eid] = _save_failed_task()
    assert service.write_store().episode(eid)["status"] == "admitted"
    text, rec = _recall(TWIN)                                 # task 2: same template, another name
    assert not rec["abstained"]
    assert {u["source_task"] for u in rec["used"]} == {"T1"}
    assert "Confirm the workflow was saved" in text           # the failed run's lesson reaches the next task


def test_live_leaves_out_only_the_very_same_task_unless_asked(live, monkeypatch):
    _save_failed_task()
    assert _recall(TASK)[1]["abstained"]                      # its own stored run: left out
    monkeypatch.setenv(service.ENV_SAME_TASK, "1")
    _, rec = _recall(TASK)                                    # smoke test of a repeat: allowed
    assert not rec["abstained"] and rec["used"]


def test_failed_run_lessons_ask_for_what_to_try_next_time():
    p = enrich.build_prompt({"task_text": TASK, "apps": [AUTOMATOR], "segments": [], "facts": []}, "failure", "benchmark")
    assert "what to try instead the next time" in p and "not what to avoid" in p


def test_ids_run_in_the_order_given():
    run_mod = _mod("run")
    tasks = [{"id": i, "source": "macosworld", "domain": "d"} for i in ("a1", "b2", "c3")]
    assert [t["id"] for t in run_mod.select(tasks, "c3,a1", None, None, 0)] == ["c3", "a1"]


def test_runner_records_live_memory(tmp_path):
    run_mod = _mod("run")

    class FakeApi:
        def __init__(self):
            self.calls = []

        async def call(self, method, path, **kw):
            self.calls.append(path)
            if path.endswith("/status"):
                return {"mode": "benchmark", "live": True, "serve_tool": True, "floor": 0.45, "embedder": "m"}
            return {}

    api = FakeApi()
    fields = run(run_mod.memory_setup(api, tmp_path))
    assert fields["memory_mode"] == "live" and fields["memory_write"] == "memory/write.db"
    assert "/api/procedures/benchmark/write-db" in api.calls and "memory_same_task" not in fields


def _row(i, **kw):
    return {"id": i, "source": "macosworld", "domain": "d", "run": 1, "score": 1.0, "status": "ok", "steps": 3,
            "track": "native", "max_steps": 100, "step_unit": "turns", "setup_mode": "sync", **kw}


def test_report_labels_live_runs_and_never_pools_them():
    rep = _mod("report")
    text = rep.render(rep.summarize([_row("a", memory_mode="live", memory_calls=1)]))
    assert "procedural memory: live" in text and "learning curve" in text
    with pytest.raises(ValueError, match="different procedural memory"):
        rep.summarize([_row("a", memory_mode="live"), _row("b")])
