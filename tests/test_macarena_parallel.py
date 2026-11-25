"""Running the MacArena bench on two VMs at once: slots in bench.sh, --shard in run.py, pooling in report.py."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MA = ROOT / "scripts" / "macarena"
BENCH = MA / "bench.sh"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(MA))
try:
    run = _load("macarena_run_parallel", MA / "run.py")
    report = _load("macarena_report_parallel", MA / "report.py")
finally:
    sys.path.remove(str(MA))


def _tasks():
    return [{"id": f"{d}-{i}", "source": "osworld", "domain": d} for d in ("calc", "chrome", "gimp") for i in range(5)]


@pytest.mark.parametrize("n", [1, 2, 3])
def test_shards_cover_the_selection_once_and_mix_domains(n):
    tasks = _tasks()
    parts = [run.shard(tasks, k, n) for k in range(1, n + 1)]
    ids = [t["id"] for p in parts for t in p]
    assert sorted(ids) == sorted(t["id"] for t in tasks) and len(ids) == len(set(ids))
    assert max(len(p) for p in parts) - min(len(p) for p in parts) <= 1
    if n == 2:
        assert all({t["domain"] for t in p} == {"calc", "chrome", "gimp"} for p in parts)


def test_shard_order_does_not_depend_on_input_order():
    tasks = _tasks()
    assert run.shard(tasks, 1, 2) == run.shard(list(reversed(tasks)), 1, 2)


@pytest.mark.parametrize("text,want", [(None, (1, 1)), ("1/2", (1, 2)), ("2/2", (2, 2))])
def test_parse_shard(text, want):
    assert run.parse_shard(text) == want


@pytest.mark.parametrize("text", ["3/2", "0/2", "half", "1-2"])
def test_parse_shard_rejects_bad_values(text):
    with pytest.raises(SystemExit):
        run.parse_shard(text)


def _row(i, **kw):
    return {"id": f"t{i}", "run": 1, "score": 1.0, "status": "ok", "source": "osworld", "domain": "d", "steps": 3,
            "track": "native", **kw}


def test_report_keeps_runs_with_different_vm_counts_apart():
    with pytest.raises(ValueError, match="numbers of VMs"):
        report.summarize([_row(1, parallel=2), _row(2)])               # rows without the field ran alone
    s = report.summarize([_row(1, parallel=2), _row(2, parallel=2)])
    assert s["parallel"] == 2 and "VMs at once: 2" in report.render(s)
    assert "VMs at once" not in report.render(report.summarize([_row(1)]))


def _folder(tmp_path, name, rows):
    d = tmp_path / name
    d.mkdir()
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def test_report_combines_shard_folders_and_refuses_overlaps(tmp_path):
    a = _folder(tmp_path, "a", [_row(1, parallel=2), _row(2, parallel=2, score=0.0)])
    b = _folder(tmp_path, "b", [_row(3, parallel=2)])
    rows = report.load_many([a, b])
    assert sorted(r["id"] for r in rows) == ["t1", "t2", "t3"]
    assert report.summarize(rows)["tasks"] == 3
    c = _folder(tmp_path, "c", [_row(1, parallel=2)])
    with pytest.raises(ValueError, match="in both"):
        report.load_many([a, c])


def test_report_cli_prints_combined_and_writes_nothing(tmp_path):
    a = _folder(tmp_path, "a", [_row(1, parallel=2)])
    b = _folder(tmp_path, "b", [_row(2, parallel=2)])
    out = subprocess.run([sys.executable, str(MA / "report.py"), str(a), str(b)], capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("combined: ") and not (a / "summary.txt").exists()


def _bench(tmp_path, *args, slot=None):
    env = {**os.environ, "MACARENA_RUNS": str(tmp_path)}
    env.pop("MACARENA_SLOT", None)
    if slot is not None:
        env["MACARENA_SLOT"] = str(slot)
    return subprocess.run(["bash", str(BENCH), *args], capture_output=True, text=True, env=env, timeout=60)


@pytest.mark.parametrize("slot", ["0", "3", "two"])
def test_bench_refuses_slots_macos_cannot_run(tmp_path, slot):
    out = _bench(tmp_path, "slots", slot=slot)
    assert out.returncode == 2 and "MACARENA_SLOT must be 1 to 2" in out.stderr


def test_bench_lists_slots_and_sees_a_running_bridge(tmp_path):
    out = _bench(tmp_path, "slots")
    assert out.returncode == 0 and out.stdout.splitlines() == ["slot 1: down", "slot 2: down"]
    (tmp_path / "slot2").mkdir()
    (tmp_path / "slot2" / "bridge.pid").write_text(f"{os.getpid()}\n")      # a live process stands in for it
    out = _bench(tmp_path, "slots")
    assert out.stdout.splitlines()[1] == f"slot 2: up (bridge 18195, backend 18193, logs in {tmp_path}/slot2)"


def test_slot_two_gets_its_own_ports_and_state():
    text = BENCH.read_text()
    assert 'API_PORT=$(( 18093 + 100 * (SLOT - 1) ))' in text and "slot_bridge_port" in text
    backend = (MA / "start_backend.sh").read_text()
    assert 'BENCH_HOME="$STATE/home"' in backend and 'OTTO_DESKTOP_LOCK_PATH="$STATE/desktop.lock"' in backend
    assert '--port "$API_PORT"' in backend
