"""Local dataset, configurable image set, no-pull behaviour and the bench.sh flags (no Docker)."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "swebench"
sys.path.insert(0, str(SCRIPTS))
import data  # noqa: E402
import docker_env  # noqa: E402
import env as swe_env  # noqa: E402
import score  # noqa: E402


# ── local dataset ────────────────────────────────────────────────────

def _write_jsonl(path, ids):
    path.write_text("".join(json.dumps({"instance_id": i, "repo": "r/r", "FAIL_TO_PASS": "[]"}) + "\n" for i in ids))


def test_load_rows_reads_a_local_jsonl_without_the_datasets_library(tmp_path):
    f = tmp_path / "ds.jsonl"
    _write_jsonl(f, ["a", "b"])
    assert [r["instance_id"] for r in data.load_rows(str(f))] == ["a", "b"]
    assert data.is_local(str(f)) and not data.is_local("verified")


def test_load_rows_reads_a_local_json_list(tmp_path):
    f = tmp_path / "ds.json"
    f.write_text(json.dumps([{"instance_id": "x"}]))
    assert data.load_rows(str(f)) == [{"instance_id": "x"}]


def test_export_round_trips_and_leaves_no_partial_file(tmp_path):
    src, out = tmp_path / "src.jsonl", tmp_path / "sub" / "out.jsonl"
    _write_jsonl(src, ["a", "b", "c"])
    assert data.export(str(src), out) == 3
    assert [r["instance_id"] for r in data.load_rows(str(out))] == ["a", "b", "c"]
    assert not list(out.parent.glob("*.tmp"))


def test_short_names_resolve_to_hugging_face_ids():
    assert data.resolve("verified") == "princeton-nlp/SWE-bench_Verified"
    assert data.resolve("/some/where/ds.jsonl") == "/some/where/ds.jsonl"


# ── image set ────────────────────────────────────────────────────────

def test_image_name_follows_the_configured_namespace(monkeypatch):
    monkeypatch.setattr(docker_env, "IMAGE_NAMESPACE", "swebench")
    assert docker_env.image_name("django__django-1") == "swebench/sweb.eval.x86_64.django_1776_django-1:latest"
    docker_env.configure(namespace="logicstar")
    assert docker_env.image_name("django__django-1") == "logicstar/sweb.eval.x86_64.django_1776_django-1:latest"
    assert docker_env.image_name("django__django-1", "other").startswith("other/")


def test_configure_can_disable_pulling(monkeypatch):
    monkeypatch.setattr(docker_env, "ALLOW_PULL", True)
    monkeypatch.setattr(docker_env, "IMAGE_NAMESPACE", "swebench")      # restored after the test
    docker_env.configure(allow_pull=False)
    assert docker_env.ALLOW_PULL is False
    docker_env.configure(namespace="x")                  # leaving allow_pull alone keeps it
    assert docker_env.ALLOW_PULL is False


def test_cleanup_removes_only_images_we_pulled(monkeypatch):
    calls = []
    monkeypatch.setattr(docker_env.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    ours = swe_env.Prepared(Path("/x"), "s", "t", [], mode="docker", container="c1", image="i1", pulled=True)
    local = swe_env.Prepared(Path("/x"), "s", "t", [], mode="docker", container="c2", image="i2", pulled=False)
    docker_env.cleanup(ours)
    docker_env.cleanup(local)
    assert ["docker", "rmi", "-f", "i1"] in calls
    assert ["docker", "rmi", "-f", "i2"] not in calls          # a loaded bundle image survives
    assert ["docker", "rm", "-f", "c1"] in calls and ["docker", "rm", "-f", "c2"] in calls


# ── scoring without pulls ────────────────────────────────────────────

def test_scoring_with_no_pull_records_an_error_for_a_missing_image(tmp_path, monkeypatch):
    present = {score.image_name("ns", "a__x-1")}
    calls = []

    def fake_docker(*args, timeout=None):
        calls.append(args)
        if args[:2] == ("image", "inspect"):
            return subprocess.CompletedProcess(args, 0 if args[2] in present else 1, "", "")
        return subprocess.CompletedProcess(args, 0, "h\n/ 1 2 999999999 4 5", "")

    monkeypatch.setattr(score, "_docker", fake_docker)
    monkeypatch.setattr(score, "run_local_batch", lambda *a, **k: 0)
    monkeypatch.setattr(score, "harvest", lambda *a, **k: {})
    preds = {i: {"instance_id": i, "model_name_or_path": "o", "model_patch": "p"} for i in ("a__x-1", "b__y-2")}
    (tmp_path / "preds.jsonl").write_text("")
    args = argparse.Namespace(method="local", namespace="ns", batch=0, retry_errors=False,
                              workers=1, keep_images=False, no_pull=True)
    try:
        score.score_folder(tmp_path, preds, "ds", "run", args)
    except SystemExit:
        pass
    assert not [c for c in calls if c[0] == "pull"]                       # nothing was pulled
    assert not [c for c in calls if c[0] == "rmi"]                        # nothing was removed
    led = score.Ledger(tmp_path / "score_ledger.jsonl").state()
    assert "pulling is disabled" in led["b__y-2"]["reason"] and led["b__y-2"]["status"] == "error"


# ── bench.sh flags ───────────────────────────────────────────────────

@pytest.mark.skipif(sys.platform != "darwin", reason="bench.sh is macOS-only")
def test_bench_help_lists_the_offline_options():
    out = subprocess.run([str(SCRIPTS / "bench.sh"), "--help"], capture_output=True, text=True).stdout
    for word in ("warm", "--offline", "--image-source"):
        assert word in out


@pytest.mark.skipif(sys.platform != "darwin", reason="bench.sh is macOS-only")
def test_bench_rejects_a_bad_image_source():
    r = subprocess.run([str(SCRIPTS / "bench.sh"), "check", "--image-source", "nope"], capture_output=True, text=True)
    assert r.returncode != 0 and "official or bundle" in (r.stdout + r.stderr)


# ── the scorer's GitHub lookups ──────────────────────────────────────

def _fake_response(status, text):
    import requests
    r = requests.models.Response()
    r.status_code, r._content, r.encoding = status, text.encode(), "utf-8"
    return r


def test_http_cache_records_online_and_replays_offline(tmp_path, monkeypatch):
    import requests
    import http_cache
    monkeypatch.setenv("SWEBENCH_HTTP_CACHE", str(tmp_path))
    monkeypatch.delenv("SWEBENCH_OFFLINE", raising=False)
    calls = []

    def real_get(url, *a, **k):
        calls.append(url)
        return _fake_response(200, "numpy==1.0\n")

    monkeypatch.setattr(requests, "get", real_get)
    http_cache.install()
    url = "https://raw.githubusercontent.com/o/r/abc/requirements.txt"
    assert requests.get(url).text == "numpy==1.0\n" and calls == [url]          # online: fetched
    assert requests.get(url).text == "numpy==1.0\n" and calls == [url]          # again: from the cache

    monkeypatch.setenv("SWEBENCH_OFFLINE", "1")
    assert requests.get(url).text == "numpy==1.0\n" and calls == [url]          # offline: replayed
    miss = requests.get("https://raw.githubusercontent.com/o/r/zzz/requirements.txt")
    assert miss.status_code == 200 and miss.text == "" and calls == [url]       # unseen: empty, no network


def test_http_cache_leaves_other_urls_alone_and_does_not_double_wrap(tmp_path, monkeypatch):
    import requests
    import http_cache
    monkeypatch.setenv("SWEBENCH_HTTP_CACHE", str(tmp_path))
    seen = []
    monkeypatch.setattr(requests, "get", lambda url, *a, **k: seen.append(url) or _fake_response(200, "x"))
    http_cache.install()
    wrapped = requests.get
    http_cache.install()
    assert requests.get is wrapped                                              # installed once
    requests.get("https://example.com/a")
    assert seen == ["https://example.com/a"] and not list(tmp_path.iterdir())   # passed through, not cached


def test_http_cache_keeps_404_answers_but_not_server_errors(tmp_path, monkeypatch):
    import requests
    import http_cache
    monkeypatch.setenv("SWEBENCH_HTTP_CACHE", str(tmp_path))
    monkeypatch.delenv("SWEBENCH_OFFLINE", raising=False)
    statuses = iter([404, 503])
    monkeypatch.setattr(requests, "get", lambda url, *a, **k: _fake_response(next(statuses), ""))
    http_cache.install()
    requests.get("https://raw.githubusercontent.com/o/r/c/a.txt")               # 404 kept
    requests.get("https://raw.githubusercontent.com/o/r/c/b.txt")               # 503 not kept
    assert len(list(tmp_path.iterdir())) == 1
