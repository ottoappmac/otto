"""Ledger, resume, retry and reset logic of the batched SWE-bench scorer (no Docker)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "swebench"))
import score  # noqa: E402


def _preds(*items):
    return {i: {"instance_id": i, "model_name_or_path": "otto", "model_patch": p}
            for i, p in items}


def _report(out, run_id, iid, resolved=True, applied=True):
    d = out / "logs" / "run_evaluation" / run_id / "otto" / iid
    d.mkdir(parents=True)
    (d / "report.json").write_text(json.dumps(
        {iid: {"resolved": resolved, "patch_successfully_applied": applied}}))


def test_pending_skips_scored_and_rescores_changed_patches(tmp_path):
    ledger = score.Ledger(tmp_path / "l.jsonl")
    preds = _preds(("a", "diff a"), ("b", "diff b"), ("c", "diff c"))
    ledger.add("a", "resolved", patch_sha=score.sha("diff a"))
    ledger.add("b", "unresolved", patch_sha=score.sha("OLD b"))   # patch changed since
    todo = score.select_pending(preds, ledger.state(), retry_errors=False)
    assert todo == ["b", "c"]


def test_errors_only_retried_on_request(tmp_path):
    ledger = score.Ledger(tmp_path / "l.jsonl")
    preds = _preds(("a", "diff a"))
    ledger.add("a", "error", patch_sha=score.sha("diff a"))
    assert score.select_pending(preds, ledger.state(), retry_errors=False) == []
    assert score.select_pending(preds, ledger.state(), retry_errors=True) == ["a"]


def test_reset_makes_instance_pending_and_keeps_history(tmp_path):
    ledger = score.Ledger(tmp_path / "l.jsonl")
    preds = _preds(("a", "diff a"))
    ledger.add("a", "resolved", patch_sha=score.sha("diff a"))
    ledger.add("a", "pending", reason="reset")
    assert score.select_pending(preds, ledger.state(), False) == ["a"]
    assert len(score.Ledger(tmp_path / "l.jsonl").events) == 2   # history kept on disk


def test_ledger_survives_reload(tmp_path):
    p = tmp_path / "l.jsonl"
    score.Ledger(p).add("a", "resolved", patch_sha="x")
    assert score.Ledger(p).state()["a"]["status"] == "resolved"


def test_harvest_reads_partial_batch_reports(tmp_path):
    _report(tmp_path, "run1", "a", resolved=True)
    _report(tmp_path, "run1", "b", resolved=False)
    _report(tmp_path, "run1", "c", resolved=False, applied=False)
    got = score.harvest(tmp_path, "run1", ["a", "b", "c", "d"])   # d never finished
    assert got == {"a": "resolved", "b": "unresolved", "c": "patch_failed"}


def test_batch_run_id_changes_with_patch_not_with_order():
    p1 = _preds(("a", "x"), ("b", "y"))
    p2 = _preds(("a", "x"), ("b", "CHANGED"))
    assert score.batch_run_id("r", ["a", "b"], p1) == score.batch_run_id("r", ["b", "a"], p1)
    assert score.batch_run_id("r", ["a", "b"], p1) != score.batch_run_id("r", ["a", "b"], p2)


def test_load_preds_last_line_wins(tmp_path):
    lines = [{"instance_id": "a", "model_patch": "old"}, {"instance_id": "a", "model_patch": "new"}]
    (tmp_path / "preds.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert score.load_preds(tmp_path)["a"]["model_patch"] == "new"


def test_results_summary_counts(tmp_path):
    preds = _preds(("a", "pa"), ("b", "pb"), ("c", ""), ("d", "pd"))
    ledger = score.Ledger(tmp_path / "l.jsonl")
    ledger.add("a", "resolved", patch_sha="1")
    ledger.add("b", "unresolved", patch_sha="2")
    ledger.add("c", "empty_patch", patch_sha="3")
    res = score.write_results(tmp_path, preds, ledger, "ds", "local", "run")
    assert (res["submitted"], res["scored"], res["resolved"]) == (4, 3, 1)
    assert res["pending"] == ["d"] and res["empty_patch"] == ["c"]
    assert res["resolved_pct_of_scored"] == 33.3
    assert (tmp_path / "results.json").exists()


# ── prebuilt images and gold validation ──────────────────────────────

class _Fake:
    def __init__(self, fail=()):
        self.calls, self.fail = [], set(fail)

    def __call__(self, *args, timeout=None):
        import subprocess
        self.calls.append(args)
        name = args[-1]
        code = 1 if (args[0] == "pull" and name in self.fail) else 0
        return subprocess.CompletedProcess(args, code, stdout="", stderr="")


def test_image_name_matches_official_naming():
    assert score.image_name("swebench", "django__django-10097") == \
        "swebench/sweb.eval.x86_64.django_1776_django-10097:latest"
    assert score.image_name("swebench", "sphinx-doc__sphinx-10323") == \
        "swebench/sweb.eval.x86_64.sphinx-doc_1776_sphinx-10323:latest"


def test_pull_reports_failures_and_remove_removes_everything(monkeypatch):
    fake = _Fake(fail={"b"})
    monkeypatch.setattr(score, "_docker", fake)
    assert score.pull_images(["a", "b", "c"], jobs=2) == {"a": True, "b": False, "c": True}
    assert all("linux/amd64" in c for c in fake.calls if c[0] == "pull")
    score.remove_images(["a", "b"])
    assert [c for c in fake.calls if c[0] == "rmi"] == [("rmi", "-f", "a"), ("rmi", "-f", "b")]


def test_gold_summary_lists_instances_failing_with_the_gold_patch(tmp_path):
    preds = _preds(("a", "pa"), ("b", "pb"), ("c", "pc"), ("d", "pd"))
    gold = tmp_path / "gold"
    gold.mkdir()
    gl = score.Ledger(gold / "score_ledger.jsonl")
    gl.add("a", "resolved", patch_sha="1")
    gl.add("b", "unresolved", patch_sha="2")
    gl.add("c", "error", patch_sha="3")
    g = score.gold_summary(tmp_path, preds)
    assert g["gold_valid_ids"] == ["a"] and g["gold_failed_ids"] == ["b"]
    assert g["gold_error_ids"] == ["c"] and g["gold_unchecked_ids"] == ["d"]


def test_gold_summary_absent_without_validation(tmp_path):
    assert score.gold_summary(tmp_path, _preds(("a", "p"))) is None


def test_score_over_gold_valid_instances(tmp_path):
    preds = _preds(("a", "pa"), ("b", "pb"), ("c", "pc"))
    gold = tmp_path / "gold"
    gold.mkdir()
    gl = score.Ledger(gold / "score_ledger.jsonl")
    for i, st in (("a", "resolved"), ("b", "resolved"), ("c", "unresolved")):
        gl.add(i, st, patch_sha="x")
    ledger = score.Ledger(tmp_path / "l.jsonl")
    ledger.add("a", "resolved", patch_sha="1")
    ledger.add("b", "unresolved", patch_sha="2")
    ledger.add("c", "resolved", patch_sha="3")   # solved one the gold patch cannot pass
    res = score.write_results(tmp_path, preds, ledger, "ds", "local", "run")
    assert res["resolved"] == 2                      # a and c
    assert res["resolved_of_gold_valid"] == 1        # only a counts: c is not gold-valid
    assert res["resolved_pct_of_gold_valid"] == 50.0  # 1 of the 2 valid ids (a, b)


def test_images_already_present_are_not_pulled_or_removed(tmp_path, monkeypatch):
    """A loaded bundle must survive scoring: only images we pulled ourselves get removed."""
    import argparse

    present = {score.image_name("ns", "a__x-1")}
    calls = []

    def fake_docker(*args, timeout=None):
        import subprocess
        calls.append(args)
        if args[:2] == ("image", "inspect"):
            return subprocess.CompletedProcess(args, 0 if args[2] in present else 1, "", "")
        if args[0] == "run":                       # docker_free_gb
            return subprocess.CompletedProcess(args, 0, "h\n/ 1 2 999999999 4 5", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(score, "_docker", fake_docker)
    monkeypatch.setattr(score, "run_local_batch", lambda *a, **k: 0)
    monkeypatch.setattr(score, "harvest", lambda *a, **k: {})
    preds = _preds(("a__x-1", "pa"), ("b__y-2", "pb"))
    (tmp_path / "preds.jsonl").write_text("")
    args = argparse.Namespace(method="local", namespace="ns", batch=0, retry_errors=False,
                              workers=1, keep_images=False)
    try:
        score.score_folder(tmp_path, preds, "ds", "run", args)
    except SystemExit:
        pass                                        # the "no report" error path may exit; irrelevant here
    pulled = [c[-1] for c in calls if c[0] == "pull"]
    removed = [c[-1] for c in calls if c[0] == "rmi"]
    assert pulled == [score.image_name("ns", "b__y-2")]            # the present one is not pulled
    assert score.image_name("ns", "a__x-1") not in removed          # and is never removed
    assert removed == [score.image_name("ns", "b__y-2")]
