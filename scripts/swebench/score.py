#!/usr/bin/env python
"""Score a preds.jsonl from run.py in batches, with a resumable ledger.

    uv run --with-requirements scripts/swebench/requirements.txt \
        python scripts/swebench/score.py --out <run folder> --prebuilt

Every scored instance is recorded in <out>/score_ledger.jsonl (append-only), and
<out>/results.json is rewritten after each batch, so a stop at any point keeps
everything finished so far. Run it again to carry on:

  default            score instances that are new or whose patch changed
  --retry-errors     also re-score instances that errored last time
  --status           print counts and exit (no Docker needed)
  --reset ids|errors|all   mark instances as pending again (history is kept)
  --batch N          instances per batch (default 50; 0 = one batch)

Environments:
  default            build each instance's image locally. Packages are resolved
                     today, so a correct patch can fail on a drifted dependency.
  --prebuilt         use the official prebuilt images (x86_64, run under Rosetta).
                     Each batch's images are pulled first and removed afterwards,
                     so disk use stays at one batch. Use this for numbers you quote.
  --keep-images      do not remove pulled images after a batch

Validation:
  --validate-gold    first score the dataset's own correct patches for the same
                     instances (in <out>/gold). results.json then lists instances
                     that fail even with the gold patch, and reports the score
                     over the instances that pass it.
  --gold-only        run the gold validation and stop

Methods:
  local   SWE-bench's own Docker harness (default). Needs Docker running.
  sbcli   Uploads patches to SWE-bench's cloud service with `sb-cli`. Needs
          `sb login`. Patches leave this machine. Not batched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATASETS = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
}
DONE = {"resolved", "unresolved", "patch_failed", "empty_patch"}
import data  # noqa: E402  (local module next to this script)

PREBUILT_NAMESPACE = "swebench"
GB_PER_PREBUILT_IMAGE = 3.5   # measured: 2.8-3.0 GB each, little layer sharing between repos


def sha(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:10]


# ── ledger ───────────────────────────────────────────────────────────

class Ledger:
    """Append-only event log; the current state of an instance is its last event."""

    def __init__(self, path: Path):
        self.path = path
        self.events: list[dict] = []
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    self.events.append(json.loads(line))

    def state(self) -> dict[str, dict]:
        cur: dict[str, dict] = {}
        for ev in self.events:
            cur[ev["instance_id"]] = ev
        return cur

    def add(self, instance_id: str, status: str, **extra) -> None:
        ev = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "instance_id": instance_id,
              "status": status, **extra}
        self.events.append(ev)
        with self.path.open("a") as f:
            f.write(json.dumps(ev) + "\n")


def load_preds(out: Path) -> dict[str, dict]:
    """One row per instance; if run.py retried one, the last line wins."""
    rows: dict[str, dict] = {}
    for line in (out / "preds.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["instance_id"]] = r
    return rows


def select_pending(preds: dict[str, dict], state: dict[str, dict],
                   retry_errors: bool) -> list[str]:
    todo = []
    for iid, row in preds.items():
        ev = state.get(iid)
        psha = sha(row["model_patch"])
        if ev is None or ev["status"] == "pending" or ev.get("patch_sha") != psha:
            todo.append(iid)
        elif ev["status"] == "error" and retry_errors:
            todo.append(iid)
    return todo


def batch_run_id(run_name: str, batch: list[str], preds: dict[str, dict]) -> str:
    parts = sorted(i + ":" + sha(preds[i]["model_patch"]) for i in batch)
    return run_name + "-" + sha(",".join(parts))


def harvest(out: Path, run_id: str, ids: list[str]) -> dict[str, str]:
    """Read per-instance harness reports. Works for a batch that was interrupted."""
    found: dict[str, str] = {}
    base = out / "logs" / "run_evaluation" / run_id
    for iid in ids:
        for rep in base.glob(f"*/{iid}/report.json"):
            data = json.loads(rep.read_text()).get(iid, {})
            if data.get("resolved"):
                found[iid] = "resolved"
            elif data.get("patch_successfully_applied") is False:
                found[iid] = "patch_failed"
            else:
                found[iid] = "unresolved"
    return found


# ── docker helpers (one place, so tests can replace it) ──────────────

def _docker(*args: str, timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def image_name(namespace: str, instance_id: str) -> str:
    """Name of the official prebuilt image, e.g. swebench/sweb.eval.x86_64.django_1776_django-10097:latest"""
    return f"{namespace}/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest"


def docker_free_gb() -> float | None:
    """Free space inside Docker's own virtual disk, or None if it cannot be read."""
    try:
        r = _docker("run", "--rm", "alpine", "df", "-k", "/", timeout=120)
        return int(r.stdout.splitlines()[1].split()[3]) / 1048576
    except Exception:  # noqa: BLE001
        return None


def image_exists(name: str) -> bool:
    return _docker("image", "inspect", name).returncode == 0


def pull_images(names: list[str], jobs: int = 4) -> dict[str, bool]:
    def pull(name: str) -> bool:
        r = _docker("pull", "--platform", "linux/amd64", "-q", name)
        return r.returncode == 0
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        return dict(zip(names, pool.map(pull, names)))


def remove_images(names: list[str]) -> None:
    for name in names:
        _docker("rmi", "-f", name)


# ── scoring methods ──────────────────────────────────────────────────

def run_local_batch(out: Path, batch_ids: list[str], preds: dict[str, dict], dataset: str,
                    run_id: str, workers: int, namespace: str) -> int:
    batch_file = out / "score_batches" / f"{run_id}.jsonl"
    batch_file.parent.mkdir(exist_ok=True)
    batch_file.write_text("".join(json.dumps(preds[i]) + "\n" for i in batch_ids))
    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / "harness.py"),
        "--dataset_name", dataset, "--predictions_path", str(batch_file),
        "--instance_ids", *batch_ids,
        "--max_workers", str(workers), "--run_id", run_id,
        "--namespace", namespace, "--timeout", "1800",
    ]
    print("+", " ".join(cmd[:8]), "…", flush=True)
    return subprocess.run(cmd, cwd=out).returncode


def run_sbcli(out: Path, preds_path: Path, dataset: str, run_id: str) -> dict:
    sb_dataset = {"princeton-nlp/SWE-bench_Verified": "swe-bench_verified",
                  "princeton-nlp/SWE-bench_Lite": "swe-bench_lite",
                  "princeton-nlp/SWE-bench": "swe-bench"}[dataset]
    cmd = ["sb-cli", "submit", sb_dataset, "test", "--predictions_path", str(preds_path),
           "--run_id", run_id, "--output_dir", str(out / "sbcli")]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    reports = sorted((out / "sbcli").glob("*.json"))
    if not reports:
        raise SystemExit("sb-cli finished but wrote no report json")
    return json.loads(reports[-1].read_text())


# ── summary ──────────────────────────────────────────────────────────

def gold_summary(out: Path, preds: dict[str, dict]) -> dict | None:
    """Gold-patch validation results for these instances, if validation was run."""
    gl = out / "gold" / "score_ledger.jsonl"
    if not gl.exists():
        return None
    state = Ledger(gl).state()
    status = {i: state.get(i, {}).get("status") for i in preds}
    return {
        "gold_valid_ids": sorted(i for i, s in status.items() if s == "resolved"),
        "gold_failed_ids": sorted(i for i, s in status.items() if s in
                                  ("unresolved", "patch_failed", "empty_patch")),
        "gold_error_ids": sorted(i for i, s in status.items() if s == "error"),
        "gold_unchecked_ids": sorted(i for i, s in status.items() if s is None or s == "pending"),
    }


def write_results(out: Path, preds: dict[str, dict], ledger: Ledger, dataset: str,
                  method: str, run_name: str) -> dict:
    state = ledger.state()
    status = {iid: state.get(iid, {}).get("status", "pending") for iid in preds}
    resolved = sorted(i for i, s in status.items() if s == "resolved")
    scored = sorted(i for i, s in status.items() if s in DONE)
    traj = {}
    for iid in preds:
        f = out / "trajectories" / f"{iid}.json"
        if f.exists():
            d = json.loads(f.read_text())
            traj[iid] = {"status": d.get("status"), "seconds": d.get("seconds"),
                         "patch_chars": d.get("patch_chars"), "env_notes": d.get("env_notes")}
    results = {
        "run_id": run_name, "dataset": dataset, "method": method,
        "model_name_or_path": next(iter(preds.values())).get("model_name_or_path"),
        "submitted": len(preds),
        "scored": len(scored),
        "pending": sorted(i for i, s in status.items() if s == "pending"),
        "resolved": len(resolved),
        "resolved_pct_of_submitted": round(100 * len(resolved) / len(preds), 1),
        "resolved_pct_of_scored": round(100 * len(resolved) / len(scored), 1) if scored else None,
        "empty_patch": sorted(i for i, s in status.items() if s == "empty_patch"),
        "patch_failed": sorted(i for i, s in status.items() if s == "patch_failed"),
        "resolved_ids": resolved,
        "unresolved_ids": sorted(i for i, s in status.items() if s in
                                 ("unresolved", "patch_failed", "empty_patch")),
        "error_ids": sorted(i for i, s in status.items() if s == "error"),
        "instances": traj,
    }
    gold = gold_summary(out, preds)
    if gold:
        results.update(gold)
        valid = set(gold["gold_valid_ids"])
        n_ok = len([i for i in resolved if i in valid])
        results["resolved_of_gold_valid"] = n_ok
        results["resolved_pct_of_gold_valid"] = round(100 * n_ok / len(valid), 1) if valid else None
    (out / "results.json").write_text(json.dumps(results, indent=1))
    return results


def print_status(results: dict) -> None:
    print(f"run {results['run_id']}: {results['scored']}/{results['submitted']} scored, "
          f"{results['resolved']} resolved "
          f"({results['resolved_pct_of_submitted']}% of submitted"
          + (f", {results['resolved_pct_of_scored']}% of scored" if results['scored'] else "")
          + f")\n  pending {len(results['pending'])}  errors {len(results['error_ids'])}  "
          f"empty patches {len(results['empty_patch'])}  patch failed {len(results['patch_failed'])}")
    if "gold_valid_ids" in results:
        print(f"  gold check: {len(results['gold_valid_ids'])} valid, "
              f"{len(results['gold_failed_ids'])} fail even with the gold patch, "
              f"{len(results['gold_error_ids'])} errored, {len(results['gold_unchecked_ids'])} unchecked"
              + (f"; {results['resolved_pct_of_gold_valid']}% of the valid ones resolved"
                 if results.get("resolved_pct_of_gold_valid") is not None else ""))
        if results["gold_failed_ids"]:
            print("  fail with gold patch:", ", ".join(results["gold_failed_ids"]))


# ── the scoring pipeline (shared by agent patches and gold patches) ──

def score_folder(out: Path, preds: dict[str, dict], dataset: str, run_name: str, args) -> None:
    ledger = Ledger(out / "score_ledger.jsonl")
    state = ledger.state()
    # Empty patches cannot resolve anything; record them without starting Docker.
    for iid, row in preds.items():
        ev = state.get(iid)
        psha = sha(row["model_patch"])
        if not row["model_patch"].strip() and (ev is None or ev.get("patch_sha") != psha
                                               or ev["status"] == "pending"):
            ledger.add(iid, "empty_patch", patch_sha=psha)
    state = ledger.state()
    todo = [i for i in select_pending(preds, state, args.retry_errors)
            if preds[i]["model_patch"].strip()]
    print(f"[{run_name}] {len(preds)} submitted, {len(todo)} to score now", flush=True)

    if args.method == "sbcli" and todo:
        report = run_sbcli(out, out / "preds.jsonl", dataset, run_name)
        resolved = set(report.get("resolved_ids", []))
        for iid in todo:
            ledger.add(iid, "resolved" if iid in resolved else "unresolved",
                       patch_sha=sha(preds[iid]["model_patch"]), method="sbcli")
        todo = []

    namespace = args.namespace
    size = args.batch if args.batch > 0 else max(len(todo), 1)
    batches = [todo[i:i + size] for i in range(0, len(todo), size)]
    for k, batch in enumerate(batches, 1):
        images: list[str] = []
        if namespace:
            wanted = {iid: image_name(namespace, iid) for iid in batch}
            # Images already in Docker (for example a bundle loaded earlier) are used as they
            # are: not pulled, and never removed afterwards. Only what we pull is ours to delete.
            present = {n for n in wanted.values() if image_exists(n)}
            missing = [n for n in wanted.values() if n not in present]
            if missing:
                free = docker_free_gb()
                need = len(missing) * GB_PER_PREBUILT_IMAGE + 10
                if free is not None and free < need:
                    sys.exit(f"Docker has {free:.0f} GB free; pulling {len(missing)} prebuilt images "
                             f"needs about {need:.0f} GB. Lower --batch or raise Docker's disk limit.")
            if getattr(args, "no_pull", False):
                print(f"{len(present)} prebuilt image(s) present; pulling is disabled, "
                      f"{len(missing)} missing", flush=True)
                ok = {n: n in present for n in wanted.values()}
                images = []
            else:
                print(f"{len(present)} prebuilt image(s) already present, pulling {len(missing)}…", flush=True)
                ok = {n: True for n in present}
                ok.update(pull_images(missing))
                images = missing
            for iid, name in wanted.items():
                if not ok[name]:
                    why = (f"image {name} is not local and pulling is disabled"
                           if getattr(args, "no_pull", False) else f"could not pull {name}")
                    ledger.add(iid, "error", patch_sha=sha(preds[iid]["model_patch"]), reason=why)
            batch = [i for i in batch if ok[wanted[i]]]
            if not batch:
                remove_images(images)
                continue
        run_id = batch_run_id(run_name, batch, preds)
        print(f"\n== [{run_name}] batch {k}/{len(batches)}: {len(batch)} instances ({run_id}) ==",
              flush=True)
        interrupted = False
        try:
            rc = run_local_batch(out, batch, preds, dataset, run_id, args.workers, namespace)
        except KeyboardInterrupt:
            rc, interrupted = 130, True
        finally:
            if images and not args.keep_images:
                remove_images(images)
        for iid, st in harvest(out, run_id, batch).items():
            ledger.add(iid, st, patch_sha=sha(preds[iid]["model_patch"]), batch=run_id)
        if rc == 0:
            have = ledger.state()
            for iid in batch:   # harness finished but wrote no report: a harness-side error
                if have.get(iid, {}).get("status") not in DONE:
                    ledger.add(iid, "error", patch_sha=sha(preds[iid]["model_patch"]),
                               batch=run_id, reason="no report from harness")
        results = write_results(out, preds, ledger, dataset, args.method, run_name)
        print_status(results)
        if interrupted or rc != 0:
            print(f"\nBatch stopped (exit {rc}). Finished instances are saved. "
                  "Run the same command again to carry on.")
            sys.exit(rc or 1)

    print()
    print_status(write_results(out, preds, ledger, dataset, args.method, run_name))
    print(f"wrote {out / 'results.json'}")


def make_gold_preds(out: Path, dataset: str, ids: list[str]) -> dict[str, dict]:
    """Write <out>/gold/preds.jsonl from the dataset's own correct patches."""
    want = set(ids)
    gold = {r["instance_id"]: r for r in data.load_rows(dataset) if r["instance_id"] in want}
    rows = {i: {"instance_id": i, "model_name_or_path": "gold", "model_patch": gold[i]["patch"]}
            for i in ids if i in gold}
    d = out / "gold"
    d.mkdir(exist_ok=True)
    (d / "preds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
    return rows


# ── main ─────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="run folder containing preds.jsonl")
    p.add_argument("--dataset", default="verified")
    p.add_argument("--method", choices=["local", "sbcli"], default="local")
    p.add_argument("--batch", type=int, default=50, help="instances per batch (0 = all at once)")
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) // 2)))
    p.add_argument("--namespace", default="",
                   help="docker image namespace; '' builds locally (needed on arm64)")
    p.add_argument("--prebuilt", action="store_true",
                   help=f"use the official prebuilt images (--namespace {PREBUILT_NAMESPACE}), "
                        "pulled and removed per batch")
    p.add_argument("--keep-images", action="store_true")
    p.add_argument("--no-pull", action="store_true",
                   help="never pull images; an instance whose image is not already local is recorded as an error")
    p.add_argument("--validate-gold", action="store_true")
    p.add_argument("--gold-only", action="store_true")
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--reset", metavar="ids|errors|all",
                   help="comma-separated instance ids, or 'errors', or 'all'")
    args = p.parse_args()
    if args.prebuilt and not args.namespace:
        args.namespace = PREBUILT_NAMESPACE

    out = Path(args.out).expanduser().resolve()
    if not (out / "preds.jsonl").exists():
        raise SystemExit(f"no preds.jsonl in {out}")
    preds = load_preds(out)
    if not preds:
        raise SystemExit("preds.jsonl is empty")
    dataset = data.resolve(args.dataset)
    ledger = Ledger(out / "score_ledger.jsonl")
    run_name = out.name

    if args.reset:
        state = ledger.state()
        if args.reset == "all":
            ids = list(state)
        elif args.reset == "errors":
            ids = [i for i, e in state.items() if e["status"] == "error"]
        else:
            ids = [i for i in args.reset.split(",") if i in preds]
        for iid in ids:
            ledger.add(iid, "pending", reason="reset")
        print(f"reset {len(ids)} instance(s); history kept in {ledger.path.name}")

    if args.status or args.reset:
        print_status(write_results(out, preds, ledger, dataset, args.method, run_name))
        return

    if args.validate_gold or args.gold_only:
        gold_preds = make_gold_preds(out, dataset, list(preds))
        missing = [i for i in preds if i not in gold_preds]
        if missing:
            print(f"no gold patch in the dataset for: {', '.join(missing)}")
        score_folder(out / "gold", gold_preds, dataset, run_name + "-gold", args)
        if args.gold_only:
            print_status(write_results(out, preds, Ledger(out / "score_ledger.jsonl"),
                                       dataset, args.method, run_name))
            return

    score_folder(out, preds, dataset, run_name, args)


if __name__ == "__main__":
    main()
