#!/usr/bin/env python
"""Summarise a MacArena run folder (``results.jsonl``).

Scores are reported three ways, because the choice changes the number:
  best-of-runs  the MacArena paper's protocol: each task keeps the better of its runs (pass@k-like)
  mean-of-runs  the average over runs (pass@1 in expectation): the honest per-attempt number
  run-1         the first run only

Runs that failed for infrastructure reasons (VM would not reset, bridge down) are NOT counted as agent
failures: they are excluded from every rate and reported as a coverage gap. A task is "solved" at score 1.0.

A second, adjusted line leaves out runs whose task setup failed (``setup_status`` "failed", from the bridge's
sync setup) and tasks whose grader is confirmed broken (``known_bad.json``). The first line is always the score
as measured; quote the adjusted one only next to it, with the counts it left out.

    python scripts/macarena/report.py ../otto-macarena-runs/out/<run>
    python scripts/macarena/report.py <shard 1 folder> <shard 2 folder>    # one selection run on two VMs

With several folders the rows are scored together (they must not repeat a task's run) and the summary is printed,
not written into any of them.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean


KNOWN_BAD = Path(__file__).with_name("known_bad.json")


def known_bad(path: Path = KNOWN_BAD) -> dict[str, dict]:
    """Task id -> entry from ``known_bad.json`` (``status`` confirmed / suspected)."""
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")).get("tasks") or {})
    except (OSError, ValueError):
        return {}


def adjusted_rows(rows: list[dict], bad: dict[str, dict]) -> tuple[list[dict], dict]:
    """Rows without failed setups and without tasks whose grader is confirmed broken, and what was left out."""
    confirmed = {t for t, e in bad.items() if e.get("status") == "confirmed"}
    setup_failed = [r for r in rows if r.get("setup_status") == "failed"]
    graders = sorted({r["id"] for r in rows if r["id"] in confirmed})
    keep = [r for r in rows if r.get("setup_status") != "failed" and r["id"] not in confirmed]
    suspected = sorted({r["id"] for r in rows if (bad.get(r["id"]) or {}).get("status") == "suspected"})
    return keep, {"setup_failed_runs": len(setup_failed), "bad_grader_tasks": graders,
                  "suspected_grader_tasks": suspected,
                  "setup_errors": sorted({f"{r['id'][:8]}: {r.get('setup_error')}" for r in setup_failed})}


def load(path: Path) -> list[dict]:
    rows: dict[tuple, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[(r["id"], r.get("run", 1))] = r           # a re-run of the same (task, run) replaces the old row
    return list(rows.values())


def per_task(rows: list[dict]) -> list[dict]:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["id"]].append(r)
    out = []
    for tid, rs in by.items():
        rs.sort(key=lambda r: r.get("run", 1))
        ok = [r for r in rs if r.get("status") != "infra_error"]
        first = next((r for r in ok if r.get("run", 1) == 1), None)
        out.append({
            "id": tid, "source": rs[0].get("source", "?"), "domain": rs[0].get("domain", "?"),
            "infeasible": bool(rs[0].get("infeasible")),
            "runs": len(rs), "valid_runs": len(ok),
            "best": max((r["score"] for r in ok), default=None),
            "mean": mean(r["score"] for r in ok) if ok else None,
            "first": first["score"] if first else None,
            "steps": [r.get("steps", 0) for r in ok],
            "timeouts": sum(1 for r in ok if r.get("status") == "timeout"),
            "capped": sum(1 for r in ok if r.get("status") == "agent_limit"),
        })
    return out


def rate(tasks: list[dict], key: str) -> dict:
    vals = [t[key] for t in tasks if t[key] is not None]
    if not vals:
        return {"n": 0, "mean_score": None, "solved": None}
    return {"n": len(vals), "mean_score": round(mean(vals), 4), "solved": round(sum(v >= 1.0 for v in vals) / len(vals), 4)}


def dialog_stats(rows: list[dict], cap: int) -> dict | None:
    """Model calls and dialog messages per run (rows that recorded them: Otto-native runs from after the
    backend reported them). macOSWorld's limit, "15 screenshots or 30 dialog turns", is a loop of 15 model calls;
    a run "over" counts the runs that used more steps than the cap (should be none) or more than twice the cap in
    dialog messages (possible in macOSWorld's own loop too, when a response makes several tool calls)."""
    have = [r for r in rows if "model_calls" in r and r.get("status") != "infra_error"]
    if not have:
        return None
    mc = [int(r.get("model_calls") or 0) for r in have]
    dm = [int(r.get("dialog_messages") or 0) for r in have]
    return {"runs": len(have), "model_calls_max": max(mc), "model_calls_mean": round(mean(mc), 2),
            "dialog_messages_max": max(dm), "dialog_messages_mean": round(mean(dm), 2),
            "runs_over_step_cap": sum(1 for r in have if cap and int(r.get("steps") or 0) > cap),
            "runs_over_twice_cap_messages": sum(1 for d in dm if cap and d > 2 * cap)}


def summarize(rows: list[dict]) -> dict:
    tracks = sorted({r.get("track", "standard") for r in rows})
    if len(tracks) > 1:
        raise ValueError(f"results from more than one track ({tracks}) cannot be scored together")
    caps = sorted({r.get("max_steps", 15) for r in rows})          # rows from before this field used 15
    if len(caps) > 1:
        raise ValueError(f"results with different step caps ({caps}; 0 = no limit) cannot be scored together")
    units = sorted({r.get("step_unit", "actions") for r in rows})       # rows from before this field: actions
    if len(units) > 1:
        raise ValueError(f"results counted in different step units ({units}) cannot be scored together")
    # Otto-native harness settings (rows from before these fields: observe_after off, no failure notices).
    # How many VMs ran at once: agents on two VMs share one model server, so each is slower (and closer to its
    # time limit). Rows from before this field ran alone.
    parallel = sorted({int(r.get("parallel") or 1) for r in rows})
    if len(parallel) > 1:
        raise ValueError(f"results run with different numbers of VMs at once ({parallel}) cannot be scored together")
    harness = sorted({(bool(r.get("observe_after")), int(r.get("fail_notice_after") or 0),
                       int(r.get("repeat_notice_after") or 0), int(r.get("finish_nudge") or 0),
                       bool(r.get("dismiss_nags")))
                      for r in rows if r.get("track") == "native"})
    if len(harness) > 1:
        raise ValueError("results with different harness settings (observe_after, fail_notice_after, "
                         f"repeat_notice_after, finish_nudge, dismiss_nags: {harness}) cannot be scored together")
    # Procedural memory: rows served from different snapshots (or none) are different arms of an experiment.
    memory = sorted({(r.get("memory_mode") or "off", r.get("snapshot_id"), r.get("memory_floor"),
                      bool(r.get("memory_same_task"))) for r in rows}, key=str)
    if len(memory) > 1:
        raise ValueError(f"results with different procedural memory ({memory}) cannot be scored together; "
                         "report each arm separately")
    # How each task's setup ran: rows from before this field started it without waiting (MacArena's way).
    setup_modes = sorted({r.get("setup_mode") or "launch" for r in rows})
    if len(setup_modes) > 1:
        raise ValueError(f"results with different setup modes ({setup_modes}) cannot be scored together")
    leaked = sum(1 for r in rows if r.get("memory_leak"))
    if leaked:
        if all(r.get("memory_same_task") for r in rows if r.get("memory_leak")):
            raise ValueError(f"{leaked} run(s) recalled their own task's stored run, as asked (same-task recall smoke "
                             "test, MACARENA_MEMORY_SAME_TASK=1): that is not a score; use show_memory.py to read it")
        raise ValueError(f"{leaked} run(s) were served their own task's memory (memory_leak); the snapshot is not "
                         "valid for these tasks")
    tasks = per_task(rows)
    keep, left_out = adjusted_rows(rows, known_bad())
    kept = per_task(keep)
    out: dict = {
        "setup_mode": setup_modes[0] if setup_modes else "launch",
        "setup_checked_runs": sum(1 for r in rows if r.get("setup_status") in ("ok", "failed")),
        "adjusted": {"left_out": left_out, "tasks": len(kept),
                     **{k: rate(kept, key) for k, key in (("best_of_runs", "best"), ("mean_of_runs", "mean"))}},
        "track": tracks[0] if tracks else "standard",
        "max_steps": caps[0] if caps else 15, "step_unit": units[0] if units else "actions",
        "parallel": parallel[0] if parallel else 1,
        "tasks": len(tasks), "runs": len(rows),
        "infra_error_runs": sum(1 for r in rows if r.get("status") == "infra_error"),
        "tasks_with_no_valid_run": sum(1 for t in tasks if t["valid_runs"] == 0),
        "overall": {k: rate(tasks, key) for k, key in (("best_of_runs", "best"), ("mean_of_runs", "mean"), ("run_1", "first"))},
        "mean_steps_per_run": round(mean(s for t in tasks for s in t["steps"]), 2) if any(t["steps"] for t in tasks) else None,
        "timeout_runs": sum(t["timeouts"] for t in tasks),
        "agent_limit_runs": sum(t["capped"] for t in tasks),
        "tool_call_cap": max((r.get("tool_call_cap") or 0 for r in rows), default=0),
        "harness": ({"observe_after": harness[0][0], "fail_notice_after": harness[0][1],
                     "repeat_notice_after": harness[0][2], "finish_nudge": harness[0][3],
                     "dismiss_nags": harness[0][4]} if harness else None),
        "host_slept_runs": sum(1 for r in rows if (r.get("host_slept_s") or 0) > 60),
        "dialog": dialog_stats(rows, caps[0] if caps else 15),
        "memory": memory_stats(rows, memory[0] if memory else ("off", None)),
        "by_source": {}, "by_domain": {},
    }
    for field, name in (("source", "by_source"), ("domain", "by_domain")):
        groups: dict[str, list[dict]] = defaultdict(list)
        for t in tasks:
            groups[t[field]].append(t)
        for g, ts in sorted(groups.items()):
            out[name][g] = {"tasks": len(ts), **{k: rate(ts, key) for k, key in (("best_of_runs", "best"), ("mean_of_runs", "mean"))}}
    inf = [t for t in tasks if t["infeasible"]]
    if inf:
        out["infeasible_tasks"] = {"n": len(inf), "mean_of_runs": rate(inf, "mean")}
    return out


def memory_stats(rows: list[dict], arm: tuple) -> dict | None:
    """What procedural memory did in this arm: how often it was called and returned something, the pass rate of runs
    it returned something for vs runs where it abstained or was not called, and the items used most."""
    mode, snapshot = arm[0], arm[1]
    if mode == "off":
        return None
    valid = [r for r in rows if r.get("status") != "infra_error"]
    used = [r for r in valid if r.get("memory_used")]
    unused = [r for r in valid if not r.get("memory_used")]

    def pass_rate(rs: list[dict]) -> float | None:
        return round(sum(1 for r in rs if (r.get("score") or 0) >= 1.0) / len(rs), 4) if rs else None

    items: dict[str, list[float]] = defaultdict(list)
    for r in used:
        for u in r["memory_used"]:
            items[str(u.get("item"))].append(float(r.get("score") or 0))
    top = sorted(items.items(), key=lambda kv: -len(kv[1]))[:10]
    return {"mode": mode, "snapshot_id": snapshot, "floor": rows[0].get("memory_floor") if rows else None,
            "snapshot_size": max((int(r.get("snapshot_size") or 0) for r in rows), default=0),
            "runs_calling": sum(1 for r in valid if r.get("memory_calls")),
            "runs_returned": len(used), "calls": sum(int(r.get("memory_calls") or 0) for r in valid),
            "abstained_calls": sum(int(r.get("memory_abstained") or 0) for r in valid),
            "mean_tokens_per_run": round(mean(int(r.get("memory_tokens") or 0) for r in valid), 1) if valid else None,
            "pass_rate_returned": pass_rate(used), "pass_rate_not_returned": pass_rate(unused),
            "top_items": [{"item": k, "uses": len(v), "pass_rate": round(sum(1 for x in v if x >= 1.0) / len(v), 3)}
                          for k, v in top]}


def render(s: dict) -> str:
    def pct(x):
        return "n/a" if x is None else f"{100 * x:.1f}%"
    o = s["overall"]
    track = s.get("track", "standard")
    lines = [
        f"track: {track}" + ("  (Otto's macOS desktop agent and tools; NOT comparable with the paper's baselines)"
                              if track == "native" else "  (MacArena's screenshot + mouse/keyboard action space)"),
        "step limit: " + (str(s.get("max_steps", 15)) if s.get("max_steps", 15) else
                          "none  (NOT comparable with the paper's 15-step results)")
        + (f" {s.get('step_unit', 'actions')}" if s.get("max_steps", 15) else "")
        + (f"   tool-call cap: {s['tool_call_cap']} per run (reads included)" if s.get("tool_call_cap") else "")
        + (f"   VMs at once: {s['parallel']} (agents shared the model server)" if s.get("parallel", 1) > 1 else ""),
        *([f"per run: model calls max {d['model_calls_max']} (mean {d['model_calls_mean']}), dialog messages max "
            f"{d['dialog_messages_max']} (mean {d['dialog_messages_mean']}); over the step cap: {d['runs_over_step_cap']}, "
            f"over {2 * s.get('max_steps', 15)} messages: {d['runs_over_twice_cap_messages']}  (n={d['runs']})"]
          if (d := s.get("dialog")) else []),
        *([f"harness: observe_after={'on' if s['harness']['observe_after'] else 'off'}   "
            f"failure notice after {s['harness']['fail_notice_after'] or 'never'}   "
            f"repeat notice after {s['harness'].get('repeat_notice_after') or 'never'}   "
            f"finish nudges {s['harness'].get('finish_nudge') or 0}   "
            f"update nags {'closed' if s['harness'].get('dismiss_nags') else 'left open'}"] if s.get("harness") else []),
        *([f"procedural memory: {m['mode']} "
            + ("(each task could recall the earlier tasks of the run: order-dependent, a learning curve)   "
               if m["mode"] == "live" else f"({m['snapshot_size']} items)   ")
            + f"runs calling it {m['runs_calling']}, "
            f"returned something {m['runs_returned']}   calls {m['calls']} (abstained {m['abstained_calls']})   "
            f"mean tokens/run {m['mean_tokens_per_run']}   pass rate when returned "
            f"{pct(m['pass_rate_returned'])} vs not {pct(m['pass_rate_not_returned'])}"]
          if (m := s.get("memory")) else ["procedural memory: off"]),
        f"tasks {s['tasks']}   runs {s['runs']}   infra-error runs {s['infra_error_runs']}   "
        f"tasks with no valid run {s['tasks_with_no_valid_run']}   timeouts {s['timeout_runs']}   cut at the tool-call cap {s.get('agent_limit_runs', 0)}   mean steps/run {s['mean_steps_per_run']}",
        "",
        f"best-of-runs (paper protocol): mean score {pct(o['best_of_runs']['mean_score'])}  solved {pct(o['best_of_runs']['solved'])}  (n={o['best_of_runs']['n']})",
        f"mean-of-runs (per attempt)   : mean score {pct(o['mean_of_runs']['mean_score'])}  solved {pct(o['mean_of_runs']['solved'])}  (n={o['mean_of_runs']['n']})",
        f"run 1 only                   : mean score {pct(o['run_1']['mean_score'])}  solved {pct(o['run_1']['solved'])}  (n={o['run_1']['n']})",
        "",
    ]
    if (a := s.get("adjusted")) is not None:
        lo = a["left_out"]
        lines += [
            f"task setup: {s.get('setup_mode', 'launch')}"
            + (f" (checked in {s['setup_checked_runs']} runs, failed in {lo['setup_failed_runs']})"
               if s.get("setup_checked_runs") else " (not checked: failures are invisible in these rows)"),
            f"adjusted (without failed setups and confirmed bad graders): mean-of-runs "
            f"{pct(a['mean_of_runs']['mean_score'])} (n={a['mean_of_runs']['n']})   best-of-runs "
            f"{pct(a['best_of_runs']['mean_score'])}   left out: {lo['setup_failed_runs']} run(s) with a failed "
            f"setup, {len(lo['bad_grader_tasks'])} task(s) with a confirmed bad grader. Quote it only beside the "
            "line above.",
            *[f"  setup failed  {e}" for e in lo["setup_errors"]],
            *[f"  bad grader    {t[:8]} (known_bad.json)" for t in lo["bad_grader_tasks"]],
            *[f"  suspected     {t[:8]} grader (known_bad.json; NOT left out until checked)"
              for t in lo["suspected_grader_tasks"]],
            "",
        ]
    for name in ("by_source", "by_domain"):
        lines.append(name.replace("_", " "))
        for g, v in s[name].items():
            lines.append(f"  {g:<28} tasks {v['tasks']:>3}   mean-of-runs {pct(v['mean_of_runs']['mean_score']):>6}   best-of-runs {pct(v['best_of_runs']['mean_score']):>6}")
        lines.append("")
    if s.get("infeasible_tasks"):
        lines.append(f"infeasible tasks: {s['infeasible_tasks']}")
    if s.get("host_slept_runs"):
        lines.append(f"WARNING: the host slept for more than a minute during {s['host_slept_runs']} run(s); their "
                     "time limit was partly spent asleep.")
    if s["infra_error_runs"] or s["tasks_with_no_valid_run"]:
        lines.append("WARNING: some runs failed for infrastructure reasons and are excluded from the rates above. "
                     "Re-run them before quoting a number.")
    if track == "native":
        lines.append("Otto-native track: compare only with other Otto-native runs at the same step cap and run count "
                     "(see scripts/macarena/README.md).")
    else:
        lines.append("Compare with published numbers only at the same step cap, observation type and run count "
                     "(see scripts/macarena/README.md).")
    return "\n".join(lines)


# ── Live status (written while a run is in progress) ────────────────────────────────────────────────────────────
def _bucket(rows: list[dict], planned: int) -> dict:
    """Counts and score for one group. ``rows`` are the finished runs of the group (latest row per task and run);
    runs that failed for infrastructure reasons are listed but kept out of the score, as in ``summarize``."""
    valid = [r for r in rows if r.get("status") != "infra_error"]
    scores = [float(r.get("score") or 0) for r in valid]
    return {
        "planned_runs": planned, "finished_runs": len(rows), "remaining_runs": max(planned - len(rows), 0),
        "scored_runs": len(valid), "infra_error_runs": len(rows) - len(valid),
        "passed": sum(s >= 1.0 for s in scores), "partial": sum(0 < s < 1.0 for s in scores),
        "failed": sum(s <= 0 for s in scores),
        "score_sum": round(sum(scores), 3),
        "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
        "mean_steps": round(mean(int(r.get("steps") or 0) for r in valid), 1) if valid else None,
        "step_limit_runs": sum(1 for r in valid if r.get("agent_status") == "run_over"),
        "timeout_runs": sum(1 for r in valid if r.get("status") == "timeout"),
    }


def live_status(planned: list[dict], runs_per_task: int, rows: list[dict], current: dict | None,
                started_at: float, now: float) -> dict:
    """Snapshot of a run in progress: overall and per source and domain, how many runs are done, passed, partial,
    failed and left, the running score, and the task being run now. ``planned`` are the task dicts of this run
    (``id``, ``source``, ``domain``); ``rows`` are the finished result rows (any rows for other tasks are ignored)."""
    ids = {t["id"] for t in planned}
    latest = {(r["id"], r.get("run", 1)): r for r in rows if r.get("id") in ids}
    done = list(latest.values())
    total = len(planned) * runs_per_task
    out: dict = {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "elapsed_s": round(now - started_at), "current": current,
        "overall": _bucket(done, total), "by_source": {}, "by_domain": {},
    }
    n_done_here = out["overall"]["finished_runs"]
    out["overall"]["pct_done"] = round(100 * n_done_here / total, 1) if total else None
    for field, name in (("source", "by_source"), ("domain", "by_domain")):
        keys = sorted({t[field] for t in planned})
        for g in keys:
            g_planned = sum(1 for t in planned if t[field] == g) * runs_per_task
            g_rows = [r for r in done if r.get("source" if field == "source" else "domain") == g]
            out[name][g] = _bucket(g_rows, g_planned)
    return out


def render_live(s: dict) -> str:
    def pct(x):
        return "  n/a" if x is None else f"{100 * x:5.1f}%"
    o = s["overall"]
    cur = s.get("current")
    lines = [
        f"updated {s['updated']}   elapsed {s['elapsed_s'] // 3600}h{s['elapsed_s'] % 3600 // 60:02d}m",
        f"running now: {cur['id']} ({cur['source']}/{cur['domain']}, run {cur['run']}, "
        f"{(cur.get('instruction') or '')[:70]})" if cur else "running now: nothing",
        "",
        f"overall   finished {o['finished_runs']}/{o['planned_runs']} ({o.get('pct_done')}%)   left {o['remaining_runs']}   "
        f"score {o['score_sum']} = {pct(o['mean_score']).strip()}   passed {o['passed']}   partial {o['partial']}   "
        f"failed {o['failed']}   infra errors {o['infra_error_runs']}",
        "",
    ]
    for name in ("by_source", "by_domain"):
        lines.append(name.replace("_", " "))
        lines.append(f"  {'group':<22}{'done':>9}{'left':>6}{'score':>8}{'mean':>8}{'pass':>6}{'part':>6}{'fail':>6}{'steps':>7}{'capped':>8}")
        for g, v in s[name].items():
            lines.append(f"  {g:<22}{str(v['finished_runs']) + '/' + str(v['planned_runs']):>9}{v['remaining_runs']:>6}"
                         f"{v['score_sum']:>8.1f}{pct(v['mean_score']):>8}{v['passed']:>6}{v['partial']:>6}{v['failed']:>6}"
                         f"{(v['mean_steps'] if v['mean_steps'] is not None else 0):>7.0f}{v['step_limit_runs']:>8}")
        lines.append("")
    return "\n".join(lines)


def write_live(folder: Path, planned: list[dict], runs_per_task: int, rows: list[dict], current: dict | None,
               started_at: float) -> None:
    """Write ``live_status.json`` and ``live_status.txt`` into the run folder, atomically so a reader never sees half
    a file. Never raises: a stats file must not stop a run."""
    try:
        s = live_status(planned, runs_per_task, rows, current, started_at, time.time())
        for name, text in (("live_status.json", json.dumps(s, indent=2)), ("live_status.txt", render_live(s) + "\n")):
            tmp = folder / (name + ".tmp")
            tmp.write_text(text)
            tmp.replace(folder / name)
    except Exception as exc:  # noqa: BLE001
        print(f"(could not write the live status: {exc})", file=sys.stderr)


def load_many(folders: list[Path]) -> list[dict]:
    """Every folder's rows; a task's run found in two folders is an error (shards are disjoint)."""
    seen: dict[tuple, Path] = {}
    rows: list[dict] = []
    for folder in folders:
        for r in load(folder / "results.jsonl"):
            key = (r["id"], r.get("run", 1))
            if key in seen:
                raise ValueError(f"task {r['id']} run {key[1]} is in both {seen[key]} and {folder}; "
                                 "report folders that ran different tasks")
            seen[key] = folder
            rows.append(r)
    return rows


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    folders = [Path(a) for a in sys.argv[1:]]
    try:
        s = summarize(load_many(folders))
    except ValueError as exc:
        sys.exit(f"cannot report: {exc}")
    text = render(s)
    if len(folders) == 1:
        (folders[0] / "summary.json").write_text(json.dumps(s, indent=2))
        (folders[0] / "summary.txt").write_text(text + "\n")
    else:
        text = f"combined: {', '.join(str(f) for f in folders)}\n" + text
    print(text)


if __name__ == "__main__":
    main()
