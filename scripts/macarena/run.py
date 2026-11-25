#!/usr/bin/env python
"""Run Otto on MacArena tasks through Otto's local backend, scoring with MacArena's evaluators.

Two tracks, set by the bridge (``bridge.py`` / ``bench.sh up``), not here:
  standard  ``macarena-agent``: screenshot + mouse/keyboard, MacArena's own action space. Comparable with the paper.
  native    ``macarena-desktop-agent``: Otto's macOS desktop agent (its prompt and Accessibility tools), the
            tool calls run in the VM. Measures Otto's agent; not comparable with the paper's baselines.

Runs in Otto's venv. It never imports MacArena: it talks to the bridge (``bridge.py``, in MacArena's own
environment), which owns the VM, and to an isolated Otto backend (``start_backend.sh``).

Per task and run: bridge reset (fresh VM copy) -> one Otto session with the task text -> bridge evaluate.
Defaults follow the MacArena paper's protocol where it is stated: 15 steps, 2 runs per task. Where this
harness differs (a step is one action; a fresh VM for every run) it is stricter or cleaner, and
README.md says how that affects comparison.

    python scripts/macarena/run.py --macarena-dir ../MacArena --sources macarena --sample 5
    python scripts/macarena/run.py --macarena-dir ../MacArena --sources macosworld --per-group 2 --seed 0
    python scripts/macarena/run.py --macarena-dir ../MacArena --sources macarena --noop   # do-nothing baseline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx
import websockets

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import preflight  # noqa: E402
import report  # noqa: E402

AGENTS = {"standard": "macarena-agent", "native": "macarena-desktop-agent"}
AGENT = AGENTS["standard"]
MAX_RESUMES = 3
NO_USER = {
    "standard": "No user is available. Do the task with the vm_* tools and finish; do not ask questions.",
    "native": "No user is available. Do the task with the desktop tools, then vm_done; do not ask questions.",
}
SOURCES = ("osworld", "macosworld", "macarena")


# ── tasks ───────────────────────────────────────────────────────────────────

def vm_for(source: str) -> str:
    """MacArena's README: OSWorld and macOSWorld tasks use the 'osworld' VM, MacArena's own use 'macarena'."""
    return "macarena" if source == "macarena" else "osworld"


def find_tasks(macarena_dir: Path, sources: list[str]) -> list[dict]:
    base = macarena_dir / "evaluation_examples"
    out = []
    for src in sources:
        root = base / src
        if not root.is_dir():
            raise SystemExit(f"missing {root}")
        for f in sorted(root.rglob("*.json")):
            try:
                t = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(t, dict) or "id" not in t or "instruction" not in t or "evaluator" not in t:
                continue                                    # not a task (macos_test.py also skips these)
            ev = t["evaluator"]
            out.append({"id": t["id"], "source": src, "domain": f.parent.name, "path": str(f),
                        "instruction": t["instruction"], "vm": vm_for(src),
                        "apps": [str(a) for a in (t.get("related_apps") or [])],
                        "infeasible": isinstance(ev, dict) and ev.get("func") == "infeasible"})
    return out


def csv(value: str | None) -> set[str]:
    return {x.strip() for x in (value or "").split(",") if x.strip()}


def excluded(tasks: list[dict], apps: str | None = None, ids: str | None = None) -> list[dict]:
    """Tasks that need one of ``apps`` (case-insensitive, MacArena's ``related_apps``) or are in ``ids``."""
    bad_apps, bad_ids = {a.lower() for a in csv(apps)}, csv(ids)
    return [t for t in tasks if t["id"] in bad_ids or bad_apps & {a.lower() for a in t.get("apps", [])}]


def parse_shard(text: str | None) -> tuple[int, int]:
    """``"K/N"`` as ``(K, N)`` with 1 <= K <= N; ``(1, 1)`` when not given."""
    if not text:
        return 1, 1
    try:
        k, n = (int(x) for x in text.split("/"))
    except ValueError:
        raise SystemExit(f"--shard must look like 1/2, not {text!r}") from None
    if not 1 <= k <= n:
        raise SystemExit(f"--shard {text}: K must be 1 to N")
    return k, n


def shard(tasks: list[dict], k: int, n: int) -> list[dict]:
    """Part ``k`` of ``n`` of the selection, for running it on several VMs at once: every n-th task after sorting,
    so each part gets a share of every domain, and the parts together are the selection exactly once."""
    ordered = sorted(tasks, key=lambda t: (t["source"], t["domain"], t["id"]))
    return ordered[k - 1::n]


def select(tasks: list[dict], ids: str | None, domain: str | None, sample: int | None, seed: int,
           exclude_apps: str | None = None, exclude_ids: str | None = None,
           per_group: int | None = None) -> list[dict]:
    drop = {t["id"] for t in excluded(tasks, exclude_apps, exclude_ids)}
    tasks = [t for t in tasks if t["id"] not in drop]            # before sampling, so the sample is still N tasks
    if ids:
        order = {i: n for n, i in enumerate(ids.split(","))}
        # In the order given: with live memory, a later task can recall the earlier ones.
        tasks = sorted((t for t in tasks if t["id"] in order), key=lambda t: order[t["id"]])
    if domain:
        tasks = [t for t in tasks if t["domain"] == domain]
    if sample and sample < len(tasks):
        by: dict[str, list[dict]] = {}
        for t in tasks:
            by.setdefault(f"{t['source']}/{t['domain']}", []).append(t)
        rng = random.Random(seed)
        picked: list[dict] = []
        groups = sorted(by)
        for g in groups:                                    # stratified: proportional, at least one per group
            k = max(1, round(sample * len(by[g]) / len(tasks)))
            picked += rng.sample(by[g], min(k, len(by[g])))
        rng.shuffle(picked)
        tasks = picked[:sample] if len(picked) >= sample else picked
        tasks.sort(key=lambda t: (t["source"], t["domain"], t["id"]))
    if per_group:
        tasks = sample_per_group(tasks, per_group, seed)
    return tasks


def sample_per_group(tasks: list[dict], n: int, seed: int) -> list[dict]:
    """``n`` random tasks from every source/domain group (all of a group smaller than ``n``). The same seed and
    selection give the same tasks, whatever order they were found in."""
    by: dict[tuple[str, str], list[dict]] = {}
    for t in sorted(tasks, key=lambda t: (t["source"], t["domain"], t["id"])):
        by.setdefault((t["source"], t["domain"]), []).append(t)
    rng = random.Random(seed)
    picked = [t for g in sorted(by) for t in rng.sample(by[g], min(n, len(by[g])))]
    return sorted(picked, key=lambda t: (t["source"], t["domain"], t["id"]))


# ── Otto API (same protocol as scripts/swebench and scripts/benchcad) ───────

class Api:
    def __init__(self, base: str, token: str | None):
        self.base = base.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.http = httpx.AsyncClient(base_url=self.base, headers=self.headers, timeout=60)

    async def call(self, method: str, path: str, **kw):
        r = await self.http.request(method, path, **kw)
        r.raise_for_status()
        return r.json() if r.content else {}

    def ws_url(self, sid: str) -> str:
        scheme = "wss" if self.base.startswith("https") else "ws"
        url = f"{scheme}://{self.base.split('://', 1)[1]}/ws/chat/{sid}"
        tok = self.headers.get("Authorization", "")[7:]
        return f"{url}?token={tok}" if tok else url


class Bridge:
    def __init__(self, base: str, token: str):
        self.http = httpx.AsyncClient(base_url=base.rstrip("/"), headers={"Authorization": f"Bearer {token}"}, timeout=600)

    async def call(self, path: str, body: dict | None = None) -> dict:
        r = await (self.http.post(path, json=body) if body is not None else self.http.get(path))
        if r.status_code >= 400:
            raise RuntimeError(f"bridge {path}: HTTP {r.status_code} {r.text[:300]}")
        return r.json()


def log_event(path: Path, msg: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps({"ts": round(time.time(), 3), **msg}, default=str) + "\n")


# Otto's own per-run cap on tool calls (reads and finishing calls included), set by start_backend.sh. Without a
# step limit this is what ends a run that neither finishes nor times out; 0 means the cap is off.
TOOL_CALL_CAP = int(os.environ.get("MACARENA_TOOL_BUDGET", "100"))
STATE_CHECK_EVERY = 2.0      # seconds between looks at the bridge while the agent is busy
# Times the agent is told the run is not finished when its turn ends with no vm_done / vm_fail and steps are left.
# 19 of 101 failed runs in one OSWorld set ended that way: an answer written in chat (the check reads the screen),
# a "Done" with no vm_done, or a turn cut short. 0 turns it off; recorded with every run.
FINISH_NUDGES = int(os.environ.get("MACARENA_FINISH_NUDGES", "1"))
FINISH_NUDGE = ("The run is not finished: you have not called vm_done or vm_fail, and nobody will read your reply. "
                "Look at the screen once more. If it shows the task done as asked (the page in the front tab, the "
                "cells, the setting or the file the task names), call vm_done. If the task cannot be done here, "
                "call the vm_fail tool with the reason (writing \"vm_fail\" in a reply does nothing). Otherwise carry "
                "on with the task.")
NUDGE_MIN_SECONDS = 60       # time a nudged turn needs to be worth starting
OVER_GRACE = 4.0             # seconds the agent may keep going after the bridge says the run is over


def should_nudge(state: dict, nudged: int, nudges: int, seconds_left: float) -> bool:
    """Tell the agent the run is not finished? Only when its turn ended without vm_done / vm_fail, the bridge says
    the run is still open with steps left, the nudges are not used up, and there is time for another turn."""
    if nudged >= nudges or seconds_left < NUDGE_MIN_SECONDS or state.get("finished"):
        return False
    cap = int(state.get("max_steps") or 0)
    return cap == 0 or int(state.get("steps") or 0) < cap


async def drive(api: Api, bridge: Bridge, sid: str, text: str, deadline: float, events: Path,
                track: str = "standard", nudges: int = 0) -> dict:
    """Send the task and answer approvals. Ends when the agent finishes, the run is over on the bridge
    (budget spent, done, fail), or the deadline passes. When the agent's turn ends with the run still open,
    it is told so up to ``nudges`` times (``FINISH_NUDGE``)."""
    stats = {"status": "timeout", "error": None}
    nudged = 0
    async with websockets.connect(api.ws_url(sid), max_size=None) as ws:
        await ws.send(json.dumps({"content": text, "depth": "deep", "mode": "agent"}))
        over_since = None
        last_check = 0.0

        async def run_is_over_for_too_long() -> bool:
            """The bridge knows when the run is over; do not wait for a model that keeps going. Checked on a
            clock, not only when the socket is quiet: an agent that keeps calling tools after the run ended
            (benchmark runs showed 40 to 77 repeated vm_done / vm_fail calls) never leaves the socket quiet."""
            nonlocal over_since, last_check
            last_check = time.time()
            if (await bridge.call("/ctl/state")).get("finished"):
                over_since = over_since or time.time()
                return time.time() - over_since > OVER_GRACE
            return False

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                await api.call("POST", f"/api/sessions/{sid}/stop")
                return stats
            if time.time() - last_check >= STATE_CHECK_EVERY and await run_is_over_for_too_long():
                await api.call("POST", f"/api/sessions/{sid}/stop")
                stats["status"] = "run_over"
                return stats
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=min(remaining, 5)))
            except asyncio.TimeoutError:
                continue
            log_event(events, msg)
            kind, meta = msg.get("type"), msg.get("metadata") or {}
            if kind == "tool_call":
                stats["tool_calls"] = stats.get("tool_calls", 0) + 1
            if kind == "done" and nudges and should_nudge(await bridge.call("/ctl/state"), nudged, nudges,
                                                          deadline - time.time()):
                nudged += 1
                stats["finish_nudges"] = nudged
                await ws.send(json.dumps({"content": FINISH_NUDGE, "depth": "deep", "mode": "agent"}))
                continue
            if kind in ("done", "stopped"):
                stats["status"] = kind
                return stats
            if kind == "error":
                stats["errors"] = stats.get("errors", 0) + 1
                stats["error"] = msg.get("content")
                if stats["errors"] <= MAX_RESUMES and time.time() < deadline - 30:
                    await asyncio.sleep(5)
                    await ws.send(json.dumps({"content": "The connection was interrupted. Continue from "
                                              "where you left off and finish.", "depth": "deep", "mode": "agent"}))
                    continue
                stats["status"] = "agent_error"
                return stats
            if kind == "ask_user":
                await ws.send(json.dumps({"type": "hitl_response", "decisions": [{"answer": NO_USER[track]}],
                                          "request_id": meta.get("request_id")}))
            elif kind == "hitl_request":
                n = len(meta.get("action_requests") or []) or 1
                if meta.get("type") == "request_credential":
                    dec = [{"answer": "cancelled"}] * n
                else:
                    dec = [{"type": "approve"}] * n        # the benchmark agents only reach VM-side tools
                await ws.send(json.dumps({"type": "hitl_response", "decisions": dec,
                                          "request_id": meta.get("request_id")}))


# ── one run of one task ─────────────────────────────────────────────────────

# How steps are counted, as the native agent is told it. "turns" is macOSWorld's rule (one step per model reply that
# calls tools, every call in that reply included); "actions" counts each acting call.
NATIVE_STEP_RULES = {
    "actions": ("Every acting tool call counts as a step (batch_actions: one per action in it); reading tools such "
                "as list_apps and get_screen_controls are free. "),
    "turns": ("A step is one reply of yours that calls tools: every tool call in that reply, reads included, counts "
              "as that one step, and a reply that only reads uses a step too. So put the calls you can already "
              "decide into the same reply (a read together with the action it is for, or several independent "
              "actions, or one batch_actions); an action's result already lists the controls on screen, so do "
              "not spend a reply reading them again. "),
}

# OSWorld and macOSWorld include tasks that cannot be done as asked; their checker credits only an explicit FAIL.
INFEASIBLE_NOTE = ("Some tasks in this set cannot be done as asked (the app has no such feature, or the request "
                   "contradicts itself). If you find that this one cannot, call vm_fail with the reason; do not "
                   "spend the remaining steps on stand-ins. You have found it when you notice the request "
                   "contradicts itself, or when the app the task names has no such feature and what is left is a "
                   "shell script, another program, installing something, or changing the task's own files: call "
                   "vm_fail then, rather than make a stand-in. ")
SOURCES_WITH_INFEASIBLE = frozenset({"osworld", "macosworld"})


def prompt_for(task: dict, max_steps: int, track: str = "standard", step_unit: str = "actions",
               start_state: str = "") -> str:
    """The task message. On the native track ``start_state`` (from the bridge's reset) says what is on screen as
    the task starts: a screenshot agent's first observation, as text, since Otto's desktop agent starts blind."""
    budget = (f"Step budget: {max_steps}. " if max_steps > 0
              else "There is no step limit; take the steps you need, but the run has a time limit. ")
    if track == "native":
        infeasible = INFEASIBLE_NOTE if task.get("source") in SOURCES_WITH_INFEASIBLE else ""
        screen = f"{start_state.strip()}\n\n" if start_state and start_state.strip() else ""
        return (f"Task: {task['instruction']}\n\n{screen}{budget}{NATIVE_STEP_RULES[step_unit]}{infeasible}"
                "Finish with vm_done when the task is done.")
    return (f"Task: {task['instruction']}\n\n{budget}"
            "Every action tool call counts as a step; vm_screenshot is free. Start with vm_screenshot.")


def awake_clock() -> float:
    """Seconds that only advance while the host is awake (macOS CLOCK_UPTIME_RAW); wall time where unavailable."""
    clock = getattr(time, "CLOCK_UPTIME_RAW", None)
    return time.clock_gettime(clock) if clock is not None else time.time()


# A run whose time limit was spent with the host asleep did not get its time: retried on resume, like a VM failure.
SLEPT_TOO_LONG = 60.0


def mark_host_sleep(rec: dict) -> dict:
    """A run that timed out while the host slept for a long time did not get its time limit: make it an
    infrastructure error, which a resume retries. A run that finished on its own keeps its score."""
    if rec.get("agent_status") == "timeout" and rec.get("host_slept_s", 0) > SLEPT_TOO_LONG:
        rec["status"] = "infra_error"
        rec["error"] = (f"the host slept {rec['host_slept_s']:.0f} s during the run's time limit; retried on "
                        "resume (bench.sh run keeps the Mac awake; closing the lid still sleeps it)")
    return rec


def setup_fields(armed) -> dict:
    """What the task's setup did (the bridge's reset answer): its mode, ``ok`` / ``failed`` / ``launched`` (not
    checked) / ``none``, and why it failed. A failed setup is kept as the agent's run, scored as usual; the report
    also gives the score without those runs. Bridges from before this field: launch, not checked."""
    st = (armed or {}) if isinstance(armed, dict) else {}
    setup = st.get("setup") or {}
    out = {"setup_mode": st.get("setup_mode") or "launch", "setup_status": setup.get("status") or "unknown"}
    if setup.get("status") == "failed":
        out["setup_error"] = setup.get("reason")
    if setup.get("attempts"):
        out["setup_attempts"] = setup["attempts"]
    return out


def final_status(agent_status: str | None, noop: bool, eval_error, capped: bool = False) -> str:
    """The run's status once the VM answered. A grader that could not run (the checker command failed in the VM)
    gives no valid score, so it is an infrastructure error, which a resume retries on a fresh VM. ``capped``: the
    agent was cut off at Otto's tool-call cap without finishing; the score is real but the run did not choose to end."""
    if eval_error:
        return "infra_error"
    if noop:
        return "noop"
    if capped and agent_status not in ("timeout", "agent_error"):
        return "agent_limit"
    return agent_status if agent_status in ("timeout", "agent_error") else "ok"


INGEST_TIMEOUT = 900   # seconds: saving a run includes one LLM call for its intents and lessons


async def memory_setup(api: Api | None, out: Path) -> dict:
    """Procedural memory for this run (``backend/procedural_memory``), as the backend was started with it
    (``bench.sh up --memory ...``). Points the backend's write store at ``out/memory/write.db`` so this run's
    episodes stay with its results, keeps a copy of the served snapshot next to them, and returns the fields every
    result row records. A backend without the API (older) counts as memory off."""
    fields = {"memory_mode": "off", "snapshot_id": None, "snapshot_size": 0}
    if api is None:
        return fields
    try:
        st = await api.call("GET", "/api/procedures/status")
    except Exception:  # noqa: BLE001
        return fields
    if st.get("mode") != "benchmark":
        if st.get("mode") == "app":
            raise SystemExit("the benchmark backend is using the app's procedure store; restart it with bench.sh up "
                             "(memory comes only from OTTO_PROCEDURE_* there)")
        return fields
    mem = out / "memory"
    mem.mkdir(parents=True, exist_ok=True)
    await api.call("POST", "/api/procedures/benchmark/write-db", json={"path": str(mem / "write.db")})
    fields["memory_write"] = "memory/write.db"
    if st.get("live"):
        # Served from that same write store: each finished task is recallable by the ones after it. Order matters,
        # so the run's task order is recorded too (results are a learning curve, not a frozen-memory score).
        fields.update(memory_mode="live", memory_floor=st.get("floor"), memory_embedder=st.get("embedder"))
        if st.get("same_task"):
            fields["memory_same_task"] = True
        return fields
    if st.get("serve_tool"):
        if st.get("error"):
            raise SystemExit(f"procedure snapshot: {st['error']}")
        src = Path(str(st.get("serve_path") or ""))
        dst = mem / "serve.db"
        if src.is_file() and not dst.exists():
            import shutil
            shutil.copyfile(src, dst)
        fields.update(memory_mode=f"snapshot:{st.get('snapshot_id') or src.stem}", snapshot_id=st.get("snapshot_id"),
                      snapshot_size=int(st.get("snapshot_size") or 0), memory_floor=st.get("floor"),
                      memory_embedder=st.get("embedder"))
        if st.get("same_task"):
            fields["memory_same_task"] = True    # recall could return this very task's stored run: not a clean score
    return fields


async def memory_after(api: Api | None, rec: dict, st: dict) -> None:
    """Record what memory the task used, save its run with the evaluator's score, and score the items it used."""
    rec.update(memory_calls=st.get("memory_calls", 0), memory_turns=st.get("memory_turns", 0),
               memory_used=st.get("memory_used") or [], memory_abstained=st.get("memory_abstained", 0),
               memory_tokens=st.get("memory_tokens", 0))
    leaked = [u.get("item") for u in rec["memory_used"] if u.get("source_task") == rec["id"]]
    if leaked:          # the snapshot held this very task: the score is not a valid memory-on result
        rec["memory_leak"] = leaked
        print(f"WARNING {rec['id']}: memory returned this task's own run ({leaked}); exclude this row", flush=True)
    if api is None or not rec.get("memory_write") or not rec.get("session_id") or rec.get("status") == "infra_error":
        return
    try:
        await api.call("POST", "/api/procedures/benchmark/outcome",
                       json={"session_id": rec["session_id"], "score": rec["score"]})
        # Saving writes lessons with the LLM, which on a busy local model server takes minutes; with live memory the
        # next task must not start before this one is recallable, so wait for it (a 60 s client timeout used to
        # give up while the backend carried on).
        saved = await api.call("POST", "/api/procedures/benchmark/ingest", timeout=INGEST_TIMEOUT,
                               json={"session_id": rec["session_id"], "score": rec["score"], "task_id": rec["id"]})
        rec["memory_saved"] = saved.get("episodes") or []
    except Exception as exc:  # noqa: BLE001 — bookkeeping never fails the task
        rec["memory_save_error"] = repr(exc)[:300]
        if rec.get("memory_mode") == "live":
            print(f"WARNING {rec['id']}: not saved to live memory ({rec['memory_save_error']}); later tasks "
                  "will not see it", flush=True)


async def run_one(api: Api | None, bridge: Bridge, task: dict, run_idx: int, args, out: Path) -> dict:
    run_dir = out / "tasks" / task["source"] / task["domain"] / task["id"] / f"run_{run_idx}"
    rec: dict = {"id": task["id"], "source": task["source"], "domain": task["domain"], "run": run_idx,
                 "infeasible": task["infeasible"], "score": 0.0, "status": "infra_error", "steps": 0,
                 "protocol": "paper" if args.paper_protocol else "fresh-vm", "noop": bool(args.noop),
                 "max_steps": args.max_steps, "step_unit": args.step_unit,
                 "track": args.track, "parallel": max(1, int(getattr(args, "parallel", 1) or 1)),
                 **(getattr(args, "memory_fields", None) or {})}
    t0 = time.time()
    try:
        fresh = not (args.paper_protocol and run_idx > 1)
        start_state = ""
        for attempt in (1, 2):                              # a VM can fail to come up; that is not the agent's fault
            try:
                armed = await bridge.call("/ctl/reset", {"task_path": task["path"], "run_dir": str(run_dir),
                                                         "fresh": fresh})
                start_state = str((armed or {}).get("start_state") or "") if isinstance(armed, dict) else ""
                rec.update(setup_fields(armed))
                break
            except Exception as exc:  # noqa: BLE001
                rec["reset_error"] = repr(exc)
                if attempt == 2:
                    raise
                await asyncio.sleep(10)
        if not args.noop:
            assert api is not None
            sess = await api.call("POST", "/api/sessions", json={"agent_name": AGENTS[args.track]})
            sid = sess["id"]
            rec["session_id"] = sid
            traj = out / "trajectories"
            traj.mkdir(parents=True, exist_ok=True)
            stem = f"{task['id']}.run{run_idx}"
            events = traj / f"{stem}.events.jsonl"
            events.write_text("")
            t_agent, awake0 = time.time(), awake_clock()
            rec["start_state"] = start_state
            stats = await drive(api, bridge, sid, prompt_for(task, args.max_steps, args.track, args.step_unit,
                                                             start_state),
                                t_agent + args.timeout, events, args.track, args.finish_nudges)
            rec["agent_status"] = stats.pop("status")
            rec.update(stats, agent_seconds=round(time.time() - t_agent, 1))
            slept = (time.time() - t_agent) - (awake_clock() - awake0)
            if slept > 5:
                rec["host_slept_s"] = round(slept, 1)
            try:
                (traj / f"{stem}.messages.json").write_text(
                    json.dumps(await api.call("GET", f"/api/sessions/{sid}/messages"), default=str))
            except Exception as exc:  # noqa: BLE001
                rec["messages_error"] = str(exc)
        st = await bridge.call("/ctl/state")
        ev = await bridge.call("/ctl/evaluate", {})
        rec.update(score=float(ev["score"]), eval_error=ev.get("eval_error"),
                   steps=st.get("steps", 0), obs_calls=st.get("obs_calls", 0),
                   invalid_actions=st.get("invalid_actions", 0), finished=st.get("finished"))
        if args.track == "native":
            rec["native_calls"] = st.get("native_calls", 0)
            # The harness settings this run used, so runs with different settings are never pooled by mistake.
            rec.update(observe_after=st.get("observe_after"), fail_notice_after=st.get("fail_notice_after"),
                       fail_notices=st.get("fail_notices", 0), repeat_notice_after=st.get("repeat_notice_after", 0),
                       repeat_notices=st.get("repeat_notices", 0), finish_nudge=args.finish_nudges,
                       dismiss_nags=bool(st.get("dismiss_nags")))
            # Reported by the backend for each model response and tool call (macOSWorld's 15 screenshots / 30
            # dialog turns are its 15-model-call loop): recorded so a report can show the run kept inside it.
            rec.update(model_calls=st.get("model_calls", 0), tool_messages=st.get("tool_messages", 0),
                       dialog_messages=st.get("dialog_messages", 0))
        await bridge.call("/ctl/finish", {"score": rec["score"]})
        # Reached evaluation, so the VM worked: whatever happened is the agent's outcome, not infrastructure.
        rec["tool_call_cap"] = TOOL_CALL_CAP
        capped = bool(TOOL_CALL_CAP and rec.get("tool_calls", 0) >= TOOL_CALL_CAP and not st.get("finished"))
        rec["status"] = final_status(rec.get("agent_status"), args.noop, rec.get("eval_error"), capped)
        if rec["status"] == "infra_error":
            rec["error"] = f"the grader failed, so the score is not valid: {str(rec['eval_error'])[:300]}"
        else:
            mark_host_sleep(rec)
        if args.track == "native":
            await memory_after(api, rec, st)
    except Exception as exc:  # noqa: BLE001
        rec["error"] = repr(exc)
        rec["status"] = "infra_error"
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


async def check_permissions(bridge: Bridge, tasks: list[dict], out: Path) -> list[dict]:
    """Probe the apps the tasks script; return the tasks that can run and record the rest in excluded.json."""
    apps = sorted(preflight.needed_apps(tasks))
    if not apps:
        return tasks
    print(f"checking {len(apps)} scripted app(s) in the VM (starts each once; an app waiting on a permission "
          "dialog takes about 20 s) ...", flush=True)
    results: dict = {}
    for app in apps:                                   # one short request per app: a hung dialog costs 20 s, not minutes
        results.update(await bridge.call("/ctl/preflight", {"apps": [app]}))
        print(f"  {app:<22} {results[app]['status']}", flush=True)
    print(preflight.render(results), flush=True)
    if all(r["status"] == "error" for r in results.values()):
        raise SystemExit("every probe failed, so the VM is probably not reachable (is the bridge up, and was it "
                         "started from Terminal.app?). Not skipping any task on that evidence.")
    runnable, blocked = preflight.split_tasks(tasks, results)
    if blocked:
        print(f"\n{len(blocked)} of {len(tasks)} task(s) cannot work and are skipped:", flush=True)
        for b in blocked:
            print(f"  {b['id'][:8]}  {b['domain']:<20} {b['reason']}", flush=True)
        print(preflight.FIX_HINT, flush=True)
        path = out / "excluded.json"
        old = json.loads(path.read_text()) if path.exists() else []
        path.write_text(json.dumps(old + [{**b, "by": "permission-check"} for b in blocked], indent=1))
    (out / "permission_check.json").write_text(preflight.dump(results))
    return runnable


async def main_async(args) -> None:
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    res = out / "results.jsonl"
    all_tasks = find_tasks(Path(args.macarena_dir).expanduser().resolve(), args.sources.split(","))
    tasks = select(all_tasks, args.ids, args.domain, args.sample, args.seed, args.exclude_apps, args.exclude_ids,
                   getattr(args, "per_group", None))
    k, n = parse_shard(getattr(args, "shard", None))
    if n > 1:
        whole = len(tasks)
        tasks = shard(tasks, k, n)
        print(f"shard {k}/{n}: {len(tasks)} of {whole} selected task(s)", flush=True)
    skipped = excluded(all_tasks, args.exclude_apps, args.exclude_ids)
    if skipped:
        print(f"excluded {len(skipped)} of {len(all_tasks)} task(s) (apps={args.exclude_apps or '-'}, "
              f"ids={args.exclude_ids or '-'})", flush=True)
        (out / "excluded.json").write_text(json.dumps(
            [{"id": t["id"], "domain": t["domain"], "apps": t["apps"]} for t in skipped], indent=1))
    if not tasks:
        raise SystemExit("no tasks selected")
    vms = {t["vm"] for t in tasks}
    if len(vms) > 1:
        raise SystemExit(f"the selection needs more than one VM image ({sorted(vms)}); run one source family at a time")
    vm = vms.pop()
    bridge = Bridge(args.bridge, args.control_token)
    try:
        st = await bridge.call("/ctl/state")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"cannot reach the bridge at {args.bridge}: {exc}")
    if st.get("vm") and st["vm"] != vm:
        raise SystemExit(f"the bridge runs VM '{st['vm']}' but these tasks need '{vm}'")
    if st.get("step_unit", "actions") != args.step_unit:
        raise SystemExit(f"bridge step_unit={st.get('step_unit', 'actions')} but --step-unit={args.step_unit}; "
                         "they must match")
    if st.get("max_steps") != args.max_steps:
        raise SystemExit(f"bridge max_steps={st.get('max_steps')} but --max-steps={args.max_steps}; they must match")
    bridge_track = st.get("track", "standard")
    if args.track and args.track != bridge_track:
        raise SystemExit(f"--track {args.track} but the bridge runs the {bridge_track} track (bench.sh up "
                         f"{'--native ' if args.track == 'native' else ''}...)")
    args.track = bridge_track
    if args.preflight_only or not (args.no_preflight or args.dry_run):
        tasks = await check_permissions(bridge, tasks, out)
        if args.preflight_only:
            return
        if not tasks:
            raise SystemExit("no task can run: see the permission check above")
    done = set()
    if res.exists():
        for line in res.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("track", "standard") != args.track:
                    raise SystemExit(f"{res} holds {r.get('track', 'standard')}-track results; use another --out "
                                     f"for the {args.track} track")
                if r.get("status") != "infra_error":        # infra failures are retried on resume
                    done.add((r["id"], r["run"]))
    todo = [(t, k) for t in tasks for k in range(1, args.runs + 1) if (t["id"], k) not in done]
    print(f"{len(tasks)} task(s), {len(todo)} run(s) to do ({len(done)} done), vm={vm}, track={args.track} "
          f"(agent {AGENTS[args.track]}), max_steps={args.max_steps or 'none'}, "
          f"runs/task={args.runs}, protocol={'paper' if args.paper_protocol else 'fresh-vm'}")
    if args.dry_run:
        for t, k in todo:
            print("  ", t["source"], t["domain"], t["id"], f"run{k}", "(infeasible)" if t["infeasible"] else "")
        return
    api = None if args.noop else Api(args.api, args.token)
    args.memory_fields = await memory_setup(api, out) if args.track == "native" else {}
    if res.exists() and args.memory_fields:
        for line in res.read_text().splitlines():
            r = json.loads(line) if line.strip() else {}
            mf = args.memory_fields
            if r and (r.get("memory_mode", "off"), r.get("snapshot_id"), r.get("memory_floor"),
                      bool(r.get("memory_same_task"))) != (
                    mf["memory_mode"], mf["snapshot_id"], mf.get("memory_floor"), bool(mf.get("memory_same_task"))):
                raise SystemExit(f"{res} holds results with memory {r.get('memory_mode', 'off')} "
                                 f"({r.get('snapshot_id')}, floor {r.get('memory_floor')}); this backend serves "
                                 f"{mf['memory_mode']} (floor {mf.get('memory_floor')}). Use another --out.")
    if args.memory_fields.get("memory_mode", "off") != "off":
        if args.memory_fields["memory_mode"] == "live":
            print("procedural memory: LIVE: each finished task (passed or failed) is recallable by the later tasks in "
                  f"this run, in this order: {', '.join(t['id'][:8] for t in tasks)}", flush=True)
        print(f"procedural memory: {args.memory_fields['memory_mode']} ({args.memory_fields['snapshot_size']} items, "
              f"floor {args.memory_fields.get('memory_floor')}, embedder {args.memory_fields.get('memory_embedder')})",
              flush=True)
        if args.memory_fields.get("memory_same_task"):
            print("procedural memory: SAME-TASK recall on (smoke test): these scores are not clean scores", flush=True)
    infra_streak = 0
    started_at = time.time()

    def live(current: dict | None) -> None:
        """live_status.json / .txt in the run folder: overall and per group, refreshed at every task start and end."""
        rows = report.load(res) if res.exists() else []
        report.write_live(out, tasks, args.runs, rows, current, started_at)

    live(None)
    for t, k in todo:
        print(f"start {t['id']} run{k}: {t['instruction'][:80]}", flush=True)
        live({"id": t["id"], "source": t["source"], "domain": t["domain"], "run": k,
              "instruction": t["instruction"], "started": time.strftime("%H:%M:%S")})
        rec = await run_one(api, bridge, t, k, args, out)
        with res.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        live(None)
        print(f"end   {t['id']} run{k} score={rec['score']:.2f} status={rec['status']} steps={rec['steps']} "
              f"{rec.get('error') or rec.get('eval_error') or ''}", flush=True)
        infra_streak = infra_streak + 1 if rec["status"] == "infra_error" else 0
        if args.max_infra_streak and infra_streak >= args.max_infra_streak:
            # A broken VM or bridge fails every later task the same way; stop instead of burning hours.
            # Rerun with the same --out to resume: infra_error rows are retried.
            raise SystemExit(f"stopped: {infra_streak} infra errors in a row (last: {rec.get('error')}). Fix the "
                             "VM or bridge, then rerun with the same --out to resume.")
    print(report.render(report.summarize(report.load(res))))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--macarena-dir", default=os.environ.get("MACARENA_DIR"), required=not os.environ.get("MACARENA_DIR"))
    p.add_argument("--sources", default="macarena", help=f"comma-separated: {', '.join(SOURCES)}")
    p.add_argument("--ids", help="comma-separated task ids, run in this order")
    p.add_argument("--domain", help="one domain folder, e.g. productivity")
    p.add_argument("--exclude-apps", default=os.environ.get("MACARENA_EXCLUDE_APPS"),
                   help="comma-separated app names; skip every task that lists one in related_apps, "
                        "e.g. Keynote,Numbers,Pages,LibreOffice (env: MACARENA_EXCLUDE_APPS)")
    p.add_argument("--exclude-ids", default=os.environ.get("MACARENA_EXCLUDE_IDS"),
                   help="comma-separated task ids to skip (env: MACARENA_EXCLUDE_IDS)")
    p.add_argument("--sample", type=int, help="stratified sample of N tasks")
    p.add_argument("--per-group", type=int, metavar="N",
                   help="N random tasks from each domain group (all of a smaller group); --seed picks which")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs", type=int, default=2, help="runs per task (the paper uses 2)")
    p.add_argument("--shard", help="K/N: run part K of N of the selection (every N-th task), e.g. 1/2 on one VM "
                                   "and 2/2 on the other; same selection arguments on both")
    p.add_argument("--parallel", type=int, default=1,
                   help="VMs running at once, recorded with each result (bench.sh sets it from the slots that "
                        "are up; env MACARENA_PARALLEL there)")
    p.add_argument("--max-steps", type=int, default=0,
                   help="must equal the bridge's --max-steps; 0 = no limit (the paper uses 15)")
    p.add_argument("--step-unit", choices=("actions", "turns"), default="actions",
                   help="must match the bridge: count every action, or one step per agent turn (native track)")
    p.add_argument("--timeout", type=int, default=600, help="seconds of agent time per run")
    p.add_argument("--finish-nudges", type=int, default=FINISH_NUDGES,
                   help="times to tell the agent the run is not finished when it stops without vm_done / vm_fail "
                        "(0 = never; env MACARENA_FINISH_NUDGES)")
    p.add_argument("--paper-protocol", action="store_true",
                   help="run 2 does not reset the VM, as MacArena's runner does (default: a fresh VM every run)")
    p.add_argument("--max-infra-streak", type=int, default=3,
                   help="stop after this many infra errors in a row (0 = never); rerun with the same --out to resume")
    p.add_argument("--no-preflight", action="store_true",
                   help="do not check the apps' Automation permissions first (tasks that need one score 0.0)")
    p.add_argument("--preflight-only", action="store_true", help="run the permission check, print it and stop")
    p.add_argument("--noop", action="store_true", help="no agent: reset and evaluate. Finds tasks that pass with nothing done.")
    p.add_argument("--bridge", default=os.environ.get("MACARENA_BRIDGE_URL", "http://127.0.0.1:18095"))
    p.add_argument("--control-token", default=os.environ.get("MACARENA_CONTROL_TOKEN"), required=not os.environ.get("MACARENA_CONTROL_TOKEN"))
    p.add_argument("--api", default="http://127.0.0.1:18093")
    p.add_argument("--token", default=os.environ.get("OTTO_API_TOKEN"))
    p.add_argument("--out", default=str(Path(os.environ.get("MACARENA_RUNS") or HERE.parents[1].parent / "otto-macarena-runs") / "out" / time.strftime("%Y%m%d%H%M")))
    p.add_argument("--track", choices=sorted(AGENTS), help="assert the bridge's track (default: use whatever it runs)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.sample and args.per_group:
        p.error("--sample and --per-group are two ways to pick tasks; use one")
    if args.per_group is not None and args.per_group < 1:
        p.error("--per-group must be at least 1")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
