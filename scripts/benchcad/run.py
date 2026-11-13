#!/usr/bin/env python
"""Run Otto's benchcad-agent on BenchCAD 2.x cases through its local backend, then score them.

Dev harness: it uses BenchCAD's own staging, prompt and scorer (cases.py, run in BenchCAD's
environment) but drives Otto through its API, so it works with any model Otto supports,
including local ones. Scores from here are NOT reportable; the reportable path is BenchCAD's
Customized-harness track (own_agent/, Harbor, metered API). See README.md.

    python scripts/benchcad/run.py --bench ../benchcad-2-main --cases examples --workers 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

import httpx
import websockets

HERE = Path(__file__).resolve().parent
AGENT = "benchcad-agent"
MAX_RESUMES = 3
NO_USER = "No user is available. Make reasonable assumptions and finish; do not ask questions."


# ── Otto API (same protocol as scripts/swebench/run.py) ─────────────────────

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


def log_event(path: Path, msg: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps({"ts": round(time.time(), 3), **msg}, default=str) + "\n")


async def drive(api: Api, sid: str, task: str, deadline: float, events: Path) -> dict:
    stats = {"status": "timeout", "error": None, "approvals": 0, "questions": 0}
    async with websockets.connect(api.ws_url(sid), max_size=None) as ws:
        await ws.send(json.dumps({"content": task, "depth": "deep", "mode": "agent"}))
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                await api.call("POST", f"/api/sessions/{sid}/stop")
                return stats
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=remaining))
            except asyncio.TimeoutError:
                await api.call("POST", f"/api/sessions/{sid}/stop")
                return stats
            log_event(events, msg)
            kind, meta = msg.get("type"), msg.get("metadata") or {}
            if kind in ("done", "stopped"):
                stats["status"] = kind
                return stats
            if kind == "error":
                stats["errors"] = stats.get("errors", 0) + 1
                stats["error"] = msg.get("content")
                if stats["errors"] <= MAX_RESUMES and time.time() < deadline - 30:
                    await asyncio.sleep(5)
                    await ws.send(json.dumps({"content": "The connection was interrupted. Continue "
                                              "from where you left off and finish.",
                                              "depth": "deep", "mode": "agent"}))
                    continue
                stats["status"] = "error"
                return stats
            if kind == "ask_user":
                stats["questions"] += 1
                await ws.send(json.dumps({"type": "hitl_response", "decisions": [{"answer": NO_USER}],
                                          "request_id": meta.get("request_id")}))
            elif kind == "hitl_request":
                stats["approvals"] += 1
                n = len(meta.get("action_requests") or []) or 1
                if meta.get("type") == "request_credential":
                    dec = [{"answer": "cancelled"}] * n
                elif (meta.get("risk") or {}).get("category") == "persistence":
                    dec = [{"type": "reject", "message": "not allowed"}] * n
                else:
                    dec = [{"type": "approve"}] * n
                await ws.send(json.dumps({"type": "hitl_response", "decisions": dec,
                                          "request_id": meta.get("request_id")}))


# ── BenchCAD side, run in BenchCAD's own environment ────────────────────────

async def bench_py(args, *cmd: str, timeout: float = 1800) -> dict | list:
    proc = await asyncio.create_subprocess_exec(
        "uv", "run", "--project", args.bench, "python", str(HERE / "cases.py"), *cmd,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError(f"cases.py {cmd[0]} timed out")
    lines = [l for l in out.decode().splitlines() if l.strip()]
    if proc.returncode != 0 or not lines:
        raise RuntimeError(f"cases.py {cmd[0]} failed: {(err or out).decode()[-800:]}")
    return json.loads(lines[-1])


# ── one case ────────────────────────────────────────────────────────────────

async def run_one(api: Api, case: dict, args, out: Path, sem: asyncio.Semaphore) -> dict:
    cid = case["id"].replace("/", "__")
    work = out / "work" / cid
    traj = out / "trajectories"
    traj.mkdir(parents=True, exist_ok=True)
    rec: dict = {"id": case["id"], "task": case["task"], "score": 0.0, "status": "harness_error",
                 "subagents": "off" if os.environ.get("BENCHCAD_NO_SUBAGENTS", "") in ("1", "true", "yes") else "on"}
    t0 = time.time()
    try:
        await bench_py(args, "stage", args.bench, case["case"], str(work))
        task = (work / "TASK.md").read_text()
        async with sem:
            t_agent = time.time()
            grant = await api.call("POST", "/api/workspace-grants", json={"path": str(work), "mode": "write"})
            await api.call("PATCH", f"/api/workspace-grants/{grant['id']}", json={"network": False})
            sess = await api.call("POST", "/api/sessions", json={"agent_name": AGENT})
            sid = sess["id"]
            await api.call("POST", f"/api/sessions/{sid}/workspace", json={"grant_id": grant["id"]})
            events = traj / f"{cid}.events.jsonl"
            events.write_text("")
            stats = await drive(api, sid, task, t_agent + args.timeout, events)
            rec.update(stats, session_id=sid, agent_seconds=round(time.time() - t_agent, 1))
            try:
                (traj / f"{cid}.messages.json").write_text(
                    json.dumps(await api.call("GET", f"/api/sessions/{sid}/messages"), default=str))
            except Exception as exc:  # noqa: BLE001
                rec["messages_error"] = str(exc)
            try:
                await api.call("DELETE", f"/api/workspace-grants/{grant['id']}")
            except Exception:  # noqa: BLE001
                pass
        if args.no_score:
            rec["status"] = rec.get("status", "unscored")
        else:
            sc = await bench_py(args, "score", args.bench, case["case"], str(work),
                                str(out / "trajectories" / f"{cid}.score.json"), timeout=1500)
            rec["score"] = float(sc.get("score") or 0.0)
            rec["score_error"] = sc.get("error")
    except Exception as exc:  # noqa: BLE001
        rec["error"] = repr(exc)
        rec["status"] = "harness_error"
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def summarize(path: Path) -> str:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l]
    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(r["task"] or "?", []).append(r["score"])
    lines = [f"{t}: mean {sum(v)/len(v):.3f} over {len(v)} case(s)" for t, v in sorted(by.items())]
    means = [sum(v) / len(v) for v in by.values()]
    harness_err = sum(1 for r in rows if r.get("status") == "harness_error")
    lines.append(f"mean of task means: {sum(means)/len(means):.3f}   cases: {len(rows)}   harness errors: {harness_err}")
    lines.append("DEV RUN: not comparable with reported BenchCAD scores.")
    return "\n".join(lines)


async def main_async(args) -> None:
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    res = out / "results.jsonl"
    cases = await bench_py(args, "list", args.bench, str(Path(args.cases).expanduser().resolve()))
    if args.ids:
        want = set(args.ids.split(","))
        cases = [c for c in cases if c["id"] in want or c["task"] in want or Path(c["case"]).name in want]
    done = {json.loads(l)["id"] for l in res.read_text().splitlines() if l} if res.exists() else set()
    todo = [c for c in cases if c["id"] not in done]
    print(f"{len(todo)} case(s) to run ({len(done)} already done)")
    if args.dry_run:
        for c in todo:
            print("  ", c["id"])
        return
    api = Api(args.api, args.token)
    sem, lock = asyncio.Semaphore(args.workers), asyncio.Lock()

    async def worker(c):
        print("start", c["id"], flush=True)
        rec = await run_one(api, c, args, out, sem)
        async with lock:
            with res.open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        print(f"end   {c['id']} score={rec['score']:.3f} status={rec.get('status')} {rec.get('error') or rec.get('score_error') or ''}", flush=True)

    await asyncio.gather(*(worker(c) for c in todo))
    if res.exists():
        text = summarize(res)
        (out / "summary.txt").write_text(text + "\n")
        print(text)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", default=os.environ.get("BENCHCAD_DIR", str(HERE.parents[2].parent / "benchcad-2-main")),
                   help="BenchCAD checkout (use a release tag for anything you compare)")
    p.add_argument("--cases", default=None, help="folder of cases (default: <bench>/examples)")
    p.add_argument("--ids", help="comma-separated: task1/case1, task3, or a case name")
    p.add_argument("--out", default=str(Path(os.environ.get("BENCHCAD_RUNS") or HERE.parents[1].parent / "otto-benchcad-runs") / "out" / time.strftime("%Y%m%d%H%M")))
    p.add_argument("--api", default="http://127.0.0.1:18092")
    p.add_argument("--token", default=os.environ.get("OTTO_API_TOKEN"))
    p.add_argument("--workers", type=int, default=1, help="agents at once (each holds a Docker sandbox)")
    p.add_argument("--timeout", type=int, default=3600, help="seconds per case for the agent")
    p.add_argument("--no-score", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.cases = args.cases or str(Path(args.bench) / "examples")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
