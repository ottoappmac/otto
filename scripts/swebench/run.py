#!/usr/bin/env python
"""Run Otto on SWE-bench instances through its local backend API.

Usage (backend must already be running, see start_backend.sh):

    uv run --with-requirements scripts/swebench/requirements.txt \
        python scripts/swebench/run.py --dataset verified --sample 5 --seed 0

Writes <out>/preds.jsonl (one line per instance, resumable) and
<out>/trajectories/<instance_id>.json.
"""

from __future__ import annotations

import argparse
import os
import asyncio
import json
import random
import shutil
import sys
import time
from pathlib import Path

import httpx
import websockets

sys.path.insert(0, str(Path(__file__).parent))
import data  # noqa: E402
import docker_env  # noqa: E402
import env as envmod  # noqa: E402

RUNS = Path(os.environ.get("SWEBENCH_RUNS")
            or Path(__file__).resolve().parents[2].parent / "otto-swebench-runs")

DATASETS = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
}
AGENT = "swebench-coding-agent"
MAX_RESUMES = 3
NO_USER = (
    "No user is available. Make reasonable assumptions and finish the task; "
    "do not ask questions."
)


def load_instances(args) -> list[dict]:
    rows = data.load_rows(args.dataset)
    if args.ids:
        wanted = set(args.ids.split(","))
        rows = [r for r in rows if r["instance_id"] in wanted]
    elif args.repo:
        rows = [r for r in rows if args.repo in r["repo"]]
    if args.sample and not args.ids:
        # Stratify across repos, deterministic for a seed.
        rng = random.Random(args.seed)
        by_repo: dict[str, list[dict]] = {}
        for r in sorted(rows, key=lambda r: r["instance_id"]):
            by_repo.setdefault(r["repo"], []).append(r)
        for lst in by_repo.values():
            rng.shuffle(lst)
        picked: list[dict] = []
        while len(picked) < args.sample and any(by_repo.values()):
            for repo in sorted(by_repo):
                if by_repo[repo] and len(picked) < args.sample:
                    picked.append(by_repo[repo].pop())
        rows = picked
    elif args.limit:
        rows = rows[: args.limit]
    return rows


def build_task(instance: dict, prepared: envmod.Prepared) -> str:
    if prepared.mode == "docker":
        env_text = docker_env.task_text(prepared.test_cmd, prepared.repo_dir.name)
    else:
        env_text = (
            f"Python environment: a virtualenv with the project installed is at `.venv` in the "
            f"repository root. Run tools through it, e.g. "
            f"`export PATH=$PWD/.venv/bin:$PATH` first. Project test command: "
            f"`{prepared.test_cmd}` (append the specific test files or names you want).\n\n"
        )
    return (
        f"Repository: {instance['repo']} (mapped folder, a git repo with the code at the "
        f"commit the issue was filed against).\n\n"
        f"{env_text}"
        f"Resolve this issue by editing the source code.\n\n"
        f"<issue>\n{instance['problem_statement']}\n</issue>"
    )


class Api:
    def __init__(self, base: str, token: str | None):
        self.base = base.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.http = httpx.AsyncClient(base_url=self.base, headers=self.headers, timeout=60)

    async def post(self, path: str, **kw):
        r = await self.http.post(path, **kw)
        r.raise_for_status()
        return r.json()

    async def patch(self, path: str, **kw):
        r = await self.http.patch(path, **kw)
        r.raise_for_status()
        return r.json()

    async def get(self, path: str):
        r = await self.http.get(path)
        r.raise_for_status()
        return r.json()

    def ws_url(self, session_id: str) -> str:
        scheme = "wss" if self.base.startswith("https") else "ws"
        url = f"{scheme}://{self.base.split('://', 1)[1]}/ws/chat/{session_id}"
        token = self.headers.get("Authorization", "")[7:]
        return f"{url}?token={token}" if token else url


def log_event(path: Path | None, msg: dict) -> None:
    """Append one streamed frame, with the wall-clock time it arrived, to the instance's
    event log. Written as it happens, so a killed or timed-out run still leaves a record."""
    if path is None:
        return
    with path.open("a") as f:
        f.write(json.dumps({"ts": round(time.time(), 3), **msg}, default=str) + "\n")


async def drive(api: Api, session_id: str, task: str, deadline: float,
                events_path: Path | None = None) -> dict:
    """Send the task and answer approvals/questions until the run ends."""
    stats = {"status": "timeout", "error": None, "approvals": 0, "questions": 0}
    async with websockets.connect(api.ws_url(session_id), max_size=None) as ws:
        await ws.send(json.dumps({"content": task, "depth": "deep", "mode": "agent"}))
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                await api.post(f"/api/sessions/{session_id}/stop")
                return stats
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                await api.post(f"/api/sessions/{session_id}/stop")
                return stats
            msg = json.loads(raw)
            log_event(events_path, msg)
            kind = msg.get("type")
            meta = msg.get("metadata") or {}
            if kind == "done":
                stats["status"] = "done"
                return stats
            if kind == "stopped":
                stats["status"] = "stopped"
                return stats
            if kind == "error":
                stats["errors"] = stats.get("errors", 0) + 1
                stats["error"] = msg.get("content")
                # Local model servers drop connections; let the agent pick up where it was.
                if stats["errors"] <= MAX_RESUMES and time.time() < deadline - 30:
                    await asyncio.sleep(5)
                    await ws.send(json.dumps({
                        "content": "The connection was interrupted. Continue from where you "
                                   "left off and finish the task.",
                        "depth": "deep", "mode": "agent",
                    }))
                    continue
                stats["status"] = "error"
                return stats
            if kind == "ask_user":
                stats["questions"] += 1
                await ws.send(json.dumps({
                    "type": "hitl_response", "decisions": [{"answer": NO_USER}],
                    "request_id": meta.get("request_id"),
                }))
            elif kind == "hitl_request":
                stats["approvals"] += 1
                count = len(meta.get("action_requests") or []) or 1
                if meta.get("type") == "request_credential":
                    decisions = [{"answer": "cancelled"}] * count
                elif (meta.get("risk") or {}).get("category") == "persistence":
                    decisions = [{"type": "reject", "message": "not allowed"}] * count
                else:
                    decisions = [{"type": "approve"}] * count
                await ws.send(json.dumps({
                    "type": "hitl_response", "decisions": decisions,
                    "request_id": meta.get("request_id"),
                }))


async def run_one(api: Api, instance: dict, args, work: Path, out: Path,
                  run_sem: asyncio.Semaphore) -> dict:
    iid = instance["instance_id"]
    t0 = time.time()
    rec = {"instance_id": iid, "model_name_or_path": args.model_name, "model_patch": ""}
    meta = {"instance_id": iid}
    prepared = None
    try:
        # Checkout and venv are built ahead of the agent (bounded by the caller),
        # so the model is not idle while the next instance installs.
        prepared = await asyncio.to_thread(envmod.prepare, instance, work, args.env_mode)
        meta["env_notes"] = prepared.notes
        meta["prep_seconds"] = round(time.time() - t0, 1)
        meta["env_mode"] = prepared.mode
        async with run_sem, docker_env.broker(prepared):
            t_agent = time.time()
            grant = await api.post("/api/workspace-grants",
                                   json={"path": str(prepared.repo_dir), "mode": "write"})
            await api.patch(f"/api/workspace-grants/{grant['id']}", json={"network": False})
            session = await api.post("/api/sessions", json={"agent_name": AGENT})
            sid = session["id"]
            meta["session_id"] = sid
            await api.post(f"/api/sessions/{sid}/workspace", json={"grant_id": grant["id"]})
            (out / "trajectories").mkdir(parents=True, exist_ok=True)
            events_path = out / "trajectories" / f"{iid}.events.jsonl"
            events_path.write_text("")
            log_event(events_path, {"type": "harness", "event": "agent_start", "session_id": sid,
                                    "env_mode": prepared.mode})
            meta["events_file"] = events_path.name
            stats = await drive(api, sid, build_task(instance, prepared), t_agent + args.timeout,
                                events_path)
            log_event(events_path, {"type": "harness", "event": "agent_end", **stats})
            meta.update(stats)
            meta["agent_seconds"] = round(time.time() - t_agent, 1)
            try:
                meta["messages"] = await api.get(f"/api/sessions/{sid}/messages")
            except Exception as exc:  # noqa: BLE001
                meta["messages_error"] = str(exc)
            rec["model_patch"] = await asyncio.to_thread(
                envmod.extract_patch, prepared.repo_dir, prepared.base_sha)
            try:
                await api.http.request("DELETE", f"/api/workspace-grants/{grant['id']}")
            except Exception:  # noqa: BLE001
                pass
    except envmod.EnvUnavailable as exc:
        meta["status"], meta["error"] = "skipped_env", str(exc)
    except Exception as exc:  # noqa: BLE001
        meta["status"], meta["error"] = "harness_error", repr(exc)
    finally:
        if prepared is not None:
            if prepared.mode == "docker":
                await asyncio.to_thread(docker_env.cleanup, prepared, args.keep_images)
            elif not args.keep_envs:
                shutil.rmtree(prepared.repo_dir / ".venv", ignore_errors=True)
    meta["seconds"] = round(time.time() - t0, 1)
    meta["patch_chars"] = len(rec["model_patch"])
    (out / "trajectories").mkdir(parents=True, exist_ok=True)
    (out / "trajectories" / f"{iid}.json").write_text(json.dumps(meta, indent=1, default=str))
    return rec


async def main_async(args) -> None:
    docker_env.configure(args.image_namespace, not args.no_pull)
    out = Path(args.out).expanduser()
    work = Path(args.work).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    preds_path = out / "preds.jsonl"
    if args.retry_failed and preds_path.exists():
        dropped = envmod.drop_failed_preds(out)
        print(f"retrying {len(dropped)} skipped or crashed instance(s): {', '.join(dropped) or '-'}")
    done = set()
    if preds_path.exists():
        done = {json.loads(l)["instance_id"] for l in preds_path.read_text().splitlines() if l}
    instances = [i for i in load_instances(args) if i["instance_id"] not in done]
    print(f"{len(instances)} instances to run ({len(done)} already done)")
    if args.dry_run:
        need = [i for i in instances
                if envmod.wants_docker(args.env_mode, i["repo"], envmod._spec(i).get("python", "3.9"))]
        missing = [i["instance_id"] for i in need
                   if not docker_env.image_present(docker_env.image_name(i["instance_id"]))]
        print(f"{len(need)} of them use a Docker environment ({args.image_namespace} images); "
              f"{len(missing)} of those have no local image")
        for iid in missing[:20]:
            print("  missing:", iid)
        sys.exit(3 if (missing and args.no_pull) else 0)
    api = Api(args.api, args.token)
    run_sem = asyncio.Semaphore(args.workers)          # agents running at once
    slots = asyncio.Semaphore(args.workers + args.prefetch)  # prepared-but-not-finished instances
    lock = asyncio.Lock()

    async def worker(inst):
        async with slots:
            print(f"start {inst['instance_id']}", flush=True)
            rec = await run_one(api, inst, args, work, out, run_sem)
            async with lock:
                with preds_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            print(f"end   {inst['instance_id']} patch={len(rec['model_patch'])}ch", flush=True)

    await asyncio.gather(*(worker(i) for i in instances))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="verified", help="verified | lite | full | HF name")
    p.add_argument("--ids", help="comma-separated instance ids")
    p.add_argument("--repo", help="substring filter on repo, e.g. django")
    p.add_argument("--sample", type=int, help="repo-stratified sample size")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=1, help="agents running at once")
    p.add_argument("--prefetch", type=int, default=2,
                   help="extra instances to build checkouts/venvs for while agents run")
    p.add_argument("--env-mode", choices=["auto", "uv", "docker"], default="auto",
                   help="where the agent's Python environment lives: auto = a host uv virtualenv, "
                        "or Docker when uv has no build of the needed Python (3.5-3.7)")
    p.add_argument("--retry-failed", action="store_true",
                   help="re-run instances whose earlier attempt was skipped or crashed "
                        "(not ones the agent simply failed to solve)")
    p.add_argument("--keep-images", action="store_true", help="keep Docker images used for agent environments")
    p.add_argument("--image-namespace", default="swebench",
                   help="image set for Docker environments: swebench (official) or logicstar (compressed bundle)")
    p.add_argument("--no-pull", action="store_true",
                   help="never pull images; an instance whose image is not already local is skipped")
    p.add_argument("--dry-run", action="store_true",
                   help="print the selected instances and whether their images are local, then stop")
    p.add_argument("--keep-envs", action="store_true", help="keep each instance's .venv after the run")
    p.add_argument("--timeout", type=int, default=1800, help="seconds per instance")
    p.add_argument("--api", default="http://127.0.0.1:18091")
    p.add_argument("--token", default=None)
    p.add_argument("--model-name", default="otto")
    p.add_argument("--out", default=str(RUNS / "out" / "default"))
    p.add_argument("--work", default=str(RUNS / "work"))
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
