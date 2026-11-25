"""Entry points: where the stores are, saving a run, labelling it, admitting it, recording outcomes.

Two runtimes:

* **App** (default): one store, ``<app data>/procedures.db``, read and written. On only when
  ``ProceduralMemoryConfig.enabled``.
* **Benchmark**: set by the environment of the isolated benchmark backend (``scripts/macarena/start_backend.sh``):
    ``OTTO_PROCEDURE_MODE``      ``off`` (no tool, nothing saved), ``serve`` (tool on, frozen snapshot) or ``live``
                                 (tool on, served from this run's own write store: each finished task, passed or
                                 failed, is there for the tasks after it, as in the app)
    ``OTTO_PROCEDURE_SERVE_DB``  frozen snapshot, opened read-only for the whole backend's life
    ``OTTO_PROCEDURE_WRITE_DB``  where this run's new episodes go (``run.py`` points it at ``out/<run>/memory``)
    ``OTTO_PROCEDURE_SAME_TASK`` ``1`` lets a task recall stored runs of the same task, which are normally left out
                                 (smoke tests of recall only: such a run's score is not a clean score)
  With ``serve``, episodes written during a benchmark run are never served during it, so results do not depend on
  task order and a task never sees memory written by the same run. ``live`` is the opposite on purpose (a learning
  curve, order-dependent): it leaves out only stored runs of the very same task (exact task text), not similar
  tasks, unless ``OTTO_PROCEDURE_SAME_TASK=1``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from backend.procedural_memory import compact, enrich, policy
from backend.procedural_memory.embed import Embedder, default_embedder
from backend.procedural_memory.store import ProcedureStore

logger = logging.getLogger(__name__)

ENV_MODE = "OTTO_PROCEDURE_MODE"
ENV_SERVE = "OTTO_PROCEDURE_SERVE_DB"
ENV_WRITE = "OTTO_PROCEDURE_WRITE_DB"
ENV_EMBEDDER = "OTTO_PROCEDURE_EMBEDDER"
ENV_FLOOR = "OTTO_PROCEDURE_FLOOR"
ENV_SAME_TASK = "OTTO_PROCEDURE_SAME_TASK"


@dataclass
class Runtime:
    mode: str                     # off | app | benchmark (build: offline snapshot builds)
    serve_path: Path | None
    write_path: Path | None
    floor: float
    max_tokens: int
    policy: policy.Policy
    enrich_with_llm: bool
    serve_tool: bool
    same_task: bool = False       # benchmark smoke tests only: do not leave out stored runs of the same task
    live: bool = False            # benchmark: serve the run's own write store (later tasks see earlier ones)
    max_saved_runs: int = 0       # app only: most runs kept (0 = no limit); benchmark stores are never pruned

    @property
    def benchmark(self) -> bool:
        return self.mode == "benchmark"


_runtime_override: dict[str, Any] = {}
_stores: dict[tuple[str, bool], ProcedureStore] = {}
_embedder: Embedder | None = None


def _cfg() -> Any:
    from backend.config import AppConfig
    try:
        return AppConfig.load().procedural_memory
    except Exception:  # noqa: BLE001
        from backend.config import ProceduralMemoryConfig
        return ProceduralMemoryConfig()


def runtime() -> Runtime:
    cfg = _cfg()
    pol = policy.Policy(judge_success_trusted=cfg.judge_success_trusted,
                        judge_failure_trusted=cfg.judge_failure_trusted,
                        segments_from_failed_runs=cfg.segments_from_failed_runs)
    env_mode = (os.environ.get(ENV_MODE) or "").strip().lower()
    if env_mode:
        if env_mode not in ("off", "serve", "live"):
            raise ValueError(f"{ENV_MODE} must be 'off', 'serve' or 'live', not {env_mode!r}")
        live = env_mode == "live"
        serve = None if live else (os.environ.get(ENV_SERVE) or None)
        write = _runtime_override.get("write_path") or os.environ.get(ENV_WRITE) or None
        on = env_mode in ("serve", "live")
        if env_mode == "serve" and not serve:
            raise ValueError(f"{ENV_MODE}=serve needs {ENV_SERVE} (a snapshot built by scripts/procmem)")
        floor = os.environ.get(ENV_FLOOR)
        try:
            floor_v = float(floor) if floor else cfg.similarity_floor
        except ValueError:
            raise ValueError(f"{ENV_FLOOR} must be a number, not {floor!r}") from None
        return Runtime(mode="benchmark", serve_path=Path(serve) if on and serve else None,
                       write_path=Path(write) if write else None, floor=floor_v,
                       max_tokens=cfg.max_tokens, policy=pol, enrich_with_llm=cfg.enrich_with_llm, serve_tool=on,
                       same_task=on and os.environ.get(ENV_SAME_TASK, "").strip() == "1", live=live)
    if not cfg.enabled:
        return Runtime(mode="off", serve_path=None, write_path=None, floor=cfg.similarity_floor,
                       max_tokens=cfg.max_tokens, policy=pol, enrich_with_llm=cfg.enrich_with_llm, serve_tool=False)
    from backend.config import get_app_data_dir
    path = get_app_data_dir() / "procedures.db"
    return Runtime(mode="app", serve_path=path, write_path=path, floor=cfg.similarity_floor, max_tokens=cfg.max_tokens,
                   policy=pol, enrich_with_llm=cfg.enrich_with_llm, serve_tool=cfg.serve,
                   max_saved_runs=cfg.max_saved_runs)


def set_write_path(path: str | Path) -> Path:
    """Benchmark only: where this run's episodes go (called by run.py at the start of a run)."""
    if not os.environ.get(ENV_MODE):
        raise PermissionError("the write store can only be moved in a benchmark backend")
    p = Path(path).expanduser().resolve()
    from backend.config import get_app_data_dir
    if str(p).startswith(str(get_app_data_dir().resolve())):
        raise PermissionError("a benchmark write store must not live in Otto's data folder")
    _runtime_override["write_path"] = str(p)
    return p


def store(path: Path, *, read_only: bool = False) -> ProcedureStore:
    key = (str(path), read_only)
    if key not in _stores:
        _stores[key] = ProcedureStore(path, read_only=read_only)
    return _stores[key]


def serve_store(rt: Runtime | None = None) -> ProcedureStore | None:
    rt = rt or runtime()
    if rt.live:
        return write_store(rt)            # read and written: what this run learns is served to its later tasks
    if rt.serve_path is None:
        return None
    if rt.benchmark:
        if not rt.serve_path.exists():
            raise FileNotFoundError(f"procedure snapshot not found: {rt.serve_path}")
        return store(rt.serve_path, read_only=True)
    return store(rt.serve_path)


def write_store(rt: Runtime | None = None) -> ProcedureStore | None:
    rt = rt or runtime()
    if rt.write_path is None:
        return None
    return store(rt.write_path)


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = default_embedder(os.environ.get(ENV_EMBEDDER) or _snapshot_embedder())
    return _embedder


def _snapshot_embedder() -> str | None:
    """In the benchmark, embed queries with whatever the served snapshot was built with, so a hash-built smoke
    snapshot works without also setting OTTO_PROCEDURE_EMBEDDER (an explicit setting still wins)."""
    try:
        rt = runtime()
        if not rt.benchmark or rt.serve_path is None:
            return None
        st = serve_store(rt)
        return st.get_meta().get("embedder") if st is not None else None
    except Exception:  # noqa: BLE001 — fall back to the configured model
        logger.debug("[procmem] snapshot embedder not read", exc_info=True)
        return None


def set_embedder(e: Embedder | None) -> None:
    """Tests and offline scripts."""
    global _embedder
    _embedder = e


# ── saving ──────────────────────────────────────────────────────────────────────────────────────────────────

def _session_rows(session_id: str) -> tuple[list[dict], dict]:
    from backend.config import get_app_data_dir
    d = get_app_data_dir() / "sessions"
    rows = compact.load_rows((d / f"{session_id}.messages.json").read_text())
    try:
        meta = json.loads((d / f"{session_id}.json").read_text())
    except (OSError, ValueError):
        meta = {}
    return rows, meta


def episode_data(ep: compact.Episode) -> dict[str, Any]:
    """The compacted run as stored on the episode record."""
    return {
        "steps": [s.as_dict() for s in ep.steps],
        "segments": [{"ord": s.ord, "app": s.app, "label": s.label, "evidence": s.evidence,
                      "step_from": s.step_from, "step_to": s.step_to, "recipe": s.recipe(),
                      "all_lines": s.recipe(max_lines=40), "fallback_intent": compact.segment_fallback_intent(s)}
                     for s in ep.segments],
        "facts": [{"app": f.app, "text": f.text, "source": f.source} for f in ep.facts],
        "artifacts": [{"kind": a.kind, "body": a.body, "segment_ord": a.segment_ord,
                       "writes_or_sends": a.writes_or_sends} for a in ep.artifacts],
        "actions": ep.actions, "infra_error": ep.infra_error, "self_report": ep.self_report,
    }


def save_rows(rows: list[dict], *, session_id: str, direct_agent: str | None, track: str, st: ProcedureStore,
              task_id: str | None = None, model: str | None = None) -> list[str]:
    """Stage 1: every learning-agent run in the transcript saved as a pending record (no vectors, no LLM)."""
    ids = []
    for run in compact.split_runs(rows, session_id, direct_agent=direct_agent):
        ep = compact.compact(run)
        eid, _ = st.upsert_episode(session_id=session_id, subagent_tag=run.subagent_tag, agent=run.agent,
                                   family=run.family, track=track, task_id=task_id, task_text=run.task_text,
                                   apps=ep.apps, self_report=ep.self_report, data=episode_data(ep), model=model)
        existing = {(lab["source"], lab["value"]) for lab in st.labels(eid)}
        if ep.self_report != "unknown" and ("self_report", ep.self_report) not in existing:
            st.add_label(eid, "self_report", ep.self_report)
        ids.append(eid)
    return ids


async def save_session(session_id: str, *, track: str = "app", task_id: str | None = None,
                       label: str | None = None, label_source: str | None = None,
                       confidence: float | None = None, rows: list[dict] | None = None,
                       meta: dict | None = None) -> list[str]:
    """Save a finished session's learning-agent runs; with a label, also admit them. Safe to call again."""
    rt = runtime()
    st = write_store(rt)
    if st is None:
        return []
    if rows is None:
        rows, meta = await asyncio.to_thread(_session_rows, session_id)
    meta = meta or {}
    ids = await asyncio.to_thread(save_rows, rows, session_id=session_id, direct_agent=meta.get("agent_name"),
                                  track=track, st=st, task_id=task_id, model=meta.get("model"))
    if label:
        for eid in ids:
            st.add_label(eid, label_source or "user", label, confidence)
    for eid in ids:
        await reconsider(eid, st=st, rt=rt)
    if rt.mode == "app" and rt.max_saved_runs > 0:
        dropped = await asyncio.to_thread(st.prune, rt.max_saved_runs, keep=ids)
        if dropped:
            logger.info("[procmem] over the %d-run limit; removed %d oldest run(s)", rt.max_saved_runs, len(dropped))
    return ids


async def label_session(session_id: str, label: str, source: str, *, confidence: float | None = None,
                        note: str | None = None, episode_id: str | None = None) -> list[str]:
    """Add a label (user feedback, judge, verifier) to a session's runs and reconsider them."""
    if label not in policy.LABELS or source not in policy.SOURCE_RANK:
        raise ValueError(f"bad label {label!r} or source {source!r}")
    rt = runtime()
    st = write_store(rt)
    if st is None:
        return []
    eps = [e for e in st.episodes_for_session(session_id) if episode_id in (None, e["id"])]
    for e in eps:
        st.add_label(e["id"], source, label, confidence, note)
        await reconsider(e["id"], st=st, rt=rt)
    return [e["id"] for e in eps]


def _user_label(st: ProcedureStore, episode_id: str) -> str | None:
    """The latest thumbs up / down the user gave this run (``success`` / ``failure``), or None."""
    vals = [lab["value"] for lab in st.labels(episode_id) if lab["source"] == "user"]
    return vals[-1] if vals else None


def session_feedback_state(session_id: str) -> dict[str, Any]:
    """What the chat window needs to ask "did it work?" after a macOS desktop agent run.

    ``ask`` is true when procedural memory is on in the app and this session has desktop-agent runs that were
    saved but not yet reviewed. ``answer`` is the user's last answer for this session's desktop runs, if any."""
    rt = runtime()
    out: dict[str, Any] = {"enabled": rt.mode == "app", "ask": False, "answer": None, "episodes": []}
    if rt.mode != "app":
        return out
    st = write_store(rt)
    if st is None:
        return out
    eps = [e for e in st.episodes_for_session(session_id) if e.get("agent") in compact.DESKTOP_AGENTS]
    rows = []
    for e in sorted(eps, key=lambda e: e.get("created_at") or 0):
        ans = _user_label(st, e["id"])
        rows.append({"id": e["id"], "task_text": e.get("task_text") or "", "apps": e.get("apps") or [],
                     "status": e.get("status"), "user_label": ans})
        if ans:
            out["answer"] = ans
    out["episodes"] = rows
    out["ask"] = any(r["user_label"] is None and r["status"] == "pending" for r in rows)
    return out


async def apply_user_answer(episode_id: str, worked: bool, *, st: ProcedureStore, rt: Runtime) -> str:
    """The user's answer for one run, the same from the chat card and the Procedures page; returns its new status.

    Worked: a user success label, which admits the run (an excluded run comes back too).
    Did not work: recorded, and the run is excluded with everything learned from it, so nothing of it is used."""
    if worked:
        st.add_label(episode_id, "user", "success")
        await reconsider(episode_id, st=st, rt=rt)
    else:
        st.add_label(episode_id, "user", "failure")
        st.drop_items(episode_id)
        st.set_status(episode_id, "excluded", label="failure", label_source="user")
    return (st.episode(episode_id) or {}).get("status") or ""


async def record_session_feedback(session_id: str, worked: bool) -> dict[str, Any]:
    """Thumbs up / down from the chat window for this session's unreviewed desktop-agent runs.

    Worked: label the runs a user success, which admits them (what the agent learned becomes recallable).
    Did not work: record it and exclude the runs, so nothing from them is ever used."""
    rt = runtime()
    if rt.mode != "app":
        raise ValueError("procedural memory is off")
    st = write_store(rt)
    if st is None:
        raise ValueError("procedural memory is off")
    todo = [r["id"] for r in session_feedback_state(session_id)["episodes"]
            if r["user_label"] is None and r["status"] == "pending"]
    for eid in todo:
        await apply_user_answer(eid, worked, st=st, rt=rt)
    statuses = [(st.episode(eid) or {}).get("status") for eid in todo]
    return {"updated": todo, "saved": sum(1 for s in statuses if s == "admitted"),
            "still_pending": sum(1 for s in statuses if s == "pending")}


# ── admission ───────────────────────────────────────────────────────────────────────────────────────────────

async def reconsider(episode_id: str, *, st: ProcedureStore, rt: Runtime | None = None,
                     llm: enrich.LLMCall | None = None, use_llm: bool | None = None) -> policy.Decision:
    """Apply the admission policy to an episode's current labels: write and embed its pieces, or drop them."""
    rt = rt or runtime()
    ep = st.episode(episode_id)
    if ep is None:
        raise KeyError(episode_id)
    data = ep["data"]
    decision = policy.decide(st.labels(episode_id), actions=int(data.get("actions") or 0),
                             infra_error=bool(data.get("infra_error")), policy=rt.policy)
    if decision.status != "admitted":
        st.drop_items(episode_id)
        st.set_status(episode_id, decision.status, label=decision.label, label_source=decision.source,
                      confidence=decision.confidence)
        return decision
    if llm is None and (rt.enrich_with_llm if use_llm is None else use_llm):
        llm = await enrich.default_llm()
    info = await enrich.enrich({**data, "task_text": ep["task_text"], "apps": ep["apps"]}, decision.label,
                               decision.source or "", llm)
    items, artifacts = build_items(ep, decision, info, rt.policy)
    if st.get_meta().get("embedder"):
        stamp_embedder(st)                 # refuse a store built with another model before doing any work
    try:
        vecs = (await get_embedder().embed([it["intent"] + " | " + it.get("embed_extra", "") for it in items])
                if items else [])
    except Exception as exc:  # noqa: BLE001 — e.g. sentence-transformers not installed
        # Keep the run and its labels; only the searchable pieces wait. A snapshot build (scripts/procmem) or a
        # later relabel with a working embedder admits it; nothing is lost.
        _warn_no_embedder(exc)
        st.set_status(episode_id, "pending", label=decision.label, label_source=decision.source,
                      confidence=decision.confidence, intent=info["intent"])
        return replace(decision, status="pending", reason=f"not embedded ({type(exc).__name__}): {exc}"[:300])
    stamp_embedder(st)                     # only once vectors exist, so a store never names a model it lacks
    st.replace_items(episode_id, items, vecs, artifacts)
    st.set_status(episode_id, "admitted", label=decision.label, label_source=decision.source,
                  confidence=decision.confidence, intent=info["intent"])
    return decision


def build_items(ep: dict[str, Any], decision: policy.Decision, info: dict[str, Any],
                pol: policy.Policy) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eid, fam, data = ep["id"], ep["family"], ep["data"]
    primary = (ep.get("apps") or [None])[0]
    stripped_task = compact.strip_values(ep.get("task_text") or "")
    segments = data.get("segments") or []
    seg_intents = info.get("segments") or {}
    items: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    chosen = policy.admitted_segments(segments, decision.label, pol)
    seg_ids = {s["ord"]: f"{eid}:s{s['ord']}" for s in chosen}
    arts = data.get("artifacts") or []

    def seg_intent(s: dict) -> str:
        return seg_intents.get(s["ord"]) or seg_intents.get(str(s["ord"])) or s.get("fallback_intent") or ""

    if decision.label == "success":
        lines = [ln for s in segments for ln in s.get("recipe") or []]
        item_id = f"{eid}:r0"
        items.append({"id": item_id, "kind": "recipe", "ord": 0, "app": primary, "family": fam,
                      "intent": info["intent"], "label_source": decision.source, "from_outcome": "success",
                      "body": {"lines": lines[:12], "outline": [seg_intent(s) for s in segments][:6],
                               "has_script": bool(arts), "next": None},
                      "search_text": " ".join([stripped_task, *lines]), "embed_extra": stripped_task[:300]})
        for a in arts:
            artifacts.append({"id": f"{item_id}:a{len(artifacts)}", "item_id": item_id, **_art(a)})
    ordered = sorted(chosen, key=lambda s: s["ord"])
    for k, s in enumerate(ordered):
        item_id = seg_ids[s["ord"]]
        nxt = ordered[k + 1]["ord"] if k + 1 < len(ordered) and ordered[k + 1]["ord"] == s["ord"] + 1 else None
        seg_arts = [a for a in arts if a.get("segment_ord") == s["ord"]]
        source = decision.source if s["seg_source"] == "run" else "segment_evidence"
        items.append({"id": item_id, "kind": "recipe", "ord": s["ord"], "app": s.get("app") or primary, "family": fam,
                      "intent": seg_intent(s), "label_source": source, "from_outcome": decision.label,
                      "body": {"lines": s.get("recipe") or [], "evidence": s.get("evidence", ""),
                               "has_script": bool(seg_arts), "next": seg_ids.get(nxt) if nxt else None},
                      "search_text": " ".join(s.get("recipe") or [])})
        for a in seg_arts:
            artifacts.append({"id": f"{item_id}:a{len(artifacts)}", "item_id": item_id, **_art(a)})
    for n, f in enumerate(data.get("facts") or []):
        items.append({"id": f"{eid}:f{n}", "kind": "fact", "ord": n, "app": f.get("app") or primary, "family": fam,
                      "intent": f"{f.get('app') or primary}: {f['text']}", "label_source": f.get("source"),
                      "from_outcome": decision.label, "body": {"text": f["text"]}, "search_text": f["text"]})
    for n, les in enumerate(info.get("lessons") or []):
        text = f"{les['title']}. {les.get('sentence', '')} {les.get('content', '')}"
        items.append({"id": f"{eid}:l{n}", "kind": "lesson", "ord": n, "app": primary, "family": fam,
                      "intent": f"{primary}: {les['title']}", "label_source": decision.source,
                      "from_outcome": decision.label, "body": les, "search_text": text, "embed_extra": text[:300]})
    return items, artifacts


def _art(a: dict[str, Any]) -> dict[str, Any]:
    return {"kind": a["kind"], "body": a["body"], "writes_or_sends": bool(a.get("writes_or_sends"))}


# ── outcomes and status ─────────────────────────────────────────────────────────────────────────────────────

_warned_embedder = False


def _warn_no_embedder(exc: Exception) -> None:
    global _warned_embedder
    if not _warned_embedder:
        _warned_embedder = True
        logger.warning("[procmem] embedding model unavailable (%s); runs are saved with their labels but not "
                       "searchable until embedded", exc)


def stamp_embedder(st: ProcedureStore) -> None:
    """Record which embedder a store's vectors came from; retrieval refuses a store built with another one."""
    meta = st.get_meta()
    name = get_embedder().name
    if not meta.get("embedder"):
        st.set_meta(embedder=name)
    elif meta["embedder"] != name:
        raise RuntimeError(f"store {st.path} was embedded with {meta['embedder']}, not {name}; rebuild it")


def record_outcome(session_id: str, *, score: float | None, outcome: str) -> int:
    """How a run that used memory ended, written to the items it used (lifecycle: retire what keeps failing)."""
    st = write_store()
    if st is None:
        return 0
    return st.record_outcome(session_id, score=score, outcome=outcome)


def status() -> dict[str, Any]:
    rt = runtime()
    out: dict[str, Any] = {"mode": rt.mode, "serve_tool": rt.serve_tool, "floor": rt.floor,
                           "max_tokens": rt.max_tokens, "same_task": rt.same_task, "live": rt.live, "embedder": get_embedder().name if rt.serve_tool else None,
                           "serve_path": str(rt.serve_path) if rt.serve_path else None,
                           "write_path": str(rt.write_path) if rt.write_path else None}
    try:
        s = serve_store(rt)
    except FileNotFoundError as exc:
        out["error"] = str(exc)
        s = None
    if s is not None:
        meta = s.get_meta()
        counts = s.counts()
        out.update(snapshot_id=meta.get("snapshot_id"), snapshot_meta=meta, serve_counts=counts,
                   snapshot_size=sum(v for k, v in counts.items() if k.startswith("items_")))
    return out
