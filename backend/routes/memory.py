"""API routes for the Memory (memory consolidation) feature."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from backend.config import AppConfig, get_app_data_dir

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory", tags=["memory"])

_CORRECTIONS_HEADING = "## Corrections"


# ---------------------------------------------------------------------------
# Topic helper utilities
# ---------------------------------------------------------------------------


def _memory_dir() -> Path:
    d = get_app_data_dir() / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _topic_path(filename: str) -> Path | None:
    """Return a validated path for a topic file, or None if invalid."""
    if not filename.endswith(".md"):
        filename += ".md"
    filename = Path(filename).name
    if not filename or filename == "MEMORY.md":
        return None
    target = _memory_dir() / filename
    if not target.resolve().is_relative_to(_memory_dir().resolve()):
        return None
    return target


def _parse_topic(path: Path) -> dict[str, Any]:
    """Parse a topic file and return its metadata + content."""
    text = path.read_text(encoding="utf-8")
    fm: dict[str, Any] = {}

    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            fm_block = text[3:end].strip()
            for line in fm_block.splitlines():
                m = re.match(r'^(\w+)\s*:\s*(.*)$', line.strip())
                if not m:
                    continue
                key, val = m.group(1), m.group(2).strip()
                if val.startswith("[") and val.endswith("]"):
                    items = re.findall(r'"([^"]+)"|\'([^\']+)\'|([\w\-]+)', val[1:-1])
                    fm[key] = [a or b or c for a, b, c in items]
                else:
                    fm[key] = val.strip('"\'')

    return {
        "filename": path.name,
        "name": fm.get("name", path.stem),
        "description": fm.get("description", ""),
        "type": fm.get("type", ""),
        "confidence": fm.get("confidence", ""),
        "created_at": fm.get("created_at"),
        "updated_at": fm.get("updated_at"),
        "source_sessions": fm.get("source_sessions") or [],
        "size_bytes": path.stat().st_size,
        "content": text,
    }


@router.get("/status")
async def get_memory_status() -> dict[str, Any]:
    """Return current memory consolidation run status for UI polling."""
    from backend.memory import get_status

    status = get_status()
    return status.to_dict()


@router.post("/run")
async def trigger_consolidation() -> JSONResponse:
    """Manually trigger a memory consolidation run."""
    from backend.consolidation_lock import (
        last_consolidated_at, try_acquire,
    )
    from backend.memory import (
        RunState, execute_consolidation, get_status,
    )
    from backend.session_transcript import (
        list_transcripts_since,
    )

    status = get_status()
    if status.state == RunState.RUNNING:
        return JSONResponse(
            status_code=409,
            content={"error": "Memory consolidation already in progress"},
        )

    cfg = (await AppConfig.aload()).memory

    since = last_consolidated_at()
    candidates = list_transcripts_since(since)
    if not candidates:
        return JSONResponse(
            status_code=200,
            content={
            "status": "skipped",
            "reason": "No transcripts to process",
        },
        )

    prev_mtime = try_acquire()
    if prev_mtime is None:
        return JSONResponse(
            status_code=409,
            content={"error": "Could not acquire consolidation lock"},
        )

    asyncio.create_task(execute_consolidation(prev_mtime, candidates, cfg))
    return JSONResponse(
        status_code=202,
        content={"status": "started", "transcripts": len(candidates)},
    )


@router.post("/cancel")
async def cancel_consolidation() -> dict[str, str]:
    """Request cancellation of a running memory consolidation."""
    from backend.memory import (
        RunState, get_status, request_cancel,
    )

    status = get_status()
    if status.state != RunState.RUNNING:
        return {"status": "not_running"}

    request_cancel()
    return {"status": "cancel_requested"}


@router.get("/hits")
async def get_memory_hits() -> dict[str, Any]:
    """Return aggregated memory injection hit stats."""
    import json as _json
    from collections import Counter

    from backend.config import get_app_data_dir

    hits_path = get_app_data_dir() / "memory" / "memory-hits.jsonl"
    if not hits_path.exists():
        return {
            "total_injections": 0,
            "unique_sessions": 0,
            "cache_hit_rate": 0,
            "top_topics": [],
            "recent": [],
        }

    records: list[dict] = []
    try:
        for line in hits_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(_json.loads(line))
    except Exception:
        logger.debug("Failed to read memory hits", exc_info=True)

    sessions: set[str] = set()
    topic_counter: Counter[str] = Counter()
    cached_count = 0

    for r in records:
        sessions.add(r.get("session_id", ""))
        for t in r.get("topics", []):
            topic_counter[t] += 1
        if r.get("cached"):
            cached_count += 1

    total = len(records)
    top_topics = [
        {"topic": t, "count": c}
        for t, c in topic_counter.most_common(10)
    ]
    recent = [
        {"ts": r.get("ts"), "session_id": r.get("session_id", "")[:8], "topics": r.get("topics", []), "query": r.get("query", "")}
        for r in records[-20:]
    ]
    recent.reverse()

    return {
        "total_injections": total,
        "unique_sessions": len(sessions),
        "cache_hit_rate": round((cached_count / total) * 100) if total else 0,
        "top_topics": top_topics,
        "recent": recent,
    }


@router.get("/topics")
async def list_topics() -> dict[str, Any]:
    """List all memory topic files with their parsed frontmatter."""
    mem = _memory_dir()
    topics = []
    for f in sorted(mem.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        if f.name == "MEMORY.md":
            continue
        try:
            meta = await asyncio.to_thread(_parse_topic, f)
            topics.append({k: v for k, v in meta.items() if k != "content"})
        except Exception:
            logger.debug("Could not parse topic %s", f.name, exc_info=True)
    return {"topics": topics}


@router.get("/topics/{filename}")
async def get_topic(filename: str) -> dict[str, Any]:
    """Return full content and frontmatter for a single topic file."""
    target = _topic_path(filename)
    if target is None:
        return JSONResponse(status_code=400, content={"error": "Invalid filename"})
    if not target.exists():
        return JSONResponse(status_code=404, content={"error": "Topic not found"})
    return await asyncio.to_thread(_parse_topic, target)


@router.put("/topics/{filename}")
async def update_topic(filename: str, body: dict = Body(...)) -> dict[str, Any]:
    """Replace a topic file's content (user edit).

    Stamps ``updated_at`` as now and appends a user-edit marker to the
    corrections block so there is an audit trail of manual changes.
    """
    target = _topic_path(filename)
    if target is None:
        return JSONResponse(status_code=400, content={"error": "Invalid filename"})
    if not target.exists():
        return JSONResponse(status_code=404, content={"error": "Topic not found"})

    content = body.get("content", "")
    if not content:
        return JSONResponse(status_code=400, content={"error": "content is required"})

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Patch updated_at in the frontmatter
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            fm_block = content[3:end]
            patched = re.sub(r'updated_at:\s*"[^"]*"', f'updated_at: "{now_iso}"', fm_block)
            if patched == fm_block:
                patched = fm_block.rstrip() + f'\nupdated_at: "{now_iso}"'
            content = f"---{patched}{content[end:]}"

    await asyncio.to_thread(target.write_text, content, "utf-8")
    return {"status": "updated", "filename": filename, "updated_at": now_iso}


@router.delete("/topics/{filename}")
async def delete_topic(filename: str) -> dict[str, Any]:
    """Delete a topic file from memory."""
    target = _topic_path(filename)
    if target is None:
        return JSONResponse(status_code=400, content={"error": "Invalid filename"})
    if not target.exists():
        return JSONResponse(status_code=404, content={"error": "Topic not found"})
    await asyncio.to_thread(target.unlink)
    # Remove from embedding index if enabled
    try:
        cfg = await AppConfig.aload()
        if cfg.memory.embedding_enabled:
            from backend.embedding_index import get_embedding_index
            idx = await get_embedding_index()
            await idx.remove_source(str(target))
    except Exception:
        logger.debug("Could not remove %s from embedding index", filename, exc_info=True)
    return {"status": "deleted", "filename": filename}


@router.post("/topics/{filename}/correction")
async def add_correction(filename: str, body: dict = Body(...)) -> dict[str, Any]:
    """Append a user correction to the ``## Corrections`` section.

    Creates the section if it doesn't exist.  The correction text is
    prefixed with the current ISO timestamp so there is a clear audit trail.
    """
    target = _topic_path(filename)
    if target is None:
        return JSONResponse(status_code=400, content={"error": "Invalid filename"})
    if not target.exists():
        return JSONResponse(status_code=404, content={"error": "Topic not found"})

    text = body.get("text", "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "text is required"})

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    correction_line = f"- [{now_iso}] {text}"

    content = await asyncio.to_thread(target.read_text, "utf-8")
    if _CORRECTIONS_HEADING in content:
        content = content.rstrip("\n") + f"\n{correction_line}\n"
    else:
        content = content.rstrip("\n") + f"\n\n{_CORRECTIONS_HEADING}\n{correction_line}\n"

    await asyncio.to_thread(target.write_text, content, "utf-8")
    return {"status": "added", "correction": correction_line}


@router.get("/stats")
async def get_memory_stats() -> dict[str, Any]:
    """Return transcript and memory stats for the UI dashboard."""
    from backend.config import get_app_data_dir
    from backend.consolidation_lock import last_consolidated_at
    from backend.session_transcript import list_transcripts_since

    cfg = (await AppConfig.aload()).memory
    since = last_consolidated_at()
    all_transcripts = list_transcripts_since(0)
    pending_transcripts = list_transcripts_since(since)

    mem_dir = get_app_data_dir() / "memory"
    memory_files = 0
    if mem_dir.exists():
        memory_files = sum(
            1 for f in mem_dir.glob("*.md")
            if f.name != "MEMORY.md"
        )

    last_ms = last_consolidated_at()

    return {
        "total_transcripts": len(all_transcripts),
        "pending_transcripts": len(pending_transcripts),
        "memory_files": memory_files,
        "last_consolidated_at": last_ms if last_ms > 0 else None,
        "retention_days": cfg.retention_days,
    }
