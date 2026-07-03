"""Memory — background memory consolidation agent.

Reads session transcripts and distills durable knowledge into
structured memory files under ``<app_data>/memory/``.

The 4-phase pipeline mirrors Claude Code's dream-mode pattern:
  1. **Orient**  — read MEMORY.md index + topic file frontmatter.
  2. **Gather**  — read candidate transcript JSONL files.
  3. **Consolidate** — LLM call to produce memory diffs.
  4. **Prune**   — enforce limits (max files, index size).

Status is tracked in-memory so the UI can poll progress.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from backend.config import MemoryConfig, get_app_data_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Run status (shared singleton for UI polling)
# ---------------------------------------------------------------------------


class RunState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class MemoryStatus:
    state: RunState = RunState.IDLE
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    transcripts_processed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "transcripts_processed": self.transcripts_processed,
        }


_status = MemoryStatus()
_cancel_event = asyncio.Event()


def get_status() -> MemoryStatus:
    return _status


def request_cancel() -> None:
    _cancel_event.set()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _memory_dir() -> Path:
    d = get_app_data_dir() / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _index_path() -> Path:
    return _memory_dir() / "MEMORY.md"


# ---------------------------------------------------------------------------
# Phase 1 — Orient: read existing memory index + topic descriptions
# ---------------------------------------------------------------------------


def _read_index() -> str:
    p = _index_path()
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


def _scan_topic_files() -> list[dict[str, str]]:
    """Return {name, description, path} for .md files."""
    results: list[dict[str, str]] = []
    for f in _memory_dir().glob("*.md"):
        if f.name == "MEMORY.md":
            continue
        header: dict[str, str] = {"path": f.name}
        try:
            text = f.read_text(encoding="utf-8")
            for line in text.split("\n")[:10]:
                s = line.strip()
                if s.startswith("name:"):
                    val = s.split(":", 1)[1]
                    header["name"] = val.strip().strip('"')
                elif s.startswith("description:"):
                    val = s.split(":", 1)[1]
                    header["description"] = (
                        val.strip().strip('"')
                    )
        except Exception:
            pass
        results.append(header)
    return results


# ---------------------------------------------------------------------------
# Phase 2 — Gather: read candidate transcripts
# ---------------------------------------------------------------------------


def _read_transcripts(
    candidates: list[dict[str, Any]],
    max_chars: int = 200_000,
) -> str:
    """Concatenate transcript content up to *max_chars*."""
    from backend.session_transcript import _transcript_path

    parts: list[str] = []
    total = 0
    for c in candidates:
        sid = c["session_id"]
        p = _transcript_path(sid)
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8")
            if total + len(text) > max_chars:
                remaining = max_chars - total
                if remaining > 500:
                    parts.append(
                        f"--- Session {sid} (truncated)"
                        f" ---\n{text[:remaining]}"
                    )
                break
            parts.append(
                f"--- Session {sid} ---\n{text}"
            )
            total += len(text)
        except Exception:
            logger.debug(
                "Failed to read transcript %s",
                sid, exc_info=True,
            )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Phase 3 — Consolidate: LLM call
# ---------------------------------------------------------------------------


CONSOLIDATION_PROMPT = """\
You are the memory consolidation agent. Your task is to analyze \
session transcripts and produce structured memory updates.

## Current Memory Index

<memory_index>
{index}
</memory_index>

## Existing Topic Files

<topic_files>
{topics}
</topic_files>

## Session Transcripts (session IDs: {session_ids})

<transcripts>
{transcripts}
</transcripts>

## Instructions

Analyze the transcripts and extract durable knowledge. \
Produce a JSON response with this structure:

```json
{{
  "updates": [
    {{
      "file": "topic-name.md",
      "action": "create" | "update" | "skip",
      "content": "full markdown content including YAML frontmatter",
      "source_sessions": ["session_id_1", "session_id_2"],
      "confidence": "high" | "medium" | "low",
      "reason": "why this update matters"
    }}
  ],
  "index_update": "updated MEMORY.md content (full replacement)",
  "summary": "one-line summary of what changed"
}}
```

## Memory Types (frontmatter ``type`` field)
- **user** — preferences, role, communication style
- **feedback** — corrections, confirmed/rejected approaches
- **project** — goals, constraints, architecture decisions
- **reference** — external links, tool IDs, dashboards

## Frontmatter fields (include in ``content``)
- ``name``, ``description``, ``type`` — always required.
- ``source_sessions`` — list of session IDs from the ``session_ids`` \
above that contributed facts to this topic.  Use short 8-char prefixes.
- ``confidence`` — your certainty: ``high`` (explicit statement), \
``medium`` (clear implication), ``low`` (inference).
- Do NOT include ``created_at`` or ``updated_at`` — those are managed \
by the system and will be overwritten.

## Rules
- Do NOT store API keys, passwords, or credentials.
- Do NOT store ephemeral task details or one-off questions.
- Do NOT duplicate information already in memory.
- Keep each topic file focused on a single subject.
- Do NOT touch or reproduce any ``## Corrections`` section you see in \
existing topic files — corrections are user-managed and will be \
preserved automatically.
- If nothing worth saving, return:\
 {{"updates": [], "index_update": null, "summary": "no updates"}}
"""


async def _run_consolidation_llm(
    index: str, topics: str, transcripts: str, *, session_ids: str = "",
) -> dict[str, Any]:
    """Execute the consolidation LLM call."""
    from backend.config import AppConfig
    from backend.memory_relevance import (
        _create_ranking_model,
    )

    cfg = await AppConfig.aload()
    cfg.apply_to_environ()
    # Consolidation emits a structured JSON document covering every memory
    # update, which is far larger than the per-turn ranking output.  The
    # 512-token MLX default truncates JSON mid-string and breaks parsing,
    # so we lift the cap for this call only.  Frontier models ignore the
    # override (Anthropic / Bedrock cap is server-side).
    model = _create_ranking_model(
        cfg.memory, cfg.llm.provider, mlx_max_tokens=8192,
    )

    prompt = CONSOLIDATION_PROMPT.format(
        index=index or "(empty)",
        topics=topics or "(none)",
        transcripts=transcripts,
        session_ids=session_ids or "(unknown)",
    )

    from langchain_core.messages import (
        HumanMessage, SystemMessage,
    )
    response = await model.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(
            content="Analyze the transcripts above "
            "and produce memory updates.",
        ),
    ])

    text = response.content
    if isinstance(text, list):
        text = " ".join(
            p.get("text", "") for p in text
            if isinstance(p, dict) and p.get("type") == "text"
        )
    text = str(text).strip()
    try:
        return _extract_json(text)
    except (ValueError, json.JSONDecodeError) as exc:
        # Most common cause: the local model hit its generation cap and
        # truncated the JSON.  Log the head + tail so the user can see it.
        head = text[:300].replace("\n", "\\n")
        tail = text[-300:].replace("\n", "\\n")
        logger.error(
            "[memory] consolidation LLM returned unparseable JSON (%d chars). "
            "head=%r tail=%r",
            len(text), head, tail,
        )
        raise RuntimeError(
            f"Consolidation model returned unparseable JSON ({exc}). "
            "If using an MLX model, the response was likely truncated; "
            "raise MLX_MAX_TOKENS or use a stronger model.",
        ) from exc


def _extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from LLM text."""
    import re

    # Fenced code block (```json ... ```)
    fence_match = re.search(
        r"```(?:json)?\s*\n(\{.*?\})\s*\n```",
        text,
        re.DOTALL,
    )
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 2: find the outermost balanced { ... } by tracking brace depth.
    # The naive find/rfind approach fails when the content fields themselves
    # contain braces or when the LLM wraps the JSON in prose with braces.
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object in LLM response: {text[:200]}")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break

    # Strategy 3: fall back to the old approach as a last resort
    end = text.rfind("}") + 1
    if end == 0:
        raise ValueError(f"No JSON object in LLM response: {text[:200]}")
    return json.loads(text[start:end])


# ---------------------------------------------------------------------------
# Phase 4 — Write results + prune
# ---------------------------------------------------------------------------

# Sentinel that marks the start of a user-managed corrections block.
# Lines at or after this heading are preserved across consolidation runs.
_CORRECTIONS_HEADING = "## Corrections"


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter from body.  Returns (fields_dict, body_text).

    Handles the common subset of YAML used by memory topic files:
    string scalars, quoted strings, and simple lists of strings.  Uses
    stdlib only — no PyYAML dependency required.
    """
    import re

    fields: dict[str, Any] = {}
    body = content
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            fm_block = content[3:end].strip()
            body = content[end + 4:].lstrip("\n")
            for line in fm_block.splitlines():
                m = re.match(r'^(\w+)\s*:\s*(.*)$', line.strip())
                if not m:
                    continue
                key, val = m.group(1), m.group(2).strip()
                if val.startswith("[") and val.endswith("]"):
                    # Simple list — extract quoted or bare tokens
                    items = re.findall(r'"([^"]+)"|\'([^\']+)\'|([\w\-]+)', val[1:-1])
                    fields[key] = [a or b or c for a, b, c in items]
                else:
                    fields[key] = val.strip('"\'')
    return fields, body


def _extract_corrections(body: str) -> tuple[str, str]:
    """Split body into (body_without_corrections, corrections_block).

    ``corrections_block`` is empty string when the heading is absent.
    """
    idx = body.find(_CORRECTIONS_HEADING)
    if idx == -1:
        return body, ""
    return body[:idx].rstrip("\n"), body[idx:]


def _inject_provenance(
    content: str,
    *,
    created_at: str,
    updated_at: str,
    source_sessions: list[str],
    confidence: str,
) -> str:
    """Rewrite or append provenance fields in YAML frontmatter.

    Preserves all existing frontmatter fields; only the four provenance
    fields are injected / overwritten so the LLM's ``name``,
    ``description``, ``type`` are kept intact.
    """
    import re

    _PROVENANCE_KEYS = {"created_at", "updated_at", "source_sessions", "confidence"}

    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            fm_block = content[3:end]
            body_tail = content[end + 3:]  # keeps the trailing \n---\nbody

            # Remove any existing provenance lines
            clean_lines = [
                ln for ln in fm_block.splitlines()
                if not re.match(r'^\s*(' + '|'.join(_PROVENANCE_KEYS) + r')\s*:', ln)
            ]

            sessions_yaml = (
                "[" + ", ".join(f'"{s}"' for s in source_sessions) + "]"
                if source_sessions else "[]"
            )
            clean_lines += [
                f'created_at: "{created_at}"',
                f'updated_at: "{updated_at}"',
                f'source_sessions: {sessions_yaml}',
                f'confidence: {confidence}',
            ]
            new_fm = "\n".join(clean_lines)
            return f"---\n{new_fm}{body_tail}"

    # No frontmatter — prepend one
    sessions_yaml = (
        "[" + ", ".join(f'"{s}"' for s in source_sessions) + "]"
        if source_sessions else "[]"
    )
    fm = (
        f"---\n"
        f'created_at: "{created_at}"\n'
        f'updated_at: "{updated_at}"\n'
        f'source_sessions: {sessions_yaml}\n'
        f'confidence: {confidence}\n'
        f"---\n"
    )
    return fm + content


def _apply_updates(result: dict[str, Any]) -> int:
    """Write topic files and update index, injecting provenance fields."""
    mem = _memory_dir()
    written = 0
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for update in result.get("updates", []):
        action = update.get("action", "skip")
        if action == "skip":
            continue
        filename = update.get("file", "")
        content = update.get("content", "")
        if not filename or not content:
            continue
        if not filename.endswith(".md"):
            filename += ".md"

        filename = Path(filename).name
        if not filename or filename == "MEMORY.md":
            continue
        target = mem / filename
        if not target.resolve().is_relative_to(mem.resolve()):
            logger.warning("[memory] rejected path-traversal filename: %s", filename)
            continue

        # ── Provenance: read existing file to preserve created_at + corrections
        created_at = now_iso
        existing_source_sessions: list[str] = []
        corrections_block = ""

        if target.exists():
            try:
                existing_text = target.read_text(encoding="utf-8")
                existing_fm, existing_body = _parse_frontmatter(existing_text)
                if existing_ca := existing_fm.get("created_at"):
                    created_at = str(existing_ca)
                existing_source_sessions = existing_fm.get("source_sessions") or []
                _, corrections_block = _extract_corrections(existing_body)
            except Exception:
                logger.debug("[memory] could not read existing %s for provenance", filename)

        # Merge source_sessions: union of existing + LLM-provided
        llm_sessions: list[str] = update.get("source_sessions") or []
        merged_sessions = list(dict.fromkeys(existing_source_sessions + llm_sessions))

        confidence = str(update.get("confidence") or "medium").lower()
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"

        # Strip the corrections block the LLM may have hallucinated into content
        new_body_part, _ = _extract_corrections(
            _parse_frontmatter(content)[1]
        )

        # Rebuild: inject provenance into frontmatter, then re-attach corrections
        content_with_provenance = _inject_provenance(
            content,
            created_at=created_at,
            updated_at=now_iso,
            source_sessions=merged_sessions,
            confidence=confidence,
        )

        # Re-attach the user's corrections block
        if corrections_block:
            # Re-strip the LLM body so corrections always appear last
            _, body_no_prov = _parse_frontmatter(content_with_provenance)
            clean_body, _ = _extract_corrections(body_no_prov)
            fm_end = content_with_provenance.find("\n---\n", 3)
            fm_part = content_with_provenance[: fm_end + 5] if fm_end != -1 else ""
            content_with_provenance = (
                fm_part + clean_body.rstrip("\n") + "\n\n" + corrections_block
            )

        target.write_text(content_with_provenance, encoding="utf-8")
        written += 1
        logger.info("[memory] wrote %s (%s)", filename, action)

    index_update = result.get("index_update")
    if index_update:
        _index_path().write_text(
            str(index_update), encoding="utf-8",
        )
        logger.info("[memory] updated MEMORY.md index")

    return written


def _prune(cfg: MemoryConfig) -> None:
    """Enforce limits on memory file count and index size."""
    mem = _memory_dir()
    topic_files = sorted(
        (f for f in mem.glob("*.md") if f.name != "MEMORY.md"),
        key=lambda f: f.stat().st_mtime,
    )

    while len(topic_files) > cfg.max_memory_files:
        oldest = topic_files.pop(0)
        oldest.unlink()
        logger.info(
            "[memory] pruned %s (limit=%d)",
            oldest.name, cfg.max_memory_files,
        )

    idx = _index_path()
    if idx.exists():
        size_kb = idx.stat().st_size / 1024
        if size_kb > cfg.max_index_kb:
            logger.warning(
                "[memory] MEMORY.md is %.1fKB "
                "(limit %dKB)",
                size_kb, cfg.max_index_kb,
            )


# ---------------------------------------------------------------------------
# Orchestrator — called from session_manager or API
# ---------------------------------------------------------------------------


async def execute_consolidation(
    watermark_ms: float,
    candidates: list[dict[str, Any]],
    cfg: MemoryConfig,
) -> None:
    """Full memory consolidation pipeline (background task)."""
    from backend.consolidation_lock import release
    from backend.session_transcript import TranscriptRotator

    _cancel_event.clear()
    _status.state = RunState.RUNNING
    _status.started_at = datetime.now(timezone.utc).isoformat()
    _status.finished_at = None
    _status.error = None
    _status.transcripts_processed = 0

    try:
        if _cancel_event.is_set():
            _status.state = RunState.CANCELLED
            _status.finished_at = datetime.now(timezone.utc).isoformat()
            release(rollback_to_ms=watermark_ms)
            return

        index = await asyncio.to_thread(_read_index)
        topics_list = await asyncio.to_thread(_scan_topic_files)
        topics_str = "\n".join(
            "- {}: {}".format(
                t.get("name", t["path"]),
                t.get("description", "(no description)"),
            )
            for t in topics_list
        )

        if _cancel_event.is_set():
            _status.state = RunState.CANCELLED
            _status.finished_at = datetime.now(timezone.utc).isoformat()
            release(rollback_to_ms=watermark_ms)
            return

        transcripts = await asyncio.to_thread(_read_transcripts, candidates)
        if not transcripts.strip():
            logger.info("[memory] no transcript content to process")
            _status.state = RunState.SUCCESS
            _status.finished_at = datetime.now(timezone.utc).isoformat()
            release(update_mtime=True)
            return

        if _cancel_event.is_set():
            _status.state = RunState.CANCELLED
            _status.finished_at = datetime.now(timezone.utc).isoformat()
            release(rollback_to_ms=watermark_ms)
            return

        # Short session ID prefixes for the provenance prompt
        session_ids_str = ", ".join(
            c["session_id"][:8] for c in candidates[:20]
        )

        # Run the LLM call as a separate task so the cancel event can
        # interrupt it mid-flight without waiting for the full response.
        llm_task = asyncio.create_task(
            _run_consolidation_llm(
                index, topics_str, transcripts, session_ids=session_ids_str,
            )
        )
        cancel_waiter = asyncio.create_task(_cancel_event.wait())
        done, pending = await asyncio.wait(
            {llm_task, cancel_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()

        if cancel_waiter in done:
            llm_task.cancel()
            try:
                await llm_task
            except (asyncio.CancelledError, Exception):
                pass
            _status.state = RunState.CANCELLED
            _status.finished_at = datetime.now(timezone.utc).isoformat()
            release(rollback_to_ms=watermark_ms)
            return

        result = await llm_task

        if _cancel_event.is_set():
            _status.state = RunState.CANCELLED
            _status.finished_at = datetime.now(timezone.utc).isoformat()
            release(rollback_to_ms=watermark_ms)
            return

        written = await asyncio.to_thread(
            _apply_updates, result,
        )
        await asyncio.to_thread(_prune, cfg)

        # Re-index memory files now that content has changed
        try:
            from backend.config import AppConfig as _AppConfig
            _cfg = await _AppConfig.aload()
            if _cfg.memory.embedding_enabled:
                from backend.embedding_index import get_embedding_index
                _idx = await get_embedding_index()
                asyncio.create_task(_idx.index_memory())
        except Exception:
            logger.debug("[memory] embedding trigger failed after consolidation", exc_info=True)

        _status.transcripts_processed = len(candidates)
        _status.state = RunState.SUCCESS
        _status.finished_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            "[memory] done — %d file(s), %d transcript(s)",
            written, len(candidates),
        )

    except Exception as exc:
        logger.exception("[memory] consolidation failed")
        _status.state = RunState.ERROR
        _status.error = str(exc)[:500]
        _status.finished_at = datetime.now(timezone.utc).isoformat()
        release(rollback_to_ms=watermark_ms)
        return

    release(update_mtime=True)

    rotator = TranscriptRotator(
        max_age_days=cfg.retention_days, max_files=200,
    )
    await rotator.rotate_async(watermark_ms=watermark_ms)
