"""Hybrid retrieval and rendering for ``recall_procedure``.

1. Hard filters (SQL): admitted episode, active item, same agent family, same app when one is given, not the current
   benchmark task, not a near-duplicate of the current task's text.
2. Score: dense cosine on the item's intent (the query is the agent's generic description of the step), blended with
   BM25 over intent + recipe text for exact menu and feature names, plus small priors (trusted label, use record).
3. Floor: nothing is returned when the best cosine is below the floor. An unrelated example is worse than none.
4. One item (k=1: ReasoningBank's k-curve fell from 49.7 at k=1 to 44.4 at k=4), with up to three facts / lessons
   for the same app attached, under a token cap. Detail (all steps, a script) only by id on a follow-up call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from backend.procedural_memory import compact
from backend.procedural_memory.embed import Embedder
from backend.procedural_memory.store import ItemRow, ProcedureStore, tokens

DENSE_WEIGHT = 0.75
LEXICAL_WEIGHT = 0.25
TRUSTED_SOURCES = frozenset({"user", "benchmark", "verified", "run"})
NEAR_DUPLICATE_JACCARD = 0.8
MAX_ATTACHED = 3


@dataclass
class Candidate:
    item: ItemRow
    cos: float
    lexical: float
    score: float
    cut: str | None = None          # why it was not returned (floor, rank)

    def as_log(self) -> dict[str, Any]:
        return {"item": self.item.id, "kind": self.item.kind, "cos": round(self.cos, 4),
                "lexical": round(self.lexical, 4), "score": round(self.score, 4), "cut": self.cut}


@dataclass
class Recall:
    query: str
    app: str | None
    text: str
    returned: list[ItemRow] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    abstained: bool = False
    tokens: int = 0

    def used(self) -> list[dict[str, Any]]:
        out = []
        for it in self.returned:
            c = next((c for c in self.candidates if c.item.id == it.id), None)
            out.append({"item": it.id, "episode": it.episode_id, "kind": it.kind, "source_task": it.task_id,
                        "label_source": it.label_source, "score": round(c.score, 4) if c else None,
                        "cos": round(c.cos, 4) if c else None})
        return out


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


_WORD_RE = re.compile(r"[a-z0-9<>]+")


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(compact.strip_values(text).lower()))


def near_duplicate(a: str, b: str) -> bool:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= NEAR_DUPLICATE_JACCARD


async def recall(store: ProcedureStore, embedder: Embedder, *, query: str, family: str, app: str | None = None,
                 floor: float = 0.45, max_tokens: int = 800, exclude_task_ids: set[str] = frozenset(),
                 current_task_text: str = "", usage: dict[str, dict[str, float]] | None = None,
                 exact_only: bool = False) -> Recall:
    """``current_task_text`` leaves out stored runs of the current task: near-duplicates too (same template, other
    values: a frozen benchmark snapshot), or with ``exact_only`` only the very same text (live memory, where a
    similar task is exactly what should be recalled)."""
    apps = [app] if app else None
    pool = store.candidate_items(family=family, apps=apps, kinds=("recipe", "fact", "lesson"),
                                 exclude_task_ids=exclude_task_ids)
    if current_task_text:
        same = ((lambda a: " ".join(a.split()).lower() == " ".join(current_task_text.split()).lower()) if exact_only
                else (lambda a: near_duplicate(a, current_task_text)))
        pool = [(it, v) for it, v in pool if not (it.task_text and same(it.task_text))]
    out = Recall(query=query, app=app, text="")
    if not pool:
        out.abstained = True
        out.text = _nothing(app)
        return out
    qvec = (await embedder.embed([query]))[0]
    lex = store.bm25(query)
    top = max(lex.values(), default=0.0) or 1.0
    usage = usage or {}
    cands = []
    for it, vec in pool:
        cos = _dot(qvec, vec)
        lexical = lex.get(it.id, 0.0) / top
        prior = 0.02 if it.label_source in TRUSTED_SOURCES else 0.0
        u = usage.get(it.id)
        if u and u.get("known", 0) >= 3 and u["passes"] / u["known"] < 0.2:
            prior -= 0.05
        cands.append(Candidate(item=it, cos=cos, lexical=lexical, score=DENSE_WEIGHT * cos + LEXICAL_WEIGHT * lexical + prior))
    cands.sort(key=lambda c: c.score, reverse=True)
    out.candidates = cands[:20]

    recipes = [c for c in cands if c.item.kind == "recipe"]
    notes = [c for c in cands if c.item.kind in ("fact", "lesson")]
    best = next((c for c in recipes if c.cos >= floor), None)
    for c in out.candidates:
        if c.cos < floor:
            c.cut = "floor"
    attached: list[Candidate] = []
    note_floor = floor * 0.85
    pick_from = sorted(notes, key=lambda c: (best is not None and c.item.episode_id == best.item.episode_id,
                                              c.score), reverse=True)
    seen_notes: set[str] = set()
    for c in pick_from:
        if c.cos >= note_floor and len(attached) < MAX_ATTACHED and c.item.app == (best.item.app if best else c.item.app):
            key = _note_key(c.item)
            if key in seen_notes:          # the same fact from another run ("... (3 tries" vs "(4 tries"): say it once
                c.cut = "duplicate"
                continue
            seen_notes.add(key)
            attached.append(c)
    # Success and failure both: if the notes picked say nothing about what went wrong before, swap in the best
    # note that does (from a failed run), so a recipe is never offered without its known pitfalls.
    if attached and not any(c.item.from_outcome == "failure" for c in attached):
        wrong = next((c for c in pick_from if c.item.from_outcome == "failure" and c not in attached
                      and c.cos >= note_floor and c.item.app == attached[0].item.app
                      and _note_key(c.item) not in seen_notes), None)
        if wrong is not None:
            if len(attached) >= MAX_ATTACHED:
                attached.pop()
            attached.append(wrong)
    if best is None and not attached:
        out.abstained = True
        out.text = _nothing(app)
        return out
    for c in out.candidates:
        if c.cut is None and c is not best and c not in attached:
            c.cut = "rank"
    out.returned = ([best.item] if best else []) + [c.item for c in attached]
    out.text = render(best.item if best else None, [c.item for c in attached], max_tokens=max_tokens)
    out.tokens = tokens(out.text)
    return out


def _nothing(app: str | None) -> str:
    where = f" for {app}" if app else ""
    return (f"No similar procedure{where} in memory. Work it out from the screen; do not call recall_procedure again "
            "for this step.")


def _source_note(it: ItemRow) -> str:
    return {"benchmark": "a benchmark run", "user": "a run you confirmed", "verified": "a verified run",
            "run": "a successful run", "segment_evidence": "steps confirmed by the trace (that run as a whole failed)",
            "judge": "a judged run"}.get(it.label_source or "", "a past run")


def _note_key(it: ItemRow) -> str:
    return re.sub(r"[\d\W]+", " ", _note_line(it).lower()).strip()


def _note_line(it: ItemRow) -> str:
    if it.kind == "fact":
        return f"- {it.body.get('text', '')}"
    title, sentence, content = it.body.get("title", ""), it.body.get("sentence", ""), it.body.get("content", "")
    prefix = "Check: " if it.from_outcome == "failure" else ""
    return f"- {prefix}{title}. {content or sentence}".strip()


def render(item: ItemRow | None, notes: list[ItemRow], *, max_tokens: int = 800) -> str:
    lines: list[str] = []
    if item is not None:
        lines.append(f"[proc {item.id}] {item.intent}")
        lines.append(f"source: {_source_note(item)}. These are steps from another task: confirm each against the "
                     "screen and never copy values from it.")
        for n, ln in enumerate(item.body.get("lines", []), 1):
            lines.append(f"{n}. {ln}")
        if item.body.get("outline"):
            lines.append("Parts: " + "; ".join(item.body["outline"]))
    if notes:
        lines.append("Known in this app:" if item is not None else
                     f"No matching procedure; known in {notes[0].app or 'this app'}:")
        lines.extend(_note_line(n) for n in notes)
        if any(n.from_outcome == "failure" for n in notes):
            lines.append("(Notes marked Check: or 'changed nothing' come from runs that went wrong.)")
    more = []
    if item is not None:
        if item.body.get("has_script"):
            more.append(f'recall_procedure(id="{item.id}", detail="script")')
        if item.body.get("next"):
            more.append(f'recall_procedure(id="{item.body["next"]}") for the next part')
    if more:
        lines.append("More: " + "; ".join(more))
    text = "\n".join(lines)
    while tokens(text) > max_tokens and len(lines) > 3:
        # Drop recipe lines from the end first, then notes; keep the header and the source line.
        drop = next((k for k in range(len(lines) - 1, 1, -1) if re.match(r"\d+\. ", lines[k])), len(lines) - 1)
        lines.pop(drop)
        text = "\n".join(lines)
    return text


def detail(store: ProcedureStore, item_id: str, what: str = "recipe", max_tokens: int = 1500) -> str:
    it = store.item(item_id)
    if it is None:
        return f"No procedure with id {item_id!r}."
    if what == "script":
        arts = store.artifacts_for(item_id)
        if not arts:
            return f"[proc {item_id}] has no script."
        parts = [f"[proc {item_id}] scripts (from another task: replace its values, and run one that writes or "
                 "sends only when this task asks for that):"]
        for a in arts:
            flag = " (writes or sends)" if a["writes_or_sends"] else ""
            parts.append(f"--- {a['kind']}{flag}\n{a['body']}")
        text = "\n".join(parts)
        return text if tokens(text) <= max_tokens else text[: max_tokens * 4] + "\n[truncated]"
    if what == "steps":
        ep = store.episode(it.episode_id) or {}
        segs = (ep.get("data") or {}).get("segments") or []
        seg = next((s for s in segs if s.get("ord") == it.ord), None)
        lines = (seg or {}).get("all_lines") or it.body.get("lines", [])
        text = "\n".join([f"[proc {item_id}] all steps:"] + [f"{n}. {ln}" for n, ln in enumerate(lines, 1)])
        return text if tokens(text) <= max_tokens else text[: max_tokens * 4] + "\n[truncated]"
    return render(it, [], max_tokens=max_tokens)
