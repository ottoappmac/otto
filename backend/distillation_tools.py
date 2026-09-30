"""LangChain tools: train, catalog, and spawn a session on a distilled LoRA.

LoRA adapters are Standard (in-process MLX) only.  Turbo / exo cannot attach
them.  ``use_distilled_model`` hands off to a **new** session whose graph is
built with the adapter — the current session is left unchanged.
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_MAX_PROMPT_ADAPTERS = 10


def _public_row(rec: dict, *, bound_id: str = "") -> dict:
    return {
        "catalog_id": rec.get("catalog_id") or "",
        "display_name": rec.get("display_name") or "",
        "purpose": rec.get("purpose") or "",
        "kind": rec.get("kind") or "",
        "base_repo_id": rec.get("base_repo_id") or "",
        "teacher_model_id": rec.get("teacher_model_id") or "",
        "trained_at": rec.get("trained_at") or "",
        "size_mb": rec.get("size_mb") or 0,
        "bound_to_this_session": (rec.get("catalog_id") or "") == bound_id,
    }


def build_distilled_adapters_prompt_block(bound_catalog_id: str | None = None) -> str:
    """Compact ``<distilled_adapters>`` list for the orchestrator prompt."""
    try:
        from backend.distillation.catalog import iter_trained_adapters, student_short

        recs = iter_trained_adapters()
    except Exception:  # noqa: BLE001
        logger.debug("distilled adapters prompt block skipped", exc_info=True)
        recs = []
    bound = (bound_catalog_id or "").strip()
    lines = [
        "<distilled_adapters>",
        "Trained LoRAs for this machine (Standard / in-process MLX only).",
        "If a row's purpose matches this run, call use_distilled_model(catalog_id, prompt).",
        "That spawns a NEW session on the LoRA (this session stays as-is; does not change Settings).",
        "If none match, keep the current model or start_distill_train with a purpose.",
        "Training needs exclusive GPU; Turbo/exo cannot load adapters.",
    ]
    if not recs:
        lines.append("(none trained yet)")
        lines.append("</distilled_adapters>")
        return "\n".join(lines)
    for rec in recs[:_MAX_PROMPT_ADAPTERS]:
        cid = rec.get("catalog_id") or ""
        purpose = (rec.get("purpose") or "").strip() or "(no purpose set)"
        student = student_short(str(rec.get("base_repo_id") or ""))
        mark = " [BOUND]" if cid == bound else ""
        lines.append(f"- {cid}{mark} — {purpose} — student {student}")
    extra = len(recs) - _MAX_PROMPT_ADAPTERS
    if extra > 0:
        lines.append(f"(+{extra} more; call list_distilled_models)")
    lines.append("</distilled_adapters>")
    return "\n".join(lines)


def build_distillation_tools(session_id: str) -> list:
    """Return agent-facing distillation tools closed over *session_id*."""

    def _bound_id() -> str:
        try:
            from backend.state import session_mgr

            sess = session_mgr.get_session(session_id)
            return (getattr(sess, "distill_catalog_id", None) or "") if sess else ""
        except Exception:  # noqa: BLE001
            return ""

    @tool
    def list_distilled_models() -> str:
        """List trained LoRA adapters and what each one is for.

        Use this to pick an adapter whose purpose matches the current task,
        then call use_distilled_model to spawn a new session on that LoRA.
        Adapters run on Standard (in-process MLX) only — Turbo and exo cannot
        attach LoRA.
        """
        from backend.distillation.catalog import iter_trained_adapters

        bound = _bound_id()
        rows = [_public_row(r, bound_id=bound) for r in iter_trained_adapters()]
        return json.dumps({"adapters": rows, "bound_catalog_id": bound}, indent=2)

    @tool
    def distill_census() -> str:
        """Count high-quality teacher trajectories available for LoRA training.

        Call this before start_distill_train.  If n_trajectories is 0, training
        will fail until more teacher sessions exist.
        """
        from backend.distillation.census import run_census
        from backend.distillation.job import blocking_sessions

        report = run_census()
        report["blocking_sessions"] = blocking_sessions()
        return json.dumps(report, indent=2)

    @tool
    async def start_distill_train(
        purpose: str,
        name: str = "activity",
        iters: int | None = None,
        teacher_model_id: str | None = None,
        student_model_id: str | None = None,
    ) -> str:
        """Train a LoRA adapter from session trajectories (exclusive GPU).

        Requires a short purpose describing what the adapter is for so later
        sessions can be assigned to it.  Standard MLX only.

        If THIS session is currently generating on in-process MLX, training is
        queued and starts when the turn ends.  Poll distill_status.

        Args:
            purpose: Required.  What this LoRA is trained to do (assignment hint).
            name: Short kind slug (default "activity").
            iters: Optional LoRA iteration count.
            teacher_model_id: Optional teacher Hub id override.
            student_model_id: Optional student Hub id override.
        """
        from backend.distillation.job import DistillBusy, start_train_job

        text = (purpose or "").strip()
        if not text:
            return "Error: purpose is required — describe what this adapter is for."
        try:
            status = await start_train_job(
                iters=iters,
                output_name=name or "activity",
                purpose=text,
                teacher_id=teacher_model_id,
                student_id=student_model_id,
                defer_session_id=session_id,
            )
        except DistillBusy as exc:
            return json.dumps({
                "error": str(exc),
                "code": exc.code,
                "blocking_sessions": exc.blocking,
            }, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.exception("start_distill_train failed")
            return f"Error: {exc}"
        payload = status.to_dict()
        if status.state == "queued":
            payload["note"] = (
                "Queued until this MLX turn finishes generating. Poll distill_status."
            )
        return json.dumps(payload, indent=2)

    @tool
    def distill_status() -> str:
        """Poll the distillation train job (idle / queued / running / success)."""
        from backend.distillation.job import get_status

        return json.dumps(get_status().to_dict(), indent=2)

    @tool
    def cancel_distill_train() -> str:
        """Cancel a running or queued LoRA train job."""
        from backend.distillation.job import request_cancel

        return json.dumps({"status": request_cancel()})

    @tool
    def set_distilled_purpose(catalog_id: str, purpose: str) -> str:
        """Set or update the purpose on an already-trained adapter (no retrain).

        Args:
            catalog_id: otto-distill/… id from list_distilled_models.
            purpose: What this adapter should be used for.
        """
        from backend.distillation.catalog import set_adapter_purpose

        cid = (catalog_id or "").strip()
        text = (purpose or "").strip()
        if not cid:
            return "Error: catalog_id is required."
        if not text:
            return "Error: purpose is required."
        rec = set_adapter_purpose(cid, text)
        if rec is None:
            return f"Error: unknown distilled catalog id '{cid}'."
        return json.dumps(_public_row(rec, bound_id=_bound_id()), indent=2)

    @tool
    async def use_distilled_model(catalog_id: str, prompt: str) -> str:
        """Hand this task to a NEW session running a distilled LoRA (Standard MLX).

        Spawns a child session whose graph is built with the adapter — the same
        pattern as spawn_followup_session.  This session is unchanged.  Does not
        write global Settings.  Turbo/exo cannot attach adapters.

        The child does NOT inherit this chat history.  Pass a self-contained
        prompt.  Tell the user to follow the new session.

        Args:
            catalog_id: otto-distill/… id from list_distilled_models.
            prompt: First user message for the child.  Must be self-contained.
        """
        from backend.distillation.catalog import resolve_catalog_id
        from backend.session_dispatch import kick_off_message
        from backend.state import session_mgr

        cid = (catalog_id or "").strip()
        rec = resolve_catalog_id(cid)
        if rec is None:
            return (
                f"Error: unknown distilled catalog id '{cid}'. "
                "Call list_distilled_models."
            )
        if not prompt or not prompt.strip():
            return "Error: prompt must be a non-empty string."
        try:
            child = await session_mgr.spawn_child_session(
                parent_session_id=session_id,
                prompt=prompt,
                distill_catalog_id=cid,
            )
        except ValueError as exc:
            return f"Error: {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.exception("use_distilled_model failed")
            return f"Error: {exc}"
        try:
            await kick_off_message(child.id, prompt)
        except Exception as exc:  # noqa: BLE001
            logger.exception("kick_off_message failed for %s", child.id)
            return f"Error: child session created but failed to start: {exc}"
        return json.dumps({
            "child_session_id": child.id,
            "title": child.title,
            "agent_name": child.agent_name,
            "chain_depth": child.chain_depth,
            "parent_session_id": session_id,
            "catalog_id": cid,
            "display_name": rec.get("display_name") or "",
            "purpose": rec.get("purpose") or "",
            "note": "New session is running the distilled LoRA. Tell the user to follow it.",
        }, indent=2)

    @tool
    async def clear_distilled_model(prompt: str) -> str:
        """Hand off to a NEW session that uses global Settings (no LoRA).

        Use when this session is already on a distilled adapter and the next
        task should run the default model.  Same spawn pattern as
        spawn_followup_session.  This session is unchanged.

        Args:
            prompt: First user message for the child.  Must be self-contained.
        """
        from backend.session_dispatch import kick_off_message
        from backend.state import session_mgr

        if not prompt or not prompt.strip():
            return "Error: prompt must be a non-empty string."
        try:
            child = await session_mgr.spawn_child_session(
                parent_session_id=session_id,
                prompt=prompt,
                distill_catalog_id="",
            )
        except ValueError as exc:
            return f"Error: {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.exception("clear_distilled_model failed")
            return f"Error: {exc}"
        try:
            await kick_off_message(child.id, prompt)
        except Exception as exc:  # noqa: BLE001
            logger.exception("kick_off_message failed for %s", child.id)
            return f"Error: child session created but failed to start: {exc}"
        return json.dumps({
            "child_session_id": child.id,
            "title": child.title,
            "agent_name": child.agent_name,
            "chain_depth": child.chain_depth,
            "parent_session_id": session_id,
            "catalog_id": None,
            "note": "New session uses global Settings (no LoRA). Tell the user to follow it.",
        }, indent=2)

    return [
        list_distilled_models,
        distill_census,
        start_distill_train,
        distill_status,
        cancel_distill_train,
        set_distilled_purpose,
        use_distilled_model,
        clear_distilled_model,
    ]
