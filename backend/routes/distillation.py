"""API for the distillation train job (start / poll / cancel)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.distillation.census import run_census
from backend.distillation.dataset import scan_dataset, session_detail
from backend.distillation.job import (
    DistillBusy,
    blocking_sessions,
    get_status,
    request_cancel,
    start_fuse_job,
    start_train_job,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/distillation", tags=["distillation"])


class TrainRequest(BaseModel):
    iters: int | None = Field(default=None, ge=1, le=50_000)
    name: str = "activity"
    purpose: str = ""
    teacher_model_id: str | None = None
    student_model_id: str | None = None
    source: str = "sessions"
    upload_id: str | None = None


class FuseRequest(BaseModel):
    catalog_id: str


class PurposeRequest(BaseModel):
    catalog_id: str
    purpose: str = ""


@router.get("/status")
async def distill_status() -> dict[str, Any]:
    return get_status().to_dict()


@router.get("/census")
async def distill_census(
    teacher_model_id: str | None = None,
    student_model_id: str | None = None,
    filter_sessions_by_teacher: bool | None = None,
) -> dict[str, Any]:
    report = await asyncio.to_thread(
        run_census,
        teacher_model_id=teacher_model_id,
        student_model_id=student_model_id,
        filter_sessions_by_teacher=filter_sessions_by_teacher,
    )
    report["blocking_sessions"] = blocking_sessions()
    return report


@router.post("/train")
async def distill_train(body: TrainRequest | None = None) -> JSONResponse:
    req = body or TrainRequest()
    upload_id = (req.upload_id or "").strip() or None
    if req.source == "upload" and not upload_id:
        return JSONResponse(
            status_code=400,
            content={"error": "Choose an uploaded JSONL file first.", "code": "missing_upload"},
        )
    try:
        status = await start_train_job(
            iters=req.iters,
            output_name=req.name if req.source != "upload" else (req.name or "upload"),
            purpose=req.purpose,
            teacher_id=req.teacher_model_id,
            student_id=req.student_model_id,
            upload_id=upload_id if req.source == "upload" else None,
        )
    except DistillBusy as exc:
        return JSONResponse(
            status_code=409,
            content={
                "error": str(exc),
                "code": exc.code,
                "blocking_sessions": exc.blocking,
            },
        )
    return JSONResponse(status_code=202, content=status.to_dict())


@router.post("/fuse")
async def distill_fuse(body: FuseRequest) -> JSONResponse:
    """Fuse a trained LoRA into a standalone MLX model oMLX can load."""
    try:
        status = await start_fuse_job(catalog_id=body.catalog_id)
    except ValueError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc), "code": "not_found"})
    except DistillBusy as exc:
        return JSONResponse(
            status_code=409,
            content={
                "error": str(exc),
                "code": exc.code,
                "blocking_sessions": exc.blocking,
            },
        )
    code = 200 if status.state == "success" else 202
    return JSONResponse(status_code=code, content=status.to_dict())


@router.post("/cancel")
async def distill_cancel() -> dict[str, str]:
    return {"status": request_cancel()}


@router.get("/adapters")
async def distill_adapters() -> dict[str, Any]:
    """List trained LoRA adapters for the Distilled picker tab."""
    from backend.distillation.catalog import iter_trained_adapters

    recs = await asyncio.to_thread(iter_trained_adapters)
    adapters = [
        {k: v for k, v in rec.items() if k != "meta"}
        for rec in recs
    ]
    return {"adapters": adapters}


@router.patch("/adapters/purpose")
async def distill_set_purpose(body: PurposeRequest) -> JSONResponse:
    """Update the catalog purpose on an already-trained adapter."""
    from backend.distillation.catalog import set_adapter_purpose

    rec = await asyncio.to_thread(set_adapter_purpose, body.catalog_id, body.purpose)
    if rec is None:
        return JSONResponse(status_code=404, content={"error": "Adapter not found", "code": "not_found"})
    return JSONResponse(content={k: v for k, v in rec.items() if k != "meta"})


@router.get("/dataset")
async def distill_dataset(
    teacher_model_id: str | None = None,
    student_model_id: str | None = None,
    filter_sessions_by_teacher: bool | None = None,
) -> dict[str, Any]:
    """Session-level training corpus with stats for the Distill data viewer."""
    return await asyncio.to_thread(
        scan_dataset,
        teacher_model_id=teacher_model_id,
        student_model_id=student_model_id,
        filter_sessions_by_teacher=filter_sessions_by_teacher,
    )


@router.get("/dataset/{session_id}")
async def distill_dataset_session(
    session_id: str,
    teacher_model_id: str | None = None,
    student_model_id: str | None = None,
    filter_sessions_by_teacher: bool | None = None,
) -> JSONResponse:
    """One trajectory plus truncated messages."""
    row = await asyncio.to_thread(
        session_detail,
        session_id,
        teacher_model_id=teacher_model_id,
        student_model_id=student_model_id,
        filter_sessions_by_teacher=filter_sessions_by_teacher,
    )
    if row is None:
        return JSONResponse(status_code=404, content={"error": "Session not found", "code": "not_found"})
    return JSONResponse(content=row)


@router.post("/validate")
async def distill_validate(file: UploadFile = File(...)) -> dict[str, Any]:
    """Check a JSONL (or JSON array) without saving it."""
    from backend.distillation.validate import validate_bytes

    raw = await file.read()
    return await asyncio.to_thread(
        validate_bytes, raw, filename=file.filename or "upload.jsonl",
    )


@router.get("/uploads")
async def distill_uploads() -> dict[str, Any]:
    from backend.distillation.uploads import list_uploads

    return {"uploads": await asyncio.to_thread(list_uploads)}


@router.post("/uploads")
async def distill_upload(file: UploadFile = File(...)) -> JSONResponse:
    from backend.distillation.uploads import save_upload

    raw = await file.read()
    result = await asyncio.to_thread(
        save_upload, raw, filename=file.filename or "upload.jsonl",
    )
    if not result.get("saved"):
        return JSONResponse(
            status_code=400,
            content={
                "error": "File is not a valid training dataset.",
                "code": "invalid_dataset",
                "report": result.get("report"),
            },
        )
    return JSONResponse(content=result)


@router.get("/uploads/{upload_id}")
async def distill_upload_get(upload_id: str) -> JSONResponse:
    from backend.distillation.uploads import get_upload, revalidate

    rec = await asyncio.to_thread(get_upload, upload_id)
    if rec is None:
        return JSONResponse(status_code=404, content={"error": "Upload not found", "code": "not_found"})
    checked = await asyncio.to_thread(revalidate, upload_id)
    return JSONResponse(content=checked or {"upload": rec})


@router.post("/uploads/{upload_id}/validate")
async def distill_upload_revalidate(upload_id: str) -> JSONResponse:
    from backend.distillation.uploads import revalidate

    checked = await asyncio.to_thread(revalidate, upload_id)
    if checked is None:
        return JSONResponse(status_code=404, content={"error": "Upload not found", "code": "not_found"})
    return JSONResponse(content=checked)


@router.delete("/uploads/{upload_id}")
async def distill_upload_delete(upload_id: str) -> JSONResponse:
    from backend.distillation.uploads import delete_upload

    ok = await asyncio.to_thread(delete_upload, upload_id)
    if not ok:
        return JSONResponse(status_code=404, content={"error": "Upload not found", "code": "not_found"})
    return JSONResponse(content={"deleted": upload_id})
