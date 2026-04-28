"""Print-selection worklist and ordered zip export."""

from __future__ import annotations

import io
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ...undo import RestorePrintSelectionPayload
from ..deps import get_db, get_templates, get_undo_store

router = APIRouter()

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_zip_name(position: int, file_path: str) -> str:
    path = Path(file_path)
    stem = _SAFE_NAME_RE.sub("_", path.stem).strip("._") or "photo"
    suffix = path.suffix.lower() or ".jpg"
    return f"{position:04d}_{stem}{suffix}"


def _payload() -> dict[str, object]:
    db = get_db()
    items = db.list_print_selection()
    return {
        "total": len(items),
        "source_ids": [item.source.id for item in items],
        "items": [
            {
                "source_id": item.source.id,
                "position": item.position,
                "file_path": item.source.file_path,
                "added_at": item.added_at,
            }
            for item in items
        ],
    }


@router.get("/print", response_class=HTMLResponse)
def print_page(request: Request) -> HTMLResponse:
    return get_templates().TemplateResponse(
        name="print_selection.html",
        context={"kind": "people", "items": get_db().list_print_selection()},
        request=request,
    )


@router.get("/api/print-selection")
def get_print_selection() -> JSONResponse:
    return JSONResponse(_payload())


@router.post("/api/print-selection/export")
def export_print_selection() -> StreamingResponse:
    db = get_db()
    items = db.list_print_selection()
    if not items:
        raise HTTPException(400, "Print selection is empty")

    resolved: list[tuple[int, str, Path]] = []
    missing: list[str] = []
    for position, item in enumerate(items, start=1):
        source = item.source
        if source.type != "photo":
            raise HTTPException(400, "Print selection contains a non-photo source")
        original = db.resolve_path(source.file_path)
        if not original.exists() or not original.is_file():
            missing.append(source.file_path)
            continue
        resolved.append((position, source.file_path, original))
    if missing:
        raise HTTPException(404, f"Missing original source file(s): {', '.join(missing[:5])}")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as archive:
        used_names: set[str] = set()
        for position, source_path, original in resolved:
            name = _safe_zip_name(position, source_path)
            if name in used_names:
                name = f"{position:04d}_{original.stem}_{original.stat().st_ino}{original.suffix}"
            used_names.add(name)
            archive.write(original, arcname=name)
    buf.seek(0)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="ritrova-print-selection-{stamp}.zip"'
        },
    )


@router.post("/api/print-selection/clear")
def clear_print_selection() -> JSONResponse:
    db = get_db()
    prior = db.get_print_selection_ids()
    db.clear_print_selection()
    token = get_undo_store().put(
        "Cleared print selection",
        RestorePrintSelectionPayload(source_ids=prior),
    )
    return JSONResponse(
        {
            "ok": True,
            "total": 0,
            "source_ids": [],
            "items": [],
            "undo_token": token,
            "message": "Cleared print selection",
        }
    )


@router.post("/api/print-selection/reorder")
def reorder_print_selection(source_ids: list[int] = Body(..., embed=True)) -> JSONResponse:
    get_db().reorder_print_selection(source_ids)
    payload = _payload()
    payload.update({"ok": True})
    return JSONResponse(payload)


@router.post("/api/print-selection/{source_id}")
def add_print_selection(source_id: int, undo: bool = False) -> JSONResponse:
    db = get_db()
    prior = db.get_print_selection_ids()
    try:
        position = db.add_to_print_selection(source_id)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    payload = _payload()
    if undo and source_id not in prior:
        token = get_undo_store().put(
            "Added photo to print selection",
            RestorePrintSelectionPayload(source_ids=prior),
        )
        payload.update({"undo_token": token, "message": "Added photo to print selection"})
    payload.update({"ok": True, "source_id": source_id, "position": position})
    return JSONResponse(payload)


@router.delete("/api/print-selection/{source_id}")
def remove_print_selection(source_id: int, undo: bool = False) -> JSONResponse:
    db = get_db()
    prior = db.get_print_selection_ids()
    db.remove_from_print_selection(source_id)
    payload = _payload()
    if undo and source_id in prior:
        token = get_undo_store().put(
            "Removed photo from print selection",
            RestorePrintSelectionPayload(source_ids=prior),
        )
        payload.update({"undo_token": token, "message": "Removed photo from print selection"})
    payload.update({"ok": True, "source_id": source_id})
    return JSONResponse(payload)
