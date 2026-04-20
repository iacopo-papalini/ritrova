"""Together: find sources containing all of a given set of subjects."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ..deps import get_db, get_templates
from ..helpers import group_by_month, normalize_source_type

router = APIRouter()


@router.get("/together", response_class=HTMLResponse)
def together_page(request: Request) -> HTMLResponse:
    return get_templates().TemplateResponse(
        name="together.html",
        context={"kind": "people"},
        request=request,
    )


@router.get("/api/together")
def together_api(
    subject_ids: str = "", alone: bool = False, source_type: str = "either"
) -> JSONResponse:
    """Find sources containing ALL given subject IDs (comma-separated)."""
    db = get_db()
    if not subject_ids.strip():
        return JSONResponse({"sources": [], "total": 0})
    ids = [int(x) for x in subject_ids.split(",") if x.strip().isdigit()]
    type_filter = normalize_source_type(source_type)
    sources = db.get_sources_with_all_subjects(ids, alone=alone, source_type=type_filter)
    return JSONResponse(
        {
            "total": len(sources),
            "sources": [
                {"id": s.id, "file_path": s.file_path, "taken_at": s.taken_at, "type": s.type}
                for s in sources[:200]
            ],
        }
    )


@router.get("/api/together-html", response_class=HTMLResponse)
def together_html(
    request: Request,
    subject_ids: str = "",
    offset: int = 0,
    limit: int = 60,
    alone: bool = False,
    source_type: str = "either",
) -> HTMLResponse:
    db = get_db()
    templates = get_templates()
    if not subject_ids.strip():
        return HTMLResponse("")
    ids = [int(x) for x in subject_ids.split(",") if x.strip().isdigit()]
    type_filter = normalize_source_type(source_type)
    total = db.count_sources_with_all_subjects(ids, alone=alone, source_type=type_filter)
    sources = db.get_sources_with_all_subjects(
        ids, limit=limit, offset=offset, alone=alone, source_type=type_filter
    )
    groups = group_by_month([(s, s.file_path) for s in sources], key="sources")
    return templates.TemplateResponse(
        name="partials/together_results.html",
        context={
            "source_groups": groups,
            "total": total,
            "subject_ids": subject_ids,
            "alone": alone,
            "source_type": source_type,
            "offset": offset,
            "limit": limit,
            "page_count": len(sources),
        },
        request=request,
    )
