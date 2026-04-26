"""Browse sources with composable person, path-tag, and date filters."""

from __future__ import annotations

from urllib.parse import urlencode

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ..deps import get_db, get_templates
from ..helpers import group_by_month, normalize_source_type

router = APIRouter()


def _parse_subject_ids(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip().isdigit()]


def _parse_path_tags(raw: str) -> set[str]:
    return {part.strip() for part in raw.replace(",", " ").split() if part.strip()}


@router.get("/browse", response_class=HTMLResponse)
def browse_page(request: Request) -> HTMLResponse:
    return get_templates().TemplateResponse(
        name="browse.html",
        context={"kind": "people"},
        request=request,
    )


@router.get("/api/browse-html", response_class=HTMLResponse)
def browse_html(
    request: Request,
    subject_ids: str = "",
    path_tags: str = "",
    date_from: str = "",
    date_to: str = "",
    offset: int = 0,
    limit: int = 60,
    alone: bool = False,
    source_type: str = "photo",
) -> HTMLResponse:
    db = get_db()
    ids = _parse_subject_ids(subject_ids)
    tags = _parse_path_tags(path_tags)
    type_filter = normalize_source_type(source_type)
    from_filter = date_from.strip() or None
    to_filter = date_to.strip() or None
    # Browse depends on path-derived date/tag metadata. Existing archives may
    # have the table from migration but no rows yet, so make the endpoint
    # self-healing instead of returning a misleading fallback order/no tag hits.
    db.backfill_source_path_metadata(source_type=type_filter)
    total = db.count_search_sources(
        subject_ids=ids,
        path_tags=tags,
        date_from=from_filter,
        date_to=to_filter,
        source_type=type_filter,
        alone=alone,
    )
    sources = db.search_sources(
        subject_ids=ids,
        path_tags=tags,
        date_from=from_filter,
        date_to=to_filter,
        source_type=type_filter,
        alone=alone,
        limit=limit,
        offset=offset,
    )
    next_query = urlencode(
        {
            "subject_ids": subject_ids,
            "path_tags": path_tags,
            "date_from": date_from,
            "date_to": date_to,
            "source_type": source_type,
            "alone": str(alone).lower(),
            "offset": offset + len(sources),
            "limit": limit,
        }
    )
    return get_templates().TemplateResponse(
        name="partials/browse_results.html",
        context={
            "source_groups": group_by_month([(s, s.file_path) for s in sources], key="sources"),
            "total": total,
            "offset": offset,
            "limit": limit,
            "page_count": len(sources),
            "next_query": next_query,
        },
        request=request,
    )
