"""Jinja page routes for the canonical ``/{kind}/...`` scheme.

Included **last** in ``create_app`` — ``/{kind}/...`` here would shadow
specific ``/api/...`` endpoints (``CLAUDE.md:16`` + route-ordering test).
``/favicon.ico``, ``/api/export``, ``/api/singletons/faces-html`` are
grab-bag endpoints kept here rather than forced into an aggregate router
of their own.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ...cluster import (
    compare_subjects,
    find_similar_unclustered,
    rank_subjects_for_cluster,
)
from ...hints import compute_singleton_hints
from ..deps import get_db, get_templates
from ..helpers import (
    KindType,
    group_by_month,
    kind_for_species,
    species_for_kind,
)

router = APIRouter()


# ── Root + misc ────────────────────────────────────────────────────────


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    db = get_db()
    stats = db.get_stats(species="human")
    return get_templates().TemplateResponse(
        name="index.html",
        context={"stats": stats},
        request=request,
    )


@router.get("/favicon.ico")
def favicon() -> JSONResponse:
    raise HTTPException(404)


@router.get("/api/export")
def export_db() -> JSONResponse:
    data = json.loads(get_db().export_json())
    return JSONResponse(content=data)


# ── Cluster detail page (single-cluster view) ─────────────────────────


@router.get("/clusters/{cluster_id}", response_class=HTMLResponse)
def cluster_detail(
    request: Request, cluster_id: int, suggested_person: int | None = None
) -> HTMLResponse:
    db = get_db()
    total = db.get_cluster_finding_count(cluster_id)
    if total == 0:
        raise HTTPException(404, "Cluster not found")
    findings = db.get_cluster_findings(cluster_id, limit=200)
    sources = db.get_sources_batch([f.source_id for f in findings])
    face_paths = {
        f.id: sources[f.source_id].file_path if f.source_id in sources else "" for f in findings
    }
    ranked = rank_subjects_for_cluster(db, cluster_id)
    species = findings[0].species if findings else "human"
    kind = kind_for_species(species)
    return get_templates().TemplateResponse(
        name="cluster_detail.html",
        context={
            "cluster_id": cluster_id,
            "findings": findings,
            "finding_paths": face_paths,
            "total": total,
            "ranked_subjects": ranked,
            "kind": kind,
            "species": species,
        },
        request=request,
    )


# ── Singletons partial (paginated grid) ───────────────────────────────


@router.get("/api/singletons/faces-html", response_class=HTMLResponse)
def singletons_faces_html(
    request: Request, species: str = "human", offset: int = 0, limit: int = 200
) -> HTMLResponse:
    db = get_db()
    findings = db.get_singleton_findings(species=species, limit=limit, offset=offset)
    face_hints = compute_singleton_hints(db, findings, species)
    total = db.get_singleton_count(species=species)
    return get_templates().TemplateResponse(
        name="partials/singleton_grid.html",
        context={
            "findings": findings,
            "finding_hints": face_hints,
            "species": species,
            "offset": offset,
            "limit": limit,
            "total": total,
        },
        request=request,
    )


# ── Photo + search pages ──────────────────────────────────────────────


@router.get("/photos/{photo_id}", response_class=HTMLResponse)
def photo_page(request: Request, photo_id: int) -> HTMLResponse:
    db = get_db()
    source = db.get_source(photo_id)
    if not source:
        raise HTTPException(404, "Photo not found")
    findings = db.get_source_findings(photo_id)
    subjects = db.get_subjects()
    findings_data = []
    # Some older source rows were written with width/height=0 because the
    # original scan couldn't read dimensions. Skip overlay math rather
    # than 500-ing the page; thumbnails and the face grid still render.
    have_dims = source.width > 0 and source.height > 0
    for finding in findings:
        subject_name = None
        if finding.subject_id:
            s = db.get_subject(finding.subject_id)
            subject_name = s.name if s else None
        findings_data.append(
            {
                "id": finding.id,
                "bbox_x_pct": (finding.bbox_x / source.width * 100) if have_dims else None,
                "bbox_y_pct": (finding.bbox_y / source.height * 100) if have_dims else None,
                "bbox_w_pct": (finding.bbox_w / source.width * 100) if have_dims else None,
                "bbox_h_pct": (finding.bbox_h / source.height * 100) if have_dims else None,
                "subject_id": finding.subject_id,
                "subject_name": subject_name,
                "confidence": finding.confidence,
                "is_stranger": finding.exclusion_reason == "stranger",
            }
        )
    species = findings[0].species if findings else "human"
    kind = kind_for_species(species)
    return get_templates().TemplateResponse(
        name="photo.html",
        context={
            "source": source,
            "findings_data": findings_data,
            "subjects": subjects,
            "kind": kind,
        },
        request=request,
    )


@router.get("/search", response_class=HTMLResponse)
def search_page(request: Request, q: str = "") -> HTMLResponse:
    db = get_db()
    results = db.search_subjects(q) if q else []
    # Each search result redirects to /{plural-url-kind}/{subject_id}; pick
    # the plural URL kind from each subject's singular DB kind at the
    # HTTP-boundary. The conversion is a one-liner dict lookup — keeping
    # it inline avoids holding a helper that encodes singular in HTTP.
    result_kinds = {s.id: ("pets" if s.kind == "pet" else "people") for s in results}
    avatars = db.get_random_avatars([s.id for s in results])
    return get_templates().TemplateResponse(
        name="search.html",
        context={
            "query": q,
            "results": results,
            "result_kinds": result_kinds,
            "avatars": avatars,
            "kind": "people",
        },
        request=request,
    )


# ── Circles pages ─────────────────────────────────────────────────────


@router.get("/circles", response_class=HTMLResponse)
def circles_index(request: Request) -> HTMLResponse:
    db = get_db()
    return get_templates().TemplateResponse(
        name="circles.html",
        context={"circles": db.list_circles()},
        request=request,
    )


@router.get("/circles/{circle_id}", response_class=HTMLResponse)
def circle_detail(request: Request, circle_id: int) -> HTMLResponse:
    db = get_db()
    circle = db.get_circle(circle_id)
    if circle is None:
        raise HTTPException(404, "Circle not found")
    members = db.get_circle_members(circle_id)
    return get_templates().TemplateResponse(
        name="circle_detail.html",
        context={"circle": circle, "members": members},
        request=request,
    )


# ── Catch-all /{kind}/... pages (see module docstring) ───────────────


@router.get("/{kind}/clusters", response_class=HTMLResponse)
def clusters_page(request: Request, kind: KindType) -> HTMLResponse:
    db = get_db()
    species = species_for_kind(kind)
    clusters = db.get_unnamed_clusters(species=species)
    return get_templates().TemplateResponse(
        name="clusters.html",
        context={"clusters": clusters, "kind": kind},
        request=request,
    )


@router.get("/{kind}/singletons", response_class=HTMLResponse)
def singletons_page(request: Request, kind: KindType) -> HTMLResponse:
    db = get_db()
    species = species_for_kind(kind)
    total = db.get_singleton_count(species=species)
    findings = db.get_singleton_findings(species=species, limit=200)
    subjects = db.get_subjects_by_species(species)
    sources = db.get_sources_batch([f.source_id for f in findings])
    face_paths = {
        f.id: sources[f.source_id].file_path if f.source_id in sources else "" for f in findings
    }
    face_hints = compute_singleton_hints(db, findings, species)
    return get_templates().TemplateResponse(
        name="singletons.html",
        context={
            "findings": findings,
            "finding_paths": face_paths,
            "finding_hints": face_hints,
            "subjects": subjects,
            "total": total,
            "kind": kind,
        },
        request=request,
    )


@router.get("/{kind}/merge-suggestions", response_class=HTMLResponse)
def merge_suggestions_page(request: Request, kind: KindType, min_sim: float = 40.0) -> HTMLResponse:
    return get_templates().TemplateResponse(
        name="merge_suggestions.html",
        context={"min_sim": min_sim, "kind": kind},
        request=request,
    )


@router.get("/{kind}/compare", response_class=HTMLResponse)
def compare_page(
    request: Request,
    kind: KindType,
    a: int | None = None,
    b: int | None = None,
) -> HTMLResponse:
    db = get_db()
    species = species_for_kind(kind)
    subjects = db.get_subjects_by_species(species)
    result = None
    subject_a = None
    subject_b = None
    if a is not None and b is not None and a != b:
        result = compare_subjects(db, a, b)
        subject_a = db.get_subject(a)
        subject_b = db.get_subject(b)
    return get_templates().TemplateResponse(
        name="compare.html",
        context={
            "subjects": subjects,
            "subject_a": subject_a,
            "subject_b": subject_b,
            "result": result,
            "selected_a": a,
            "selected_b": b,
            "kind": kind,
        },
        request=request,
    )


@router.get("/{kind}/{subject_id}/find-similar", response_class=HTMLResponse)
def find_similar_page(
    request: Request, kind: KindType, subject_id: int, min_sim: float = 55.0
) -> HTMLResponse:
    db = get_db()
    subject = db.get_subject(subject_id)
    if not subject:
        raise HTTPException(404, "Subject not found")
    candidates = find_similar_unclustered(db, subject_id, min_similarity=min_sim / 100)
    return get_templates().TemplateResponse(
        name="find_similar.html",
        context={
            "subject": subject,
            "candidates": candidates,
            "min_sim": min_sim,
            "kind": kind,
        },
        request=request,
    )


@router.get("/{kind}/{subject_id}", response_class=HTMLResponse)
def subject_detail(request: Request, kind: KindType, subject_id: int) -> HTMLResponse:
    db = get_db()
    subject = db.get_subject(subject_id)
    if not subject:
        raise HTTPException(404, "Subject not found")
    findings_with_paths = db.get_subject_findings_with_paths(subject_id, limit=200)
    findings = [f for f, _ in findings_with_paths]
    finding_groups = group_by_month(findings_with_paths, key="findings")
    all_sources = db.get_subject_sources(subject_id)
    sources = [s for s in all_sources if s.type == "photo"]
    source_groups = group_by_month([(s, s.file_path) for s in sources], key="sources")
    # Videos tab: pair each video source with the subject's findings on it (for thumbnail + count).
    videos = db.get_subject_sources_with_findings(subject_id, source_type="video")
    video_groups = group_by_month([(entry, entry[0].file_path) for entry in videos], key="videos")
    all_subjects = db.get_subjects()
    subject_circles = db.get_subject_circles(subject_id)
    all_circles = db.list_circles()
    has_unclustered = db.has_unclustered_findings(species=species_for_kind(kind))
    return get_templates().TemplateResponse(
        name="subject_detail.html",
        context={
            "subject": subject,
            "findings": findings,
            "finding_groups": finding_groups,
            "sources": sources,
            "source_groups": source_groups,
            "videos": videos,
            "video_groups": video_groups,
            "all_subjects": all_subjects,
            "subject_circles": subject_circles,
            "all_circles": all_circles,
            "has_unclustered": has_unclustered,
            "kind": kind,
            "rename_url": f"/api/subjects/{subject_id}/rename",
            "findings_count": len(findings),
            "subject_id": subject_id,
            "offset": 0,
            "limit": 200,
            "total": subject.face_count,
        },
        request=request,
    )


@router.get("/{kind}", response_class=HTMLResponse)
def subjects_page(request: Request, kind: KindType) -> HTMLResponse:
    db = get_db()
    species = species_for_kind(kind)
    subjects = db.get_subjects_by_species(species)
    avatars = db.get_random_avatars([s.id for s in subjects])
    in_circle = {
        int(r[0])
        for r in db.conn.execute("SELECT DISTINCT subject_id FROM subject_circles").fetchall()
    }
    return get_templates().TemplateResponse(
        name="subjects.html",
        context={
            "subjects": subjects,
            "kind": kind,
            "avatars": avatars,
            "in_circle": in_circle,
        },
        request=request,
    )
