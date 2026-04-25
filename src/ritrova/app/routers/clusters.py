"""Cluster-level endpoints and the merge-suggestions data feed.

``/api/merge-suggestions`` and its ``-html`` partial live here because
the suggestions are cluster↔cluster pairings — logically the aggregate
is the cluster, not the subject.

Mutating endpoints forward to the domain-service layer
(``services_domain``) per ADR-012 §M3; routers translate
``SpeciesMismatch`` into the HTTP 409 contract.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ...cluster import find_similar_cluster, suggest_merges
from ...hints import compute_cluster_hint
from ...services_domain import SpeciesMismatch
from ..deps import (
    get_cluster_service,
    get_curation_service,
    get_db,
    get_subject_service,
    get_templates,
)
from ..helpers import kind_for_species, undo_hx_trigger

router = APIRouter()


# ── Next-cluster redirect helpers ──────────────────────────────────────


def _next_similar_cluster(subject_id: int, cluster_id: int) -> str:
    """Find the unnamed cluster most similar to this subject, return redirect URL."""
    db = get_db()
    findings = db.get_cluster_findings(cluster_id, limit=1)
    species = findings[0].species if findings else "human"
    kind = kind_for_species(species)
    fallback = f"/{kind}/clusters"

    next_cluster = find_similar_cluster(db, subject_id, species=species)
    if next_cluster:
        return f"/clusters/{next_cluster}?suggested_person={subject_id}"
    return fallback


def _next_unnamed_cluster_url(current_cluster_id: int, species: str) -> str:
    """URL of the next unnamed cluster of the same species (biggest first).
    Falls back to the species' cluster list when nothing's left."""
    db = get_db()
    kind = kind_for_species(species)
    for c in db.get_unnamed_clusters(species=species):
        if c["cluster_id"] != current_cluster_id:
            return f"/clusters/{c['cluster_id']}"
    return f"/{kind}/clusters"


# ── Merge-suggestions feed ─────────────────────────────────────────────


@router.get("/api/merge-suggestions")
def merge_suggestions_api(
    min_sim: float = 40.0,
    offset: int = 0,
    limit: int = 20,
    species: str = "human",
) -> JSONResponse:
    db = get_db()
    suggestions = suggest_merges(db, min_similarity=min_sim, species=species)
    subjects_map = {s.id: s.name for s in db.get_subjects()}
    page = suggestions[offset : offset + limit]
    return JSONResponse(
        {
            "total": len(suggestions),
            "offset": offset,
            "suggestions": [
                {
                    "cluster_a": s.cluster_a,
                    "cluster_b": s.cluster_b,
                    "similarity_pct": s.similarity_pct,
                    "size_a": s.size_a,
                    "size_b": s.size_b,
                    "sample_face_ids_a": s.sample_finding_ids_a,
                    "sample_face_ids_b": s.sample_finding_ids_b,
                    "name_a": subjects_map.get(s.cluster_a) if s.kind_a == "subject" else None,
                    "name_b": subjects_map.get(s.cluster_b) if s.kind_b == "subject" else None,
                }
                for s in page
            ],
        }
    )


@router.get("/api/merge-suggestions-html", response_class=HTMLResponse)
def merge_suggestions_html(
    request: Request,
    min_sim: float = 40.0,
    offset: int = 0,
    limit: int = 20,
    species: str = "human",
) -> HTMLResponse:
    db = get_db()
    templates = get_templates()
    suggestions = suggest_merges(db, min_similarity=min_sim, species=species)
    subjects_map = {s.id: s.name for s in db.get_subjects()}
    total = len(suggestions)
    page = suggestions[offset : offset + limit]
    items = [
        {
            "cluster_a": s.cluster_a,
            "cluster_b": s.cluster_b,
            "similarity_pct": s.similarity_pct,
            "size_a": s.size_a,
            "size_b": s.size_b,
            "sample_face_ids_a": s.sample_finding_ids_a,
            "sample_face_ids_b": s.sample_finding_ids_b,
            "name_a": subjects_map.get(s.cluster_a) if s.kind_a == "subject" else None,
            "name_b": subjects_map.get(s.cluster_b) if s.kind_b == "subject" else None,
        }
        for s in page
    ]
    return templates.TemplateResponse(
        name="partials/merge_page.html",
        context={
            "suggestions": items,
            "total": total,
            "offset": offset,
            "limit": limit,
            "min_sim": min_sim,
            "species": species,
            "kind": kind_for_species(species),
        },
        request=request,
    )


# ── Cluster hint / pagination ──────────────────────────────────────────


@router.get("/api/clusters/{cluster_id}/hint")
def cluster_hint_api(cluster_id: int) -> JSONResponse:
    """Return the best matching subject for a cluster."""
    hint = compute_cluster_hint(get_db(), cluster_id)
    if hint is None:
        return JSONResponse({"name": None})
    return JSONResponse(hint)


@router.get("/api/clusters/{cluster_id}/hint-html", response_class=HTMLResponse)
def cluster_hint_html(request: Request, cluster_id: int) -> HTMLResponse:
    """Return an HTML fragment with the assign button for the best hint."""
    hint = compute_cluster_hint(get_db(), cluster_id)
    return get_templates().TemplateResponse(
        name="partials/cluster_hint.html",
        context={"hint": hint, "cluster_id": cluster_id},
        request=request,
    )


@router.get("/api/clusters/{cluster_id}/faces")
def cluster_faces_api(cluster_id: int, offset: int = 0, limit: int = 200) -> JSONResponse:
    db = get_db()
    stubs = db.get_cluster_face_stubs(cluster_id, limit=limit, offset=offset)
    return JSONResponse([{"id": fid, "source_id": sid} for fid, sid in stubs])


@router.get("/api/clusters/{cluster_id}/faces-html", response_class=HTMLResponse)
def cluster_faces_html(
    request: Request, cluster_id: int, offset: int = 0, limit: int = 200
) -> HTMLResponse:
    db = get_db()
    stubs = db.get_cluster_face_stubs(cluster_id, limit=limit, offset=offset)
    faces = [{"id": fid, "source_id": sid} for fid, sid in stubs]
    return get_templates().TemplateResponse(
        name="partials/face_grid.html",
        context={
            "findings": faces,
            "cluster_id": cluster_id,
            "offset": offset,
            "limit": limit,
            "total": db.get_cluster_finding_count(cluster_id),
        },
        request=request,
    )


# ── Cluster mutations ──────────────────────────────────────────────────


@router.post("/api/clusters/{cluster_id}/name")
def name_cluster(cluster_id: int, name: str = Form(...)) -> RedirectResponse:
    db = get_db()
    # Derive species from the cluster's findings; the subject service
    # creates the row with the correct singular kind and registers the
    # undo (which deletes the subject on pop).
    findings = db.get_cluster_findings(cluster_id, limit=1)
    species = findings[0].species if findings else "human"
    pending_ids = db.get_unassigned_cluster_finding_ids(cluster_id)
    subject_id, _receipt = get_subject_service().create_subject_and_register_delete(
        name, species=species, n_findings=len(pending_ids)
    )
    db.assign_cluster_to_subject(cluster_id, subject_id)
    # Redirect path; the toast is recovered by /api/undo/peek on the
    # next page (same pattern as assign_cluster_to_existing's non-HX branch).
    return RedirectResponse(_next_similar_cluster(subject_id, cluster_id), status_code=303)


@router.post("/api/clusters/{cluster_id}/mark-stranger")
def mark_cluster_stranger(cluster_id: int) -> RedirectResponse:
    """Flag every uncurated finding in a cluster as a stranger.

    Writes ``exclusion_reason='stranger'`` on each uncurated finding and
    drops their cluster_findings rows (strangers don't re-cluster). No
    subject is created; the findings stay visible on their source photos
    but are hidden from clustering, merge-suggestions, auto-assign, and
    the curation queue. Reversible via the undo token.
    """
    db = get_db()
    findings = db.get_cluster_findings(cluster_id, limit=1)
    species = findings[0].species if findings else "human"
    get_curation_service().mark_cluster_stranger(cluster_id)
    return RedirectResponse(_next_unnamed_cluster_url(cluster_id, species), status_code=303)


@router.post("/api/clusters/{cluster_id}/assign", response_model=None)
def assign_cluster_to_existing(
    request: Request,
    cluster_id: int,
    subject_id: int = Form(...),
    force: bool = Form(False),
) -> RedirectResponse | HTMLResponse | JSONResponse:
    db = get_db()
    if not db.get_subject(subject_id):
        raise HTTPException(404, "Subject not found")
    try:
        receipt = get_cluster_service().assign_cluster(cluster_id, subject_id, force=force)
    except SpeciesMismatch as e:
        return JSONResponse({"error": str(e), "needs_confirm": True}, status_code=409)
    if request.headers.get("HX-Request"):
        return HTMLResponse("", headers=undo_hx_trigger(receipt.message, receipt.token))
    # Non-htmx form post: redirect to the next cluster. Toast is picked
    # up on the new page by a /api/undo/peek poll.
    return RedirectResponse(_next_similar_cluster(subject_id, cluster_id), status_code=303)


@router.post("/api/clusters/{cluster_id}/dismiss")
def dismiss_cluster(cluster_id: int) -> JSONResponse:
    """Dismiss all findings in a cluster as non-faces."""
    db = get_db()
    # Species must be read before dismiss strips cluster_findings rows.
    sample = db.get_cluster_findings(cluster_id, limit=1)
    species = sample[0].species if sample else "human"
    finding_ids = db.get_cluster_finding_ids(cluster_id)
    receipt = get_curation_service().dismiss_findings_as_cluster(cluster_id)
    if receipt is None:
        return JSONResponse({"ok": True, "dismissed": 0})
    return JSONResponse(
        {
            "ok": True,
            "dismissed": len(finding_ids),
            "undo_token": receipt.token,
            "message": receipt.message,
            "next_url": _next_unnamed_cluster_url(cluster_id, species),
        }
    )


@router.post("/api/clusters/merge", response_model=None)
def merge_clusters_api(
    request: Request,
    source_cluster: int = Form(...),
    target_cluster: int = Form(...),
) -> JSONResponse | HTMLResponse:
    """Move all findings from source cluster into target cluster."""
    receipt = get_cluster_service().merge_clusters(source_cluster, target_cluster)
    if request.headers.get("HX-Request"):
        return HTMLResponse("", headers=undo_hx_trigger(receipt.message, receipt.token))
    return JSONResponse({"ok": True, "undo_token": receipt.token, "message": receipt.message})


@router.post("/api/clusters/{cluster_id}/split")
def split_cluster_api(
    cluster_id: int,
    face_ids: list[int] = Body(..., embed=True),
) -> JSONResponse:
    """Move selected findings into a freshly-created cluster."""
    result = get_cluster_service().split_cluster(cluster_id, face_ids)
    if result is None:
        return JSONResponse({"ok": True, "split": 0})
    new_cluster_id, moved_count, receipt = result
    return JSONResponse(
        {
            "ok": True,
            "split": moved_count,
            "new_cluster_id": new_cluster_id,
            "new_url": f"/clusters/{new_cluster_id}",
            "undo_token": receipt.token,
            "message": receipt.message,
        }
    )
