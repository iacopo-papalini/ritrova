"""Cluster-level endpoints and the merge-suggestions data feed.

``/api/merge-suggestions`` and its ``-html`` partial live here because the
suggestions are cluster↔cluster pairings — logically the aggregate is the
cluster, not the subject.
"""

from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ...cluster import find_similar_cluster, suggest_merges
from ...services import compute_cluster_hint
from ...undo import (
    DeleteSubjectPayload,
    DismissPayload,
    FindingFieldsSnapshot,
    FindingPersonSnapshot,
    RestoreClusterPayload,
    RestoreFromStrangerPayload,
    RestorePersonIdsPayload,
)
from ..deps import get_db, get_templates, get_undo_store
from ..helpers import (
    describe_cluster_assign,
    describe_cluster_dismiss,
    describe_cluster_merge,
    describe_cluster_name,
    kind_for_species,
    undo_hx_trigger,
)

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
    rows = db.query(
        "SELECT f.id, f.source_id FROM findings f "
        "JOIN cluster_findings cf ON cf.finding_id = f.id "
        "WHERE cf.cluster_id = ? LIMIT ? OFFSET ?",
        (cluster_id, limit, offset),
    )
    return JSONResponse([{"id": r[0], "source_id": r[1]} for r in rows])


@router.get("/api/clusters/{cluster_id}/faces-html", response_class=HTMLResponse)
def cluster_faces_html(
    request: Request, cluster_id: int, offset: int = 0, limit: int = 200
) -> HTMLResponse:
    db = get_db()
    rows = db.query(
        "SELECT f.id, f.source_id FROM findings f "
        "JOIN cluster_findings cf ON cf.finding_id = f.id "
        "WHERE cf.cluster_id = ? LIMIT ? OFFSET ?",
        (cluster_id, limit, offset),
    )
    faces = [{"id": r[0], "source_id": r[1]} for r in rows]
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
    undo_store = get_undo_store()
    # Derive species from the cluster's findings; the DB knows how to turn
    # it into a subject (create_subject_for_species is the single boundary).
    findings = db.get_cluster_findings(cluster_id, limit=1)
    species = findings[0].species if findings else "human"
    # Snapshot exactly which findings assign_cluster_to_subject will touch
    # (WHERE subject_id IS NULL) so undo can NULL precisely those — not any
    # pre-existing assignments in the cluster.
    pending_ids = db.get_unassigned_cluster_finding_ids(cluster_id)
    subject_id = db.create_subject_for_species(name, species=species)
    db.assign_cluster_to_subject(cluster_id, subject_id)
    undo_store.put(
        description=describe_cluster_name(name, len(pending_ids)),
        payload=DeleteSubjectPayload(subject_id=subject_id),
    )
    # Redirect path; the toast is recovered by /api/undo/peek on the next
    # page (same pattern as assign_cluster_to_existing's non-HX branch).
    return RedirectResponse(_next_similar_cluster(subject_id, cluster_id), status_code=303)


@router.post("/api/clusters/{cluster_id}/mark-stranger")
def mark_cluster_stranger(cluster_id: int) -> RedirectResponse:
    """Flag every uncurated finding in a cluster as a stranger.

    Writes `exclusion_reason='stranger'` on each uncurated finding and
    drops their cluster_findings rows (strangers don't re-cluster).
    No subject is created; the findings stay visible on their source
    photos but are hidden from clustering, merge-suggestions, auto-
    assign, and the curation queue. Reversible from the photo page by
    assigning a name (which overwrites the exclusion row).
    """
    db = get_db()
    undo_store = get_undo_store()
    findings = db.get_cluster_findings(cluster_id, limit=1)
    species = findings[0].species if findings else "human"

    # Only touch findings that don't already carry a curation row.
    pending_ids = db.get_unassigned_cluster_finding_ids(cluster_id)
    if not pending_ids:
        return RedirectResponse(_next_unnamed_cluster_url(cluster_id, species), status_code=303)

    db.set_exclusions(pending_ids, "stranger")
    db.remove_cluster_memberships(pending_ids)
    noun = "face" if len(pending_ids) == 1 else "faces"
    message = f"Marked {len(pending_ids)} {noun} as stranger"
    undo_store.put(
        description=message,
        payload=RestoreFromStrangerPayload(cluster_id=cluster_id, finding_ids=pending_ids),
    )
    return RedirectResponse(_next_unnamed_cluster_url(cluster_id, species), status_code=303)


@router.post("/api/clusters/{cluster_id}/assign", response_model=None)
def assign_cluster_to_existing(
    request: Request,
    cluster_id: int,
    person_id: int = Form(...),
    force: bool = Form(False),
) -> RedirectResponse | HTMLResponse | JSONResponse:
    db = get_db()
    undo_store = get_undo_store()
    subject = db.get_subject(person_id)
    if not subject:
        raise HTTPException(404, "Subject not found")
    # Snapshot the findings that assign_cluster_to_subject will actually
    # mutate (it UPDATEs only where person_id IS NULL) so undo can flip
    # exactly those rows back to NULL.
    pending_ids = db.get_unassigned_cluster_finding_ids(cluster_id)
    try:
        db.assign_cluster_to_subject(cluster_id, person_id, force=force)
    except ValueError as e:
        return JSONResponse({"error": str(e), "needs_confirm": True}, status_code=409)
    token = undo_store.put(
        description=describe_cluster_assign(subject.name, len(pending_ids)),
        payload=RestorePersonIdsPayload(
            snapshots=[FindingPersonSnapshot(finding_id=fid, person_id=None) for fid in pending_ids]
        ),
    )
    message = describe_cluster_assign(subject.name, len(pending_ids))
    if request.headers.get("HX-Request"):
        return HTMLResponse("", headers=undo_hx_trigger(message, token))
    # Non-htmx form post: the user is being redirected to the next cluster.
    # The toast will be picked up on the new page by a /api/undo/peek poll.
    return RedirectResponse(_next_similar_cluster(person_id, cluster_id), status_code=303)


@router.post("/api/clusters/{cluster_id}/dismiss")
def dismiss_cluster(cluster_id: int) -> JSONResponse:
    """Dismiss all findings in a cluster as non-faces."""
    db = get_db()
    undo_store = get_undo_store()
    finding_ids = db.get_cluster_finding_ids(cluster_id)
    if not finding_ids:
        return JSONResponse({"ok": True, "dismissed": 0})
    # Species must be read before dismiss strips cluster_findings rows.
    sample = db.get_cluster_findings(cluster_id, limit=1)
    species = sample[0].species if sample else "human"
    # Snapshot person_id/cluster_id per finding so undo can both delete
    # the dismissed_findings rows and restore prior assignments.
    rows = db.snapshot_findings_fields(finding_ids)
    snapshots = [
        FindingFieldsSnapshot(finding_id=fid, person_id=pid, cluster_id=cid)
        for fid, pid, cid in rows
    ]
    db.dismiss_findings(finding_ids)
    message = describe_cluster_dismiss(cluster_id, len(finding_ids))
    token = undo_store.put(
        description=message,
        payload=DismissPayload(snapshots=snapshots),
    )
    return JSONResponse(
        {
            "ok": True,
            "dismissed": len(finding_ids),
            "undo_token": token,
            "message": message,
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
    db = get_db()
    undo_store = get_undo_store()
    # Snapshot the finding ids currently in the source cluster before the
    # merge rewrites their cluster_id — undo flips them back to source.
    moved_ids = db.get_cluster_finding_ids(source_cluster)
    db.merge_clusters(source_cluster, target_cluster)
    message = describe_cluster_merge(source_cluster, target_cluster, len(moved_ids))
    token = undo_store.put(
        description=message,
        payload=RestoreClusterPayload(cluster_id=source_cluster, finding_ids=moved_ids),
    )
    if request.headers.get("HX-Request"):
        return HTMLResponse("", headers=undo_hx_trigger(message, token))
    return JSONResponse({"ok": True, "undo_token": token, "message": message})
