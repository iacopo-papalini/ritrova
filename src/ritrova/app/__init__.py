"""Ritrova — web application for browsing, naming, and searching faces."""

import io
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from fastapi import Body, FastAPI, Form, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..cluster import (
    compare_subjects,
    find_similar_cluster,
    find_similar_unclustered,
    rank_subjects_for_cluster,
    suggest_merges,
)
from ..db import FaceDB
from ..images import crop_face_thumbnail, resize_photo
from ..services import (
    compute_cluster_hint,
    compute_singleton_hints,
)
from ..undo import (
    AddSubjectToCirclePayload,
    DeleteSubjectPayload,
    DismissPayload,
    FindingFieldsSnapshot,
    FindingPersonSnapshot,
    RecreateCirclePayload,
    RemoveSubjectFromCirclePayload,
    RestoreClusterPayload,
    RestoreFromStrangerBatchPayload,
    RestoreFromStrangerPayload,
    RestorePersonIdsPayload,
    ResurrectSubjectPayload,
    SubjectSnapshot,
    UndoStore,
)

KindType = Literal["people", "pets"]
# URL kind -> finding species (for finding-level filtering)
_KIND_TO_SPECIES: dict[str, str] = {"people": "human", "pets": "pet"}


def _species_for_kind(kind: KindType) -> str:
    """Map URL kind to finding species for finding-level queries."""
    return _KIND_TO_SPECIES[kind]


def _kind_for_species(species: str) -> KindType:
    """Map a finding species string to the URL kind."""
    if species in ("pet", "cat", "dog"):
        return "pets"
    return "people"


def _subject_kind_for_species(species: str) -> str:
    """Map a finding species to subject kind."""
    if species in ("pet", "cat", "dog", "other_pet"):
        return "pet"
    return "person"


def _kind_for_subject(subject_kind: str) -> KindType:
    """Map subject kind to URL kind."""
    return "pets" if subject_kind == "pet" else "people"


TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
STATIC_DIR = Path(__file__).parent.parent / "static"

_DATE_RE = re.compile(r"(\d{4})-(\d{2})")


def _month_from_path(file_path: str) -> str:
    """Extract YYYY-MM from directory names in a source path (most reliable date source)."""
    for part in reversed(Path(file_path).parts):
        m = _DATE_RE.search(part)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
    return "Unknown"


def _group_by_month(
    items: Sequence[tuple[object, str]], key: str = "items"
) -> list[dict[str, str | list[object]]]:
    """Group (item, file_path) pairs by month extracted from path."""
    groups: list[dict[str, str | list[object]]] = []
    current_month = None
    for item, path in items:
        month = _month_from_path(path)
        if month != current_month:
            current_month = month
            groups.append({"month": month, key: []})
        last = groups[-1][key]
        assert isinstance(last, list)
        last.append(item)
    return groups


def create_app(db_path: str, photos_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="Ritrova")
    db = FaceDB(db_path, base_dir=photos_dir)
    undo_store = UndoStore()
    # Expose on app.state so tests can clear/peek between cases.
    app.state.undo_store = undo_store

    thumbnails_dir = Path(db_path).parent / "tmp" / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ── Pages ──────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        stats = db.get_stats(species="human")
        return templates.TemplateResponse(
            name="index.html",
            context={"stats": stats},
            request=request,
        )

    @app.get("/clusters/{cluster_id}", response_class=HTMLResponse)
    def cluster_detail(
        request: Request, cluster_id: int, suggested_person: int | None = None
    ) -> HTMLResponse:
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
        kind = _kind_for_species(species)
        return templates.TemplateResponse(
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

    @app.get("/api/singletons/faces-html", response_class=HTMLResponse)
    def singletons_faces_html(
        request: Request, species: str = "human", offset: int = 0, limit: int = 200
    ) -> HTMLResponse:
        kind = _subject_kind_for_species(species)
        findings = db.get_singleton_findings(species=species, limit=limit, offset=offset)
        face_hints = compute_singleton_hints(db, findings, kind)
        total = db.get_singleton_count(species=species)
        return templates.TemplateResponse(
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

    @app.get("/photos/{photo_id}", response_class=HTMLResponse)
    def photo_page(request: Request, photo_id: int) -> HTMLResponse:
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
            if finding.person_id:
                s = db.get_subject(finding.person_id)
                subject_name = s.name if s else None
            findings_data.append(
                {
                    "id": finding.id,
                    "bbox_x_pct": (finding.bbox_x / source.width * 100) if have_dims else None,
                    "bbox_y_pct": (finding.bbox_y / source.height * 100) if have_dims else None,
                    "bbox_w_pct": (finding.bbox_w / source.width * 100) if have_dims else None,
                    "bbox_h_pct": (finding.bbox_h / source.height * 100) if have_dims else None,
                    "person_id": finding.person_id,
                    "person_name": subject_name,
                    "confidence": finding.confidence,
                    "is_stranger": finding.exclusion_reason == "stranger",
                }
            )
        species = findings[0].species if findings else "human"
        kind = _kind_for_species(species)
        return templates.TemplateResponse(
            name="photo.html",
            context={
                "source": source,
                "findings_data": findings_data,
                "subjects": subjects,
                "kind": kind,
            },
            request=request,
        )

    @app.get("/api/merge-suggestions")
    def merge_suggestions_api(
        min_sim: float = 40.0,
        offset: int = 0,
        limit: int = 20,
        species: str = "human",
    ) -> JSONResponse:
        kind = _subject_kind_for_species(species)
        suggestions = suggest_merges(db, min_similarity=min_sim, kind=kind)
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

    @app.get("/api/merge-suggestions-html", response_class=HTMLResponse)
    def merge_suggestions_html(
        request: Request,
        min_sim: float = 40.0,
        offset: int = 0,
        limit: int = 20,
        species: str = "human",
    ) -> HTMLResponse:
        kind = _subject_kind_for_species(species)
        suggestions = suggest_merges(db, min_similarity=min_sim, kind=kind)
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
                "kind": _kind_for_species(species),
            },
            request=request,
        )

    @app.post("/api/findings/swap")
    def swap_findings(
        face_ids: list[int] = Body(...),
        target_person_id: int = Body(...),
    ) -> JSONResponse:
        prior = db.snapshot_findings_fields(face_ids)
        snapshots = [
            FindingPersonSnapshot(finding_id=fid, person_id=pid) for fid, pid, _cid in prior
        ]
        for fid in face_ids:
            db.assign_finding_to_subject(fid, target_person_id)
        subject = db.get_subject(target_person_id)
        name = subject.name if subject else f"#{target_person_id}"
        message = _describe_findings_reassign("Swapped", name, len(face_ids))
        token = undo_store.put(
            description=message,
            payload=RestorePersonIdsPayload(snapshots=snapshots),
        )
        return JSONResponse(
            {"ok": True, "swapped": len(face_ids), "undo_token": token, "message": message}
        )

    @app.get("/search", response_class=HTMLResponse)
    def search_page(request: Request, q: str = "") -> HTMLResponse:
        results = db.search_subjects(q) if q else []
        result_kinds = {s.id: _kind_for_subject(s.kind) for s in results}
        avatars = db.get_random_avatars([s.id for s in results])
        return templates.TemplateResponse(
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

    @app.post("/api/subjects/{subject_id}/claim-faces")
    def claim_faces(
        subject_id: int,
        face_ids: list[int] = Body(..., embed=True),
        force: bool = Body(False, embed=True),
    ) -> JSONResponse:
        prior = db.snapshot_findings_fields(face_ids)
        snapshots = [
            FindingPersonSnapshot(finding_id=fid, person_id=pid) for fid, pid, _cid in prior
        ]
        try:
            for fid in face_ids:
                db.assign_finding_to_subject(fid, subject_id, force=force)
        except ValueError as e:
            return JSONResponse({"error": str(e), "needs_confirm": True}, status_code=409)
        subject = db.get_subject(subject_id)
        name = subject.name if subject else f"#{subject_id}"
        message = _describe_findings_reassign("Claimed", name, len(face_ids))
        token = undo_store.put(
            description=message,
            payload=RestorePersonIdsPayload(snapshots=snapshots),
        )
        return JSONResponse(
            {"ok": True, "claimed": len(face_ids), "undo_token": token, "message": message}
        )

    @app.get("/api/clusters/{cluster_id}/hint")
    def cluster_hint_api(cluster_id: int) -> JSONResponse:
        """Return the best matching subject for a cluster."""
        hint = compute_cluster_hint(db, cluster_id)
        if hint is None:
            return JSONResponse({"name": None})
        return JSONResponse(hint)

    @app.get("/api/clusters/{cluster_id}/hint-html", response_class=HTMLResponse)
    def cluster_hint_html(request: Request, cluster_id: int) -> HTMLResponse:
        """Return an HTML fragment with the assign button for the best hint."""
        hint = compute_cluster_hint(db, cluster_id)
        return templates.TemplateResponse(
            name="partials/cluster_hint.html",
            context={"hint": hint, "cluster_id": cluster_id},
            request=request,
        )

    # ── API: pagination ─────────────────────────────────────

    @app.get("/api/clusters/{cluster_id}/faces")
    def cluster_faces_api(cluster_id: int, offset: int = 0, limit: int = 200) -> JSONResponse:
        rows = db.query(
            "SELECT f.id, f.source_id FROM findings f "
            "JOIN cluster_findings cf ON cf.finding_id = f.id "
            "WHERE cf.cluster_id = ? LIMIT ? OFFSET ?",
            (cluster_id, limit, offset),
        )
        return JSONResponse([{"id": r[0], "source_id": r[1]} for r in rows])

    @app.get("/api/clusters/{cluster_id}/faces-html", response_class=HTMLResponse)
    def cluster_faces_html(
        request: Request, cluster_id: int, offset: int = 0, limit: int = 200
    ) -> HTMLResponse:
        rows = db.query(
            "SELECT f.id, f.source_id FROM findings f "
            "JOIN cluster_findings cf ON cf.finding_id = f.id "
            "WHERE cf.cluster_id = ? LIMIT ? OFFSET ?",
            (cluster_id, limit, offset),
        )
        faces = [{"id": r[0], "source_id": r[1]} for r in rows]
        return templates.TemplateResponse(
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

    @app.get("/api/subjects/{subject_id}/findings")
    def subject_faces_api(subject_id: int, offset: int = 0, limit: int = 200) -> JSONResponse:
        rows = db.query(
            "SELECT f.id, f.source_id FROM findings f "
            "JOIN finding_assignment fa ON fa.finding_id = f.id "
            "WHERE fa.subject_id = ? LIMIT ? OFFSET ?",
            (subject_id, limit, offset),
        )
        return JSONResponse([{"id": r[0], "source_id": r[1]} for r in rows])

    @app.get("/api/subjects/{subject_id}/findings-html", response_class=HTMLResponse)
    def subject_faces_html(
        request: Request, subject_id: int, offset: int = 0, limit: int = 200
    ) -> HTMLResponse:
        findings_with_paths = db.get_subject_findings_with_paths(
            subject_id, limit=limit, offset=offset
        )
        finding_groups = _group_by_month(findings_with_paths, key="findings")
        subject = db.get_subject(subject_id)
        total = subject.face_count if subject else 0
        return templates.TemplateResponse(
            name="partials/subject_finding_grid.html",
            context={
                "finding_groups": finding_groups,
                "subject_id": subject_id,
                "offset": offset,
                "limit": limit,
                "total": total,
                "findings_count": len(findings_with_paths),
            },
            request=request,
        )

    # ── API: images ────────────────────────────────────────
    # Images are content-addressed by immutable keys: finding_id → bbox →
    # source file, or source_id → file path. Source files are read-only per
    # design principle #5 ("Respect the archive"), so the rendered bytes for
    # a given URL never change. Tell the browser it can cache forever — this
    # makes re-navigating cluster/subject pages instant instead of refetching
    # hundreds of thumbnails on every visit.
    IMMUTABLE_IMAGE_HEADERS = {"Cache-Control": "public, max-age=31536000, immutable"}

    @app.get("/api/findings/{finding_id}/thumbnail")
    def finding_thumbnail(finding_id: int, size: int = 150) -> StreamingResponse:
        size = min(size, 500)
        cache_path = thumbnails_dir / f"{finding_id}_{size}.jpg"
        if cache_path.exists():
            return StreamingResponse(
                io.BytesIO(cache_path.read_bytes()),
                media_type="image/jpeg",
                headers=IMMUTABLE_IMAGE_HEADERS,
            )

        finding = db.get_finding(finding_id)
        if not finding:
            raise HTTPException(404)
        resolved = db.resolve_finding_image(finding)
        if resolved is None or not resolved.exists():
            raise HTTPException(404)

        buf = crop_face_thumbnail(
            resolved, finding.bbox_x, finding.bbox_y, finding.bbox_w, finding.bbox_h, size
        )
        cache_path.write_bytes(buf.getvalue())
        return StreamingResponse(buf, media_type="image/jpeg", headers=IMMUTABLE_IMAGE_HEADERS)

    @app.get("/api/sources/{source_id}/image")
    def source_image(source_id: int, max_size: int = 1600) -> StreamingResponse:
        source = db.get_source(source_id)
        if not source:
            raise HTTPException(404)
        if source.type == "video":
            # No raw image for videos; fall back to the first finding's extracted
            # frame so callers (Together grid, subject detail photo tab) can still
            # render a representative thumbnail. 404 only if truly no frames.
            findings = db.get_source_findings(source_id)
            frame_finding = next((f for f in findings if f.frame_path), None)
            if frame_finding is None:
                raise HTTPException(404, "No frames available for this video")
            resolved = db.resolve_finding_image(frame_finding)
            if resolved is None or not resolved.exists():
                raise HTTPException(404)
            buf = resize_photo(resolved, min(max_size, 2000))
            return StreamingResponse(buf, media_type="image/jpeg", headers=IMMUTABLE_IMAGE_HEADERS)
        resolved = db.resolve_path(source.file_path)
        if not resolved.exists():
            raise HTTPException(404)
        buf = resize_photo(resolved, min(max_size, 2000))
        return StreamingResponse(buf, media_type="image/jpeg", headers=IMMUTABLE_IMAGE_HEADERS)

    @app.get("/api/sources/{source_id}/original")
    def source_original(source_id: int) -> FileResponse:
        """Serve the unmodified original file from disk with a download disposition."""
        source = db.get_source(source_id)
        if not source:
            raise HTTPException(404)
        resolved = db.resolve_path(source.file_path)
        if not resolved.exists() or not resolved.is_file():
            raise HTTPException(404)
        return FileResponse(
            resolved,
            filename=resolved.name,
            # FastAPI/Starlette infers Content-Type from the path; explicit None
            # would force application/octet-stream. Let it auto-detect.
            headers=IMMUTABLE_IMAGE_HEADERS,
        )

    @app.get("/api/sources/{source_id}/info")
    def source_info(source_id: int) -> JSONResponse:
        source = db.get_source(source_id)
        if not source:
            raise HTTPException(404)
        return JSONResponse(
            {
                "file_path": str(db.resolve_path(source.file_path)),
                "latitude": source.latitude,
                "longitude": source.longitude,
                "type": source.type,
            }
        )

    @app.get("/api/findings/{finding_id}/frame")
    def finding_frame(finding_id: int, max_size: int = 1600) -> StreamingResponse:
        """Serve the image for a finding — the extracted frame for video findings,
        or the source image for photo findings. `resolve_finding_image` dispatches.
        """
        finding = db.get_finding(finding_id)
        if not finding:
            raise HTTPException(404)
        resolved = db.resolve_finding_image(finding)
        if resolved is None or not resolved.exists():
            raise HTTPException(404)
        buf = resize_photo(resolved, min(max_size, 2000))
        return StreamingResponse(buf, media_type="image/jpeg", headers=IMMUTABLE_IMAGE_HEADERS)

    @app.get("/api/findings/{finding_id}/info")
    def finding_info(finding_id: int) -> JSONResponse:
        """Metadata for the finding's underlying source. `file_path` is always the
        source's path (not the frame path) so the lightbox shows the real filename.
        """
        finding = db.get_finding(finding_id)
        if not finding:
            raise HTTPException(404)
        source = db.get_source(finding.source_id)
        if not source:
            raise HTTPException(404)
        return JSONResponse(
            {
                "source_id": source.id,
                "file_path": str(db.resolve_path(source.file_path)),
                "latitude": source.latitude,
                "longitude": source.longitude,
                "type": source.type,
            }
        )

    # ── API: actions ───────────────────────────────────────

    def _undo_hx_trigger(message: str, token: str) -> dict[str, str]:
        """Build the HX-Trigger header that fires the client-side undo toast."""
        payload = {"undoToast": {"message": message, "token": token}}
        return {"HX-Trigger": json.dumps(payload)}

    def _describe_cluster_dismiss(cluster_id: int, n: int) -> str:
        noun = "face" if n == 1 else "faces"
        return f"Dismissed {n} {noun} in cluster #{cluster_id}"

    def _describe_cluster_merge(source_id: int, target_id: int, n: int) -> str:
        noun = "face" if n == 1 else "faces"
        return f"Merged cluster #{source_id} into #{target_id} ({n} {noun})"

    def _describe_cluster_assign(subject_name: str, n: int) -> str:
        noun = "face" if n == 1 else "faces"
        return f"Assigned {n} {noun} to {subject_name}"

    def _describe_cluster_name(subject_name: str, n: int) -> str:
        noun = "face" if n == 1 else "faces"
        return f"Created {subject_name} and assigned {n} {noun}"

    def _describe_subject_delete(subject_name: str, n: int) -> str:
        noun = "face" if n == 1 else "faces"
        return f"Deleted {subject_name} ({n} {noun} unassigned)"

    def _describe_subject_merge(source_name: str, target_name: str, n: int) -> str:
        noun = "face" if n == 1 else "faces"
        return f"Merged {source_name} into {target_name} ({n} {noun})"

    def _describe_findings_dismiss(n: int) -> str:
        noun = "face" if n == 1 else "faces"
        return f"Dismissed {n} {noun}"

    def _describe_findings_exclude(cluster_id: int, n: int) -> str:
        noun = "face" if n == 1 else "faces"
        return f"Excluded {n} {noun} from cluster #{cluster_id}"

    def _describe_findings_reassign(verb: str, subject_name: str, n: int) -> str:
        noun = "face" if n == 1 else "faces"
        return f"{verb} {n} {noun} for {subject_name}"

    def _next_similar_cluster(subject_id: int, cluster_id: int) -> str:
        """Find the unnamed cluster most similar to this subject, return redirect URL."""
        findings = db.get_cluster_findings(cluster_id, limit=1)
        species = findings[0].species if findings else "human"
        kind = _kind_for_species(species)
        fallback = f"/{kind}/clusters"

        subject_kind = _subject_kind_for_species(species)
        next_cluster = find_similar_cluster(db, subject_id, kind=subject_kind)
        if next_cluster:
            return f"/clusters/{next_cluster}?suggested_person={subject_id}"
        return fallback

    def _next_unnamed_cluster_url(current_cluster_id: int, species: str) -> str:
        """URL of the next unnamed cluster of the same species (biggest first).
        Falls back to the species' cluster list when nothing's left."""
        kind = _kind_for_species(species)
        for c in db.get_unnamed_clusters(species=species):
            if c["cluster_id"] != current_cluster_id:
                return f"/clusters/{c['cluster_id']}"
        return f"/{kind}/clusters"

    @app.post("/api/clusters/{cluster_id}/name")
    def name_cluster(cluster_id: int, name: str = Form(...)) -> RedirectResponse:
        # Determine subject kind from cluster's finding species
        findings = db.get_cluster_findings(cluster_id, limit=1)
        subject_kind = _subject_kind_for_species(findings[0].species) if findings else "person"
        # Snapshot exactly which findings assign_cluster_to_subject will touch
        # (WHERE person_id IS NULL) so undo can NULL precisely those — not any
        # pre-existing assignments in the cluster.
        pending_ids = db.get_unassigned_cluster_finding_ids(cluster_id)
        subject_id = db.create_subject(name, kind=subject_kind)
        db.assign_cluster_to_subject(cluster_id, subject_id)
        undo_store.put(
            description=_describe_cluster_name(name, len(pending_ids)),
            payload=DeleteSubjectPayload(subject_id=subject_id),
        )
        # Redirect path; the toast is recovered by /api/undo/peek on the next
        # page (same pattern as assign_cluster_to_existing's non-HX branch).
        return RedirectResponse(_next_similar_cluster(subject_id, cluster_id), status_code=303)

    @app.post("/api/clusters/{cluster_id}/mark-stranger")
    def mark_cluster_stranger(cluster_id: int) -> RedirectResponse:
        """Flag every uncurated finding in a cluster as a stranger.

        Writes `exclusion_reason='stranger'` on each uncurated finding and
        drops their cluster_findings rows (strangers don't re-cluster).
        No subject is created; the findings stay visible on their source
        photos but are hidden from clustering, merge-suggestions, auto-
        assign, and the curation queue. Reversible from the photo page by
        assigning a name (which overwrites the exclusion row).
        """
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

    @app.post("/api/clusters/{cluster_id}/assign", response_model=None)
    def assign_cluster_to_existing(
        request: Request,
        cluster_id: int,
        person_id: int = Form(...),
        force: bool = Form(False),
    ) -> RedirectResponse | HTMLResponse | JSONResponse:
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
            description=_describe_cluster_assign(subject.name, len(pending_ids)),
            payload=RestorePersonIdsPayload(
                snapshots=[
                    FindingPersonSnapshot(finding_id=fid, person_id=None) for fid in pending_ids
                ]
            ),
        )
        message = _describe_cluster_assign(subject.name, len(pending_ids))
        if request.headers.get("HX-Request"):
            return HTMLResponse("", headers=_undo_hx_trigger(message, token))
        # Non-htmx form post: the user is being redirected to the next cluster.
        # The toast will be picked up on the new page by a /api/undo/peek poll.
        return RedirectResponse(_next_similar_cluster(person_id, cluster_id), status_code=303)

    @app.post("/api/clusters/{cluster_id}/dismiss")
    def dismiss_cluster(cluster_id: int) -> JSONResponse:
        """Dismiss all findings in a cluster as non-faces."""
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
        message = _describe_cluster_dismiss(cluster_id, len(finding_ids))
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

    @app.post("/api/clusters/merge", response_model=None)
    def merge_clusters_api(
        request: Request,
        source_cluster: int = Form(...),
        target_cluster: int = Form(...),
    ) -> JSONResponse | HTMLResponse:
        """Move all findings from source cluster into target cluster."""
        # Snapshot the finding ids currently in the source cluster before the
        # merge rewrites their cluster_id — undo flips them back to source.
        moved_ids = db.get_cluster_finding_ids(source_cluster)
        db.merge_clusters(source_cluster, target_cluster)
        message = _describe_cluster_merge(source_cluster, target_cluster, len(moved_ids))
        token = undo_store.put(
            description=message,
            payload=RestoreClusterPayload(cluster_id=source_cluster, finding_ids=moved_ids),
        )
        if request.headers.get("HX-Request"):
            return HTMLResponse("", headers=_undo_hx_trigger(message, token))
        return JSONResponse({"ok": True, "undo_token": token, "message": message})

    # ── FEAT-5: undo ───────────────────────────────────────

    @app.get("/api/undo/peek")
    def peek_undo() -> JSONResponse:
        """Return the currently pending undo, if any. Used by the client on
        page load (e.g. after a redirect) to restore the toast."""
        entry = undo_store.peek()
        if entry is None:
            return JSONResponse({"pending": False})
        return JSONResponse({"pending": True, "token": entry.token, "message": entry.description})

    @app.post("/api/undo/{token}")
    def apply_undo(token: str) -> JSONResponse:
        """Consume the pending undo matching ``token`` and invert its effect.

        Returns 404 if the token is missing, already consumed, or past its TTL.
        Single-shot: a successful undo clears the slot.
        """
        entry = undo_store.pop(token)
        if entry is None:
            raise HTTPException(404, "Undo token missing, already used, or expired")
        entry.payload.undo(db)
        return JSONResponse({"ok": True, "undone": entry.description})

    @app.post("/api/findings/mark-stranger")
    def mark_findings_stranger(face_ids: list[int] = Body(..., embed=True)) -> JSONResponse:
        """Flag an ad-hoc batch of findings as strangers.

        Writes `exclusion_reason='stranger'` on each one (overwriting any
        prior uncurated state) and drops their cluster_findings rows so
        they vanish from clustering / merge-suggestions / auto-assign.
        Reversible: prior cluster membership is snapshotted and restored
        on undo. Findings that had no cluster (singletons) return to the
        uncurated-unclustered state."""
        if not face_ids:
            return JSONResponse({"ok": True, "marked": 0})
        rows = db.snapshot_findings_fields(face_ids)
        snapshots = [
            FindingFieldsSnapshot(finding_id=fid, person_id=pid, cluster_id=cid)
            for fid, pid, cid in rows
        ]
        db.set_exclusions(face_ids, "stranger")
        db.remove_cluster_memberships(face_ids)
        n = len(face_ids)
        noun = "face" if n == 1 else "faces"
        message = f"Marked {n} {noun} as stranger"
        token = undo_store.put(
            description=message,
            payload=RestoreFromStrangerBatchPayload(snapshots=snapshots),
        )
        return JSONResponse({"ok": True, "marked": n, "undo_token": token, "message": message})

    @app.post("/api/findings/dismiss")
    def dismiss_findings(face_ids: list[int] = Body(..., embed=True)) -> JSONResponse:
        """Mark findings as non-faces (statues, paintings, dogs, etc.)."""
        if not face_ids:
            return JSONResponse({"ok": True, "dismissed": 0})
        rows = db.snapshot_findings_fields(face_ids)
        snapshots = [
            FindingFieldsSnapshot(finding_id=fid, person_id=pid, cluster_id=cid)
            for fid, pid, cid in rows
        ]
        db.dismiss_findings(face_ids)
        message = _describe_findings_dismiss(len(face_ids))
        token = undo_store.put(
            description=message,
            payload=DismissPayload(snapshots=snapshots),
        )
        return JSONResponse(
            {"ok": True, "dismissed": len(face_ids), "undo_token": token, "message": message}
        )

    @app.post("/api/findings/exclude")
    def exclude_findings(
        face_ids: list[int] = Body(..., embed=True),
        cluster_id: int = Body(..., embed=True),
    ) -> JSONResponse:
        if not face_ids:
            return JSONResponse({"ok": True, "excluded": 0})
        db.exclude_findings(face_ids, cluster_id=cluster_id)
        message = _describe_findings_exclude(cluster_id, len(face_ids))
        token = undo_store.put(
            description=message,
            payload=RestoreClusterPayload(cluster_id=cluster_id, finding_ids=face_ids),
        )
        return JSONResponse(
            {"ok": True, "excluded": len(face_ids), "undo_token": token, "message": message}
        )

    @app.post("/api/findings/{finding_id}/assign")
    def assign_finding(
        finding_id: int, person_id: int = Form(...), force: bool = Form(False)
    ) -> JSONResponse:
        try:
            db.assign_finding_to_subject(finding_id, person_id, force=force)
        except ValueError as e:
            return JSONResponse({"error": str(e), "needs_confirm": True}, status_code=409)
        return JSONResponse({"ok": True})

    @app.post("/api/findings/unassign")
    def unassign_findings_batch(face_ids: list[int] = Body(..., embed=True)) -> JSONResponse:
        """Unassign multiple findings from their current subjects in one call."""
        if not face_ids:
            return JSONResponse({"ok": True, "unassigned": 0})
        prior = db.snapshot_findings_fields(face_ids)
        snapshots = [
            FindingPersonSnapshot(finding_id=fid, person_id=pid)
            for fid, pid, _cid in prior
            if pid is not None
        ]
        db.unassign_findings(face_ids)
        n = len(face_ids)
        message = f"Removed {n} face{'s' if n != 1 else ''}"
        token = undo_store.put(
            description=message,
            payload=RestorePersonIdsPayload(snapshots=snapshots),
        )
        return JSONResponse({"ok": True, "unassigned": n, "undo_token": token, "message": message})

    @app.post("/api/findings/{finding_id}/unassign")
    def unassign_finding(finding_id: int) -> JSONResponse:
        prior_person_id = db.get_finding_person_id(finding_id)
        db.unassign_finding(finding_id)
        if prior_person_id is not None:
            subject = db.get_subject(prior_person_id)
            name = subject.name if subject else f"#{prior_person_id}"
            message = f"Removed face from {name}"
            token = undo_store.put(
                description=message,
                payload=RestorePersonIdsPayload(
                    snapshots=[
                        FindingPersonSnapshot(finding_id=finding_id, person_id=prior_person_id)
                    ]
                ),
            )
            return JSONResponse({"ok": True, "undo_token": token, "message": message})
        return JSONResponse({"ok": True})

    @app.post("/api/subjects/{subject_id}/rename")
    def rename_subject(subject_id: int, name: str = Form(...)) -> RedirectResponse:
        subject = db.get_subject(subject_id)
        if not subject:
            raise HTTPException(404, "Subject not found")
        kind = _kind_for_subject(subject.kind)
        db.rename_subject(subject_id, name)
        return RedirectResponse(f"/{kind}/{subject_id}", status_code=303)

    @app.post("/api/subjects/merge")
    def merge_subjects(source_id: int = Form(...), target_id: int = Form(...)) -> RedirectResponse:
        if source_id == target_id:
            raise HTTPException(400, "Cannot merge subject with themselves")
        target = db.get_subject(target_id)
        if not target:
            raise HTTPException(404, "Target subject not found")
        # Snapshot the source subject row and all its findings BEFORE merge
        # destroys both. merge_subjects flips person_id source->target on
        # every finding, then DELETEs the source row.
        source_row = db.get_subject_row(source_id)
        if not source_row:
            raise HTTPException(404, "Source subject not found")
        moved_ids = db.get_subject_finding_ids(source_id)
        source_snapshot = SubjectSnapshot(
            id=source_row[0], name=source_row[1], kind=source_row[2], created_at=source_row[3]
        )
        kind = _kind_for_subject(target.kind)
        db.merge_subjects(source_id, target_id)
        undo_store.put(
            description=_describe_subject_merge(source_snapshot.name, target.name, len(moved_ids)),
            payload=ResurrectSubjectPayload(subject=source_snapshot, finding_ids=moved_ids),
        )
        return RedirectResponse(f"/{kind}/{target_id}", status_code=303)

    @app.post("/api/subjects/{subject_id}/delete")
    def delete_subject(subject_id: int) -> RedirectResponse:
        """Unassign all findings and delete the subject."""
        subject = db.get_subject(subject_id)
        if not subject:
            raise HTTPException(404, "Subject not found")
        # Snapshot the full row + every assigned finding_id BEFORE delete
        # destroys the row and NULLs the person_ids.
        row = db.get_subject_row(subject_id)
        assert row is not None  # get_subject hit means the row exists
        finding_ids = db.get_subject_finding_ids(subject_id)
        snapshot = SubjectSnapshot(id=row[0], name=row[1], kind=row[2], created_at=row[3])
        kind = _kind_for_subject(subject.kind)
        db.delete_subject(subject_id)
        undo_store.put(
            description=_describe_subject_delete(snapshot.name, len(finding_ids)),
            payload=ResurrectSubjectPayload(subject=snapshot, finding_ids=finding_ids),
        )
        return RedirectResponse(f"/{kind}", status_code=303)

    @app.post("/api/subjects/create")
    def create_subject_api(
        name: str = Body(..., embed=True),
        kind: str = Body("person", embed=True),
    ) -> JSONResponse:
        """Create a subject and return its data. Used by typeahead picker."""
        subject_id = db.create_subject(name, kind=kind)
        subject = db.get_subject(subject_id)
        assert subject is not None
        return JSONResponse(
            {
                "id": subject.id,
                "name": subject.name,
                "kind": subject.kind,
                "face_count": subject.face_count,
            }
        )

    @app.get("/api/subjects/all")
    def all_subjects_api() -> JSONResponse:
        """All subjects (people and pets) for typeahead components, with avatar face ID."""
        subjects = db.get_subjects()
        avatars = db.get_random_avatars([s.id for s in subjects])
        return JSONResponse(
            [
                {
                    "id": s.id,
                    "name": s.name,
                    "kind": s.kind,
                    "face_count": s.face_count,
                    "face_id": avatars.get(s.id),
                }
                for s in subjects
            ]
        )

    @app.get("/together", response_class=HTMLResponse)
    def together_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            name="together.html",
            context={"kind": "people"},
            request=request,
        )

    @app.get("/favicon.ico")
    def favicon() -> JSONResponse:
        raise HTTPException(404)

    def _normalize_source_type(raw: str) -> str | None:
        """Accept 'photo' / 'video' as filters; anything else (default 'either') → None."""
        return raw if raw in ("photo", "video") else None

    @app.get("/api/together")
    def together_api(
        person_ids: str = "", alone: bool = False, source_type: str = "either"
    ) -> JSONResponse:
        """Find sources containing ALL given subject IDs (comma-separated)."""
        if not person_ids.strip():
            return JSONResponse({"sources": [], "total": 0})
        ids = [int(x) for x in person_ids.split(",") if x.strip().isdigit()]
        type_filter = _normalize_source_type(source_type)
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

    @app.get("/api/together-html", response_class=HTMLResponse)
    def together_html(
        request: Request,
        person_ids: str = "",
        offset: int = 0,
        limit: int = 60,
        alone: bool = False,
        source_type: str = "either",
    ) -> HTMLResponse:
        if not person_ids.strip():
            return HTMLResponse("")
        ids = [int(x) for x in person_ids.split(",") if x.strip().isdigit()]
        type_filter = _normalize_source_type(source_type)
        total = db.count_sources_with_all_subjects(ids, alone=alone, source_type=type_filter)
        sources = db.get_sources_with_all_subjects(
            ids, limit=limit, offset=offset, alone=alone, source_type=type_filter
        )
        groups = _group_by_month([(s, s.file_path) for s in sources], key="sources")
        return templates.TemplateResponse(
            name="partials/together_results.html",
            context={
                "source_groups": groups,
                "total": total,
                "person_ids": person_ids,
                "alone": alone,
                "source_type": source_type,
                "offset": offset,
                "limit": limit,
                "page_count": len(sources),
            },
            request=request,
        )

    @app.get("/api/export")
    def export_db() -> JSONResponse:
        data = json.loads(db.export_json())
        return JSONResponse(content=data)

    # ── Circles (FEAT-27): admin + membership mutations ──────────────────

    @app.get("/circles", response_class=HTMLResponse)
    def circles_index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            name="circles.html",
            context={"circles": db.list_circles()},
            request=request,
        )

    @app.get("/circles/{circle_id}", response_class=HTMLResponse)
    def circle_detail(request: Request, circle_id: int) -> HTMLResponse:
        circle = db.get_circle(circle_id)
        if circle is None:
            raise HTTPException(404, "Circle not found")
        members = db.get_circle_members(circle_id)
        return templates.TemplateResponse(
            name="circle_detail.html",
            context={"circle": circle, "members": members},
            request=request,
        )

    @app.post("/api/circles/create")
    def create_circle_api(
        name: str = Form(...),
        description: str = Form(""),
    ) -> RedirectResponse:
        circle_id = db.create_circle(name, description=description.strip() or None)
        # No undo for create — delete is the obvious reverse and user has that button.
        return RedirectResponse(f"/circles/{circle_id}", status_code=303)

    @app.post("/api/circles/{circle_id}/rename", response_model=None)
    def rename_circle_api(
        request: Request, circle_id: int, name: str = Form(...)
    ) -> RedirectResponse | JSONResponse:
        db.rename_circle(circle_id, name)
        if request.headers.get("hx-request") or request.headers.get("accept", "").startswith(
            "application/json"
        ):
            return JSONResponse({"ok": True, "name": name.strip()})
        return RedirectResponse(f"/circles/{circle_id}", status_code=303)

    @app.post("/api/circles/{circle_id}/delete", response_model=None)
    def delete_circle_api(request: Request, circle_id: int) -> JSONResponse | RedirectResponse:
        circle = db.get_circle(circle_id)
        if circle is None:
            raise HTTPException(404, "Circle not found")
        member_ids = db.get_circle_subject_ids(circle_id)
        db.delete_circle(circle_id)
        token = undo_store.put(
            description=f"Deleted circle '{circle.name}' ({len(member_ids)} members)",
            payload=RecreateCirclePayload(
                name=circle.name,
                description=circle.description,
                member_subject_ids=member_ids,
            ),
        )
        if request.headers.get("HX-Request"):
            return JSONResponse({"ok": True, "undo_token": token})
        return RedirectResponse("/circles", status_code=303)

    @app.post("/api/subjects/{subject_id}/circles/{circle_id}/add")
    def add_subject_to_circle_api(
        request: Request, subject_id: int, circle_id: int
    ) -> JSONResponse:
        subject = db.get_subject(subject_id)
        circle = db.get_circle(circle_id)
        if subject is None or circle is None:
            raise HTTPException(404)
        added = db.add_subject_to_circle(subject_id, circle_id)
        if not added:
            return JSONResponse({"ok": True, "already": True})
        token = undo_store.put(
            description=f"Added {subject.name} to {circle.name}",
            payload=RemoveSubjectFromCirclePayload(subject_id=subject_id, circle_id=circle_id),
        )
        return JSONResponse({"ok": True, "undo_token": token})

    @app.post("/api/subjects/{subject_id}/circles/{circle_id}/remove")
    def remove_subject_from_circle_api(
        request: Request, subject_id: int, circle_id: int
    ) -> JSONResponse:
        subject = db.get_subject(subject_id)
        circle = db.get_circle(circle_id)
        if subject is None or circle is None:
            raise HTTPException(404)
        removed = db.remove_subject_from_circle(subject_id, circle_id)
        if not removed:
            return JSONResponse({"ok": True, "already": True})
        token = undo_store.put(
            description=f"Removed {subject.name} from {circle.name}",
            payload=AddSubjectToCirclePayload(subject_id=subject_id, circle_id=circle_id),
        )
        return JSONResponse({"ok": True, "undo_token": token})

    @app.get("/api/circles/all")
    def list_circles_api() -> JSONResponse:
        circles = db.list_circles()
        return JSONResponse(
            {
                "circles": [
                    {"id": c.id, "name": c.name, "member_count": c.member_count} for c in circles
                ]
            }
        )

    # ── Catch-all /{kind}/... routes (must be after all /api/ and specific routes) ──

    @app.get("/{kind}/clusters", response_class=HTMLResponse)
    def clusters_page(request: Request, kind: KindType) -> HTMLResponse:
        species = _species_for_kind(kind)
        clusters = db.get_unnamed_clusters(species=species)
        return templates.TemplateResponse(
            name="clusters.html",
            context={"clusters": clusters, "kind": kind},
            request=request,
        )

    @app.get("/{kind}/singletons", response_class=HTMLResponse)
    def singletons_page(request: Request, kind: KindType) -> HTMLResponse:
        species = _species_for_kind(kind)
        subject_kind = _subject_kind_for_species(species)
        total = db.get_singleton_count(species=species)
        findings = db.get_singleton_findings(species=species, limit=200)
        subjects = db.get_subjects_by_kind(subject_kind)
        sources = db.get_sources_batch([f.source_id for f in findings])
        face_paths = {
            f.id: sources[f.source_id].file_path if f.source_id in sources else "" for f in findings
        }
        face_hints = compute_singleton_hints(db, findings, subject_kind)
        return templates.TemplateResponse(
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

    @app.get("/{kind}/merge-suggestions", response_class=HTMLResponse)
    def merge_suggestions_page(
        request: Request, kind: KindType, min_sim: float = 40.0
    ) -> HTMLResponse:
        return templates.TemplateResponse(
            name="merge_suggestions.html",
            context={"min_sim": min_sim, "kind": kind},
            request=request,
        )

    @app.get("/{kind}/compare", response_class=HTMLResponse)
    def compare_page(
        request: Request,
        kind: KindType,
        a: int | None = None,
        b: int | None = None,
    ) -> HTMLResponse:
        species = _species_for_kind(kind)
        subject_kind = _subject_kind_for_species(species)
        subjects = db.get_subjects_by_kind(subject_kind)
        result = None
        subject_a = None
        subject_b = None
        if a is not None and b is not None and a != b:
            result = compare_subjects(db, a, b)
            subject_a = db.get_subject(a)
            subject_b = db.get_subject(b)
        return templates.TemplateResponse(
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

    @app.get("/{kind}/{subject_id}/find-similar", response_class=HTMLResponse)
    def find_similar_page(
        request: Request, kind: KindType, subject_id: int, min_sim: float = 55.0
    ) -> HTMLResponse:
        subject = db.get_subject(subject_id)
        if not subject:
            raise HTTPException(404, "Subject not found")
        candidates = find_similar_unclustered(db, subject_id, min_similarity=min_sim / 100)
        return templates.TemplateResponse(
            name="find_similar.html",
            context={
                "subject": subject,
                "candidates": candidates,
                "min_sim": min_sim,
                "kind": kind,
            },
            request=request,
        )

    @app.get("/{kind}/{subject_id}", response_class=HTMLResponse)
    def subject_detail(request: Request, kind: KindType, subject_id: int) -> HTMLResponse:
        subject = db.get_subject(subject_id)
        if not subject:
            raise HTTPException(404, "Subject not found")
        findings_with_paths = db.get_subject_findings_with_paths(subject_id, limit=200)
        findings = [f for f, _ in findings_with_paths]
        finding_groups = _group_by_month(findings_with_paths, key="findings")
        all_sources = db.get_subject_sources(subject_id)
        sources = [s for s in all_sources if s.type == "photo"]
        source_groups = _group_by_month([(s, s.file_path) for s in sources], key="sources")
        # Videos tab: pair each video source with the subject's findings on it (for thumbnail + count).
        videos = db.get_subject_sources_with_findings(subject_id, source_type="video")
        video_groups = _group_by_month(
            [(entry, entry[0].file_path) for entry in videos], key="videos"
        )
        all_subjects = db.get_subjects()
        subject_circles = db.get_subject_circles(subject_id)
        all_circles = db.list_circles()
        has_unclustered = db.has_unclustered_findings(species=_species_for_kind(kind))
        return templates.TemplateResponse(
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
            },
            request=request,
        )

    @app.get("/{kind}", response_class=HTMLResponse)
    def subjects_page(request: Request, kind: KindType) -> HTMLResponse:
        species = _species_for_kind(kind)
        subject_kind = _subject_kind_for_species(species)
        subjects = db.get_subjects_by_kind(subject_kind)
        avatars = db.get_random_avatars([s.id for s in subjects])
        in_circle = {
            int(r[0])
            for r in db.conn.execute("SELECT DISTINCT subject_id FROM subject_circles").fetchall()
        }
        return templates.TemplateResponse(
            name="subjects.html",
            context={
                "subjects": subjects,
                "kind": kind,
                "avatars": avatars,
                "in_circle": in_circle,
            },
            request=request,
        )

    # ── Legacy redirects (301) for old query-param URLs ────

    @app.get("/clusters", response_class=RedirectResponse)
    def redirect_clusters(species: str = "human") -> RedirectResponse:
        kind = "pets" if species == "pet" else "people"
        return RedirectResponse(f"/{kind}/clusters", status_code=301)

    @app.get("/persons", response_class=RedirectResponse)
    def redirect_persons(species: str = "human") -> RedirectResponse:
        kind = "pets" if species == "pet" else "people"
        return RedirectResponse(f"/{kind}", status_code=301)

    @app.get("/persons/{person_id}", response_class=RedirectResponse)
    def redirect_person_detail(person_id: int) -> RedirectResponse:
        subject = db.get_subject(person_id)
        kind = _kind_for_subject(subject.kind) if subject else "people"
        return RedirectResponse(f"/{kind}/{person_id}", status_code=301)

    @app.get("/persons/{person_id}/find-similar", response_class=RedirectResponse)
    def redirect_find_similar(person_id: int) -> RedirectResponse:
        subject = db.get_subject(person_id)
        kind = _kind_for_subject(subject.kind) if subject else "people"
        return RedirectResponse(f"/{kind}/{person_id}/find-similar", status_code=301)

    @app.get("/singletons", response_class=RedirectResponse)
    def redirect_singletons(species: str = "human") -> RedirectResponse:
        kind = "pets" if species == "pet" else "people"
        return RedirectResponse(f"/{kind}/singletons", status_code=301)

    @app.get("/merge-suggestions", response_class=RedirectResponse)
    def redirect_merge_suggestions(species: str = "human") -> RedirectResponse:
        kind = "pets" if species == "pet" else "people"
        return RedirectResponse(f"/{kind}/merge-suggestions", status_code=301)

    @app.get("/compare", response_class=RedirectResponse)
    def redirect_compare(species: str = "human") -> RedirectResponse:
        kind = "pets" if species == "pet" else "people"
        return RedirectResponse(f"/{kind}/compare", status_code=301)

    return app
