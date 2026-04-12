"""Ritrova — web application for browsing, naming, and searching faces."""

import io
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from fastapi import Body, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .cluster import (
    compare_subjects,
    find_similar_cluster,
    find_similar_unclustered,
    rank_subjects_for_cluster,
    suggest_merges,
)
from .db import FaceDB
from .images import crop_face_thumbnail, resize_photo
from .services import (
    compute_cluster_hint,
    compute_singleton_hints,
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


TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

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
        for finding in findings:
            subject_name = None
            if finding.person_id:
                s = db.get_subject(finding.person_id)
                subject_name = s.name if s else None
            findings_data.append(
                {
                    "id": finding.id,
                    "bbox_x_pct": finding.bbox_x / source.width * 100,
                    "bbox_y_pct": finding.bbox_y / source.height * 100,
                    "bbox_w_pct": finding.bbox_w / source.width * 100,
                    "bbox_h_pct": finding.bbox_h / source.height * 100,
                    "person_id": finding.person_id,
                    "person_name": subject_name,
                    "confidence": finding.confidence,
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
        for fid in face_ids:
            db.assign_finding_to_subject(fid, target_person_id)
        return JSONResponse({"ok": True, "swapped": len(face_ids)})

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
        try:
            for fid in face_ids:
                db.assign_finding_to_subject(fid, subject_id, force=force)
        except ValueError as e:
            return JSONResponse({"error": str(e), "needs_confirm": True}, status_code=409)
        return JSONResponse({"ok": True, "claimed": len(face_ids)})

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
            "SELECT id, source_id FROM findings WHERE cluster_id = ? LIMIT ? OFFSET ?",
            (cluster_id, limit, offset),
        )
        return JSONResponse([{"id": r[0], "source_id": r[1]} for r in rows])

    @app.get("/api/clusters/{cluster_id}/faces-html", response_class=HTMLResponse)
    def cluster_faces_html(
        request: Request, cluster_id: int, offset: int = 0, limit: int = 200
    ) -> HTMLResponse:
        rows = db.query(
            "SELECT id, source_id FROM findings WHERE cluster_id = ? LIMIT ? OFFSET ?",
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
            "SELECT id, source_id FROM findings WHERE person_id = ? LIMIT ? OFFSET ?",
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

    @app.get("/api/findings/{finding_id}/thumbnail")
    def finding_thumbnail(finding_id: int, size: int = 150) -> StreamingResponse:
        size = min(size, 500)
        cache_path = thumbnails_dir / f"{finding_id}_{size}.jpg"
        if cache_path.exists():
            return StreamingResponse(io.BytesIO(cache_path.read_bytes()), media_type="image/jpeg")

        finding = db.get_finding(finding_id)
        if not finding:
            raise HTTPException(404)
        resolved = db.resolve_finding_image(finding)
        if not resolved.exists():
            raise HTTPException(404)

        buf = crop_face_thumbnail(
            resolved, finding.bbox_x, finding.bbox_y, finding.bbox_w, finding.bbox_h, size
        )
        cache_path.write_bytes(buf.getvalue())
        return StreamingResponse(buf, media_type="image/jpeg")

    @app.get("/api/sources/{source_id}/image")
    def source_image(source_id: int, max_size: int = 1600) -> StreamingResponse:
        source = db.get_source(source_id)
        if not source:
            raise HTTPException(404)
        if source.type == "video":
            raise HTTPException(404, "No image for video sources")
        resolved = db.resolve_path(source.file_path)
        if not resolved.exists():
            raise HTTPException(404)
        buf = resize_photo(resolved, min(max_size, 2000))
        return StreamingResponse(buf, media_type="image/jpeg")

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
            }
        )

    # ── API: actions ───────────────────────────────────────

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

    @app.post("/api/clusters/{cluster_id}/name")
    def name_cluster(cluster_id: int, name: str = Form(...)) -> RedirectResponse:
        # Determine subject kind from cluster's finding species
        findings = db.get_cluster_findings(cluster_id, limit=1)
        subject_kind = _subject_kind_for_species(findings[0].species) if findings else "person"
        subject_id = db.create_subject(name, kind=subject_kind)
        db.assign_cluster_to_subject(cluster_id, subject_id)
        return RedirectResponse(_next_similar_cluster(subject_id, cluster_id), status_code=303)

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
        try:
            db.assign_cluster_to_subject(cluster_id, person_id, force=force)
        except ValueError as e:
            return JSONResponse({"error": str(e), "needs_confirm": True}, status_code=409)
        if request.headers.get("HX-Request"):
            return HTMLResponse("")
        return RedirectResponse(_next_similar_cluster(person_id, cluster_id), status_code=303)

    @app.post("/api/clusters/{cluster_id}/dismiss")
    def dismiss_cluster(cluster_id: int) -> JSONResponse:
        """Dismiss all findings in a cluster as non-faces."""
        finding_ids = db.get_cluster_finding_ids(cluster_id)
        if finding_ids:
            db.dismiss_findings(finding_ids)
        return JSONResponse({"ok": True, "dismissed": len(finding_ids)})

    @app.post("/api/clusters/merge", response_model=None)
    def merge_clusters_api(
        request: Request,
        source_cluster: int = Form(...),
        target_cluster: int = Form(...),
    ) -> JSONResponse | HTMLResponse:
        """Move all findings from source cluster into target cluster."""
        db.merge_clusters(source_cluster, target_cluster)
        if request.headers.get("HX-Request"):
            return HTMLResponse("")
        return JSONResponse({"ok": True})

    @app.post("/api/findings/dismiss")
    def dismiss_findings(face_ids: list[int] = Body(..., embed=True)) -> JSONResponse:
        """Mark findings as non-faces (statues, paintings, dogs, etc.)."""
        db.dismiss_findings(face_ids)
        return JSONResponse({"ok": True, "dismissed": len(face_ids)})

    @app.post("/api/findings/exclude")
    def exclude_findings(
        face_ids: list[int] = Body(..., embed=True),
        cluster_id: int = Body(..., embed=True),
    ) -> JSONResponse:
        if face_ids:
            db.exclude_findings(face_ids, cluster_id=cluster_id)
        return JSONResponse({"ok": True, "excluded": len(face_ids)})

    @app.post("/api/findings/{finding_id}/assign")
    def assign_finding(
        finding_id: int, person_id: int = Form(...), force: bool = Form(False)
    ) -> JSONResponse:
        try:
            db.assign_finding_to_subject(finding_id, person_id, force=force)
        except ValueError as e:
            return JSONResponse({"error": str(e), "needs_confirm": True}, status_code=409)
        return JSONResponse({"ok": True})

    @app.post("/api/findings/{finding_id}/unassign")
    def unassign_finding(finding_id: int) -> JSONResponse:
        db.unassign_finding(finding_id)
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
        kind = _kind_for_subject(target.kind)
        db.merge_subjects(source_id, target_id)
        return RedirectResponse(f"/{kind}/{target_id}", status_code=303)

    @app.post("/api/subjects/{subject_id}/delete")
    def delete_subject(subject_id: int) -> RedirectResponse:
        """Unassign all findings and delete the subject."""
        subject = db.get_subject(subject_id)
        if not subject:
            raise HTTPException(404, "Subject not found")
        kind = _kind_for_subject(subject.kind)
        db.delete_subject(subject_id)
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

    @app.get("/api/together")
    def together_api(person_ids: str = "", alone: bool = False) -> JSONResponse:
        """Find sources containing ALL given subject IDs (comma-separated)."""
        if not person_ids.strip():
            return JSONResponse({"sources": [], "total": 0})
        ids = [int(x) for x in person_ids.split(",") if x.strip().isdigit()]
        sources = db.get_sources_with_all_subjects(ids, alone=alone)
        return JSONResponse(
            {
                "total": len(sources),
                "sources": [
                    {"id": s.id, "file_path": s.file_path, "taken_at": s.taken_at}
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
    ) -> HTMLResponse:
        if not person_ids.strip():
            return HTMLResponse("")
        ids = [int(x) for x in person_ids.split(",") if x.strip().isdigit()]
        total = db.count_sources_with_all_subjects(ids, alone=alone)
        sources = db.get_sources_with_all_subjects(ids, limit=limit, offset=offset, alone=alone)
        groups = _group_by_month([(s, s.file_path) for s in sources], key="sources")
        return templates.TemplateResponse(
            name="partials/together_results.html",
            context={
                "source_groups": groups,
                "total": total,
                "person_ids": person_ids,
                "alone": alone,
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
        all_subjects = db.get_subjects()
        return templates.TemplateResponse(
            name="subject_detail.html",
            context={
                "subject": subject,
                "findings": findings,
                "finding_groups": finding_groups,
                "sources": sources,
                "source_groups": source_groups,
                "all_subjects": all_subjects,
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
        return templates.TemplateResponse(
            name="subjects.html",
            context={"subjects": subjects, "kind": kind, "avatars": avatars},
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
