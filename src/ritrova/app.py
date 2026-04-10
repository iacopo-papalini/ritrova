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
from PIL import Image, ImageFile, ImageOps

from .cluster import (
    compare_persons,
    find_similar_cluster,
    find_similar_unclustered,
    rank_persons_for_cluster,
    suggest_merges,
)
from .db import FaceDB
from .services import (
    compute_cluster_hint,
    compute_singleton_hints,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

KindType = Literal["people", "pets"]
_KIND_TO_SPECIES: dict[str, str] = {"people": "human", "pets": "pet"}
_SPECIES_TO_KIND: dict[str, str] = {"human": "people", "pet": "people"}


def _species_for_kind(kind: KindType) -> str:
    return _KIND_TO_SPECIES[kind]


def _kind_for_species(species: str) -> KindType:
    """Map a DB species string to the URL kind."""
    if species in ("pet", "cat", "dog"):
        return "pets"
    return "people"


TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

_DATE_RE = re.compile(r"(\d{4})-(\d{2})")


def _month_from_path(file_path: str) -> str:
    """Extract YYYY-MM from directory names in a photo path (most reliable date source)."""
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
        total = db.get_cluster_face_count(cluster_id)
        if total == 0:
            raise HTTPException(404, "Cluster not found")
        faces = db.get_cluster_faces(cluster_id, limit=200)
        photos = db.get_photos_batch([f.photo_id for f in faces])
        face_paths = {
            f.id: photos[f.photo_id].file_path if f.photo_id in photos else "" for f in faces
        }
        ranked = rank_persons_for_cluster(db, cluster_id)
        species = faces[0].species if faces else "human"
        kind = _kind_for_species(species)
        return templates.TemplateResponse(
            name="cluster_detail.html",
            context={
                "cluster_id": cluster_id,
                "faces": faces,
                "face_paths": face_paths,
                "total": total,
                "ranked_persons": ranked,
                "kind": kind,
            },
            request=request,
        )

    @app.get("/api/singletons/faces-html", response_class=HTMLResponse)
    def singletons_faces_html(
        request: Request, species: str = "human", offset: int = 0, limit: int = 200
    ) -> HTMLResponse:
        faces = db.get_singleton_faces(species=species, limit=limit, offset=offset)
        face_hints = compute_singleton_hints(db, faces, species)
        total = db.get_singleton_count(species=species)
        return templates.TemplateResponse(
            name="partials/singleton_grid.html",
            context={
                "faces": faces,
                "face_hints": face_hints,
                "species": species,
                "offset": offset,
                "limit": limit,
                "total": total,
            },
            request=request,
        )

    @app.get("/photos/{photo_id}", response_class=HTMLResponse)
    def photo_page(request: Request, photo_id: int) -> HTMLResponse:
        photo = db.get_photo(photo_id)
        if not photo:
            raise HTTPException(404, "Photo not found")
        faces = db.get_photo_faces(photo_id)
        persons = db.get_persons()
        faces_data = []
        for face in faces:
            person_name = None
            if face.person_id:
                p = db.get_person(face.person_id)
                person_name = p.name if p else None
            faces_data.append(
                {
                    "id": face.id,
                    "bbox_x_pct": face.bbox_x / photo.width * 100,
                    "bbox_y_pct": face.bbox_y / photo.height * 100,
                    "bbox_w_pct": face.bbox_w / photo.width * 100,
                    "bbox_h_pct": face.bbox_h / photo.height * 100,
                    "person_id": face.person_id,
                    "person_name": person_name,
                    "confidence": face.confidence,
                }
            )
        species = faces[0].species if faces else "human"
        kind = _kind_for_species(species)
        return templates.TemplateResponse(
            name="photo.html",
            context={
                "photo": photo,
                "faces_data": faces_data,
                "persons": persons,
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
        suggestions = suggest_merges(db, min_similarity=min_sim, species=species)
        persons_map = {p.id: p.name for p in db.get_persons()}
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
                        "sample_face_ids_a": s.sample_face_ids_a,
                        "sample_face_ids_b": s.sample_face_ids_b,
                        "name_a": persons_map.get(s.cluster_a) if s.kind_a == "person" else None,
                        "name_b": persons_map.get(s.cluster_b) if s.kind_b == "person" else None,
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
        suggestions = suggest_merges(db, min_similarity=min_sim, species=species)
        persons_map = {p.id: p.name for p in db.get_persons()}
        total = len(suggestions)
        page = suggestions[offset : offset + limit]
        items = [
            {
                "cluster_a": s.cluster_a,
                "cluster_b": s.cluster_b,
                "similarity_pct": s.similarity_pct,
                "size_a": s.size_a,
                "size_b": s.size_b,
                "sample_face_ids_a": s.sample_face_ids_a,
                "sample_face_ids_b": s.sample_face_ids_b,
                "name_a": persons_map.get(s.cluster_a) if s.kind_a == "person" else None,
                "name_b": persons_map.get(s.cluster_b) if s.kind_b == "person" else None,
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

    @app.post("/api/faces/swap")
    def swap_faces(
        face_ids: list[int] = Body(...),
        target_person_id: int = Body(...),
    ) -> JSONResponse:
        for fid in face_ids:
            db.assign_face_to_person(fid, target_person_id)
        return JSONResponse({"ok": True, "swapped": len(face_ids)})

    @app.get("/search", response_class=HTMLResponse)
    def search_page(request: Request, q: str = "") -> HTMLResponse:
        results = db.search_persons(q) if q else []
        result_kinds = {
            p.id: "pets" if db.has_person_species(p.id, "pet") else "people" for p in results
        }
        avatars = db.get_random_avatars([p.id for p in results])
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

    @app.post("/api/persons/{person_id}/claim-faces")
    def claim_faces(person_id: int, face_ids: list[int] = Body(..., embed=True)) -> JSONResponse:
        for fid in face_ids:
            db.assign_face_to_person(fid, person_id)
        return JSONResponse({"ok": True, "claimed": len(face_ids)})

    @app.get("/api/clusters/{cluster_id}/hint")
    def cluster_hint_api(cluster_id: int) -> JSONResponse:
        """Return the best matching person for a cluster."""
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
            "SELECT id, photo_id FROM faces WHERE cluster_id = ? LIMIT ? OFFSET ?",
            (cluster_id, limit, offset),
        )
        return JSONResponse([{"id": r[0], "photo_id": r[1]} for r in rows])

    @app.get("/api/clusters/{cluster_id}/faces-html", response_class=HTMLResponse)
    def cluster_faces_html(
        request: Request, cluster_id: int, offset: int = 0, limit: int = 200
    ) -> HTMLResponse:
        rows = db.query(
            "SELECT id, photo_id FROM faces WHERE cluster_id = ? LIMIT ? OFFSET ?",
            (cluster_id, limit, offset),
        )
        faces = [{"id": r[0], "photo_id": r[1]} for r in rows]
        return templates.TemplateResponse(
            name="partials/face_grid.html",
            context={
                "faces": faces,
                "cluster_id": cluster_id,
                "offset": offset,
                "limit": limit,
                "total": db.get_cluster_face_count(cluster_id),
            },
            request=request,
        )

    @app.get("/api/persons/{person_id}/faces")
    def person_faces_api(person_id: int, offset: int = 0, limit: int = 200) -> JSONResponse:
        rows = db.query(
            "SELECT id, photo_id FROM faces WHERE person_id = ? LIMIT ? OFFSET ?",
            (person_id, limit, offset),
        )
        return JSONResponse([{"id": r[0], "photo_id": r[1]} for r in rows])

    @app.get("/api/persons/{person_id}/faces-html", response_class=HTMLResponse)
    def person_faces_html(
        request: Request, person_id: int, offset: int = 0, limit: int = 200
    ) -> HTMLResponse:
        faces_with_paths = db.get_person_faces_with_paths(person_id, limit=limit, offset=offset)
        face_groups = _group_by_month(faces_with_paths, key="faces")
        person = db.get_person(person_id)
        total = person.face_count if person else 0
        return templates.TemplateResponse(
            name="partials/person_face_grid.html",
            context={
                "face_groups": face_groups,
                "person_id": person_id,
                "offset": offset,
                "limit": limit,
                "total": total,
                "faces_count": len(faces_with_paths),
            },
            request=request,
        )

    # ── API: images ────────────────────────────────────────

    @app.get("/api/faces/{face_id}/thumbnail")
    def face_thumbnail(face_id: int, size: int = 150) -> StreamingResponse:
        size = min(size, 500)
        cache_path = thumbnails_dir / f"{face_id}_{size}.jpg"
        if cache_path.exists():
            data = cache_path.read_bytes()
            return StreamingResponse(io.BytesIO(data), media_type="image/jpeg")

        face = db.get_face(face_id)
        if not face:
            raise HTTPException(404)
        photo = db.get_photo(face.photo_id)
        if not photo:
            raise HTTPException(404)
        resolved = db.resolve_path(photo.file_path)
        if not resolved.exists():
            raise HTTPException(404)

        with Image.open(resolved) as raw_img:
            oriented = ImageOps.exif_transpose(raw_img)
            pad_w = face.bbox_w * 0.3
            pad_h = face.bbox_h * 0.3
            x1 = max(0, face.bbox_x - pad_w)
            y1 = max(0, face.bbox_y - pad_h)
            x2 = min(oriented.width, face.bbox_x + face.bbox_w + pad_w)
            y2 = min(oriented.height, face.bbox_y + face.bbox_h + pad_h)
            crop = oriented.crop((int(x1), int(y1), int(x2), int(y2)))

        crop.thumbnail((size, size))
        crop.save(cache_path, "JPEG", quality=85)

        buf = io.BytesIO()
        crop.save(buf, "JPEG", quality=85)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")

    @app.get("/api/photos/{photo_id}/image")
    def photo_image(photo_id: int) -> StreamingResponse:
        photo = db.get_photo(photo_id)
        if not photo:
            raise HTTPException(404)
        resolved = db.resolve_path(photo.file_path)
        if not resolved.exists():
            raise HTTPException(404)
        # Stream raw file — browsers handle EXIF rotation natively
        media = "image/jpeg" if resolved.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        return StreamingResponse(open(resolved, "rb"), media_type=media)  # noqa: SIM115

    @app.get("/api/photos/{photo_id}/info")
    def photo_info(photo_id: int) -> JSONResponse:
        photo = db.get_photo(photo_id)
        if not photo:
            raise HTTPException(404)
        return JSONResponse(
            {
                "file_path": str(db.resolve_path(photo.file_path)),
                "latitude": photo.latitude,
                "longitude": photo.longitude,
            }
        )

    # ── API: actions ───────────────────────────────────────

    def _next_similar_cluster(person_id: int, cluster_id: int) -> str:
        """Find the unnamed cluster most similar to this person, return redirect URL."""
        faces = db.get_cluster_faces(cluster_id, limit=1)
        species = faces[0].species if faces else "human"
        kind = _kind_for_species(species)
        fallback = f"/{kind}/clusters"

        sp = "pet" if species in db.PET_SPECIES else "human"
        next_cluster = find_similar_cluster(db, person_id, species=sp)
        if next_cluster:
            return f"/clusters/{next_cluster}?suggested_person={person_id}"
        return fallback

    @app.post("/api/clusters/{cluster_id}/name")
    def name_cluster(cluster_id: int, name: str = Form(...)) -> RedirectResponse:
        person_id = db.create_person(name)
        db.assign_cluster_to_person(cluster_id, person_id)
        return RedirectResponse(_next_similar_cluster(person_id, cluster_id), status_code=303)

    @app.post("/api/clusters/{cluster_id}/assign", response_model=None)
    def assign_cluster_to_existing(
        request: Request, cluster_id: int, person_id: int = Form(...)
    ) -> RedirectResponse | HTMLResponse:
        person = db.get_person(person_id)
        if not person:
            raise HTTPException(404, "Person not found")
        db.assign_cluster_to_person(cluster_id, person_id)
        if request.headers.get("HX-Request"):
            return HTMLResponse("")
        return RedirectResponse(_next_similar_cluster(person_id, cluster_id), status_code=303)

    @app.post("/api/clusters/{cluster_id}/dismiss")
    def dismiss_cluster(cluster_id: int) -> JSONResponse:
        """Dismiss all faces in a cluster as non-faces."""
        face_ids = db.get_cluster_face_ids(cluster_id)
        if face_ids:
            db.dismiss_faces(face_ids)
        return JSONResponse({"ok": True, "dismissed": len(face_ids)})

    @app.post("/api/clusters/merge", response_model=None)
    def merge_clusters_api(
        request: Request,
        source_cluster: int = Form(...),
        target_cluster: int = Form(...),
    ) -> JSONResponse | HTMLResponse:
        """Move all faces from source cluster into target cluster."""
        db.merge_clusters(source_cluster, target_cluster)
        if request.headers.get("HX-Request"):
            return HTMLResponse("")
        return JSONResponse({"ok": True})

    @app.post("/api/faces/dismiss")
    def dismiss_faces(face_ids: list[int] = Body(..., embed=True)) -> JSONResponse:
        """Mark faces as non-faces (statues, paintings, dogs, etc.)."""
        db.dismiss_faces(face_ids)
        return JSONResponse({"ok": True, "dismissed": len(face_ids)})

    @app.post("/api/faces/exclude")
    def exclude_faces(
        face_ids: list[int] = Body(..., embed=True),
        cluster_id: int = Body(..., embed=True),
    ) -> JSONResponse:
        if face_ids:
            db.exclude_faces(face_ids, cluster_id=cluster_id)
        return JSONResponse({"ok": True, "excluded": len(face_ids)})

    @app.post("/api/faces/{face_id}/assign")
    def assign_face(face_id: int, person_id: int = Form(...)) -> JSONResponse:
        db.assign_face_to_person(face_id, person_id)
        return JSONResponse({"ok": True})

    @app.post("/api/faces/{face_id}/unassign")
    def unassign_face(face_id: int) -> JSONResponse:
        db.unassign_face(face_id)
        return JSONResponse({"ok": True})

    @app.post("/api/persons/{person_id}/rename")
    def rename_person(person_id: int, name: str = Form(...)) -> RedirectResponse:
        kind = _kind_for_species("pet" if db.has_person_species(person_id, "pet") else "human")
        db.rename_person(person_id, name)
        return RedirectResponse(f"/{kind}/{person_id}", status_code=303)

    @app.post("/api/persons/merge")
    def merge_persons(source_id: int = Form(...), target_id: int = Form(...)) -> RedirectResponse:
        if source_id == target_id:
            raise HTTPException(400, "Cannot merge person with themselves")
        kind = _kind_for_species("pet" if db.has_person_species(target_id, "pet") else "human")
        db.merge_persons(source_id, target_id)
        return RedirectResponse(f"/{kind}/{target_id}", status_code=303)

    @app.post("/api/persons/{person_id}/delete")
    def delete_person(person_id: int) -> RedirectResponse:
        """Unassign all faces and delete the person."""
        kind = _kind_for_species("pet" if db.has_person_species(person_id, "pet") else "human")
        db.delete_person(person_id)
        return RedirectResponse(f"/{kind}", status_code=303)

    @app.post("/api/persons/create")
    def create_person_api(name: str = Body(..., embed=True)) -> JSONResponse:
        """Create a person and return its data. Used by typeahead picker."""
        person_id = db.create_person(name)
        person = db.get_person(person_id)
        assert person is not None
        return JSONResponse({"id": person.id, "name": person.name, "face_count": person.face_count})

    @app.get("/api/persons/all")
    def all_persons_api() -> JSONResponse:
        """All persons and pets for typeahead components, with avatar face ID."""
        persons = db.get_persons()
        avatars = db.get_random_avatars([p.id for p in persons])
        return JSONResponse(
            [
                {
                    "id": p.id,
                    "name": p.name,
                    "face_count": p.face_count,
                    "face_id": avatars.get(p.id),
                }
                for p in persons
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
    def together_api(person_ids: str = "") -> JSONResponse:
        """Find photos containing ALL given person IDs (comma-separated)."""
        if not person_ids.strip():
            return JSONResponse({"photos": [], "total": 0})
        ids = [int(x) for x in person_ids.split(",") if x.strip().isdigit()]
        photos = db.get_photos_with_all_persons(ids)
        return JSONResponse(
            {
                "total": len(photos),
                "photos": [
                    {"id": p.id, "file_path": p.file_path, "taken_at": p.taken_at}
                    for p in photos[:200]
                ],
            }
        )

    @app.get("/api/together-html", response_class=HTMLResponse)
    def together_html(
        request: Request, person_ids: str = "", offset: int = 0, limit: int = 60
    ) -> HTMLResponse:
        if not person_ids.strip():
            return HTMLResponse("")
        ids = [int(x) for x in person_ids.split(",") if x.strip().isdigit()]
        total = db.count_photos_with_all_persons(ids)
        photos = db.get_photos_with_all_persons(ids, limit=limit, offset=offset)
        groups = _group_by_month([(p, p.file_path) for p in photos], key="photos")
        return templates.TemplateResponse(
            name="partials/together_results.html",
            context={
                "photo_groups": groups,
                "total": total,
                "person_ids": person_ids,
                "offset": offset,
                "limit": limit,
                "page_count": len(photos),
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
        total = db.get_singleton_count(species=species)
        faces = db.get_singleton_faces(species=species, limit=200)
        persons = db.get_persons_by_species(species)
        photos = db.get_photos_batch([f.photo_id for f in faces])
        face_paths = {
            f.id: photos[f.photo_id].file_path if f.photo_id in photos else "" for f in faces
        }
        face_hints = compute_singleton_hints(db, faces, species)
        return templates.TemplateResponse(
            name="singletons.html",
            context={
                "faces": faces,
                "face_paths": face_paths,
                "face_hints": face_hints,
                "persons": persons,
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
        persons = db.get_persons_by_species(species)
        result = None
        person_a = None
        person_b = None
        if a is not None and b is not None and a != b:
            result = compare_persons(db, a, b)
            person_a = db.get_person(a)
            person_b = db.get_person(b)
        return templates.TemplateResponse(
            name="compare.html",
            context={
                "persons": persons,
                "person_a": person_a,
                "person_b": person_b,
                "result": result,
                "selected_a": a,
                "selected_b": b,
                "kind": kind,
            },
            request=request,
        )

    @app.get("/{kind}/{person_id}/find-similar", response_class=HTMLResponse)
    def find_similar_page(
        request: Request, kind: KindType, person_id: int, min_sim: float = 55.0
    ) -> HTMLResponse:
        person = db.get_person(person_id)
        if not person:
            raise HTTPException(404, "Person not found")
        candidates = find_similar_unclustered(db, person_id, min_similarity=min_sim / 100)
        return templates.TemplateResponse(
            name="find_similar.html",
            context={
                "person": person,
                "candidates": candidates,
                "min_sim": min_sim,
                "kind": kind,
            },
            request=request,
        )

    @app.get("/{kind}/{person_id}", response_class=HTMLResponse)
    def person_detail(request: Request, kind: KindType, person_id: int) -> HTMLResponse:
        person = db.get_person(person_id)
        if not person:
            raise HTTPException(404, "Person not found")
        faces_with_paths = db.get_person_faces_with_paths(person_id, limit=200)
        faces = [f for f, _ in faces_with_paths]
        face_groups = _group_by_month(faces_with_paths, key="faces")
        photos = db.get_person_photos(person_id)
        photo_groups = _group_by_month([(p, p.file_path) for p in photos], key="photos")
        all_persons = db.get_persons()
        return templates.TemplateResponse(
            name="person_detail.html",
            context={
                "person": person,
                "faces": faces,
                "face_groups": face_groups,
                "photos": photos,
                "photo_groups": photo_groups,
                "all_persons": all_persons,
                "kind": kind,
            },
            request=request,
        )

    @app.get("/{kind}", response_class=HTMLResponse)
    def persons_page(request: Request, kind: KindType) -> HTMLResponse:
        species = _species_for_kind(kind)
        persons = db.get_persons_by_species(species)
        avatars = db.get_random_avatars([p.id for p in persons])
        return templates.TemplateResponse(
            name="persons.html",
            context={"persons": persons, "kind": kind, "avatars": avatars},
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
        kind = _kind_for_species("pet" if db.has_person_species(person_id, "pet") else "human")
        return RedirectResponse(f"/{kind}/{person_id}", status_code=301)

    @app.get("/persons/{person_id}/find-similar", response_class=RedirectResponse)
    def redirect_find_similar(person_id: int) -> RedirectResponse:
        kind = _kind_for_species("pet" if db.has_person_species(person_id, "pet") else "human")
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
