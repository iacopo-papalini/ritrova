"""FastAPI web application for browsing, naming, and searching faces."""

import io
import json
from pathlib import Path

from fastapi import Body, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps

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

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def create_app(db_path: str, photos_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="Face Recognition")
    db = FaceDB(db_path, base_dir=photos_dir)

    thumbnails_dir = Path(db_path).parent / "tmp" / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ── Pages ──────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, species: str = "human") -> HTMLResponse:
        stats = db.get_stats(species=species)
        return templates.TemplateResponse(
            name="index.html",
            context={"stats": stats, "species": species},
            request=request,
        )

    @app.get("/clusters", response_class=HTMLResponse)
    def clusters_page(request: Request, species: str = "human") -> HTMLResponse:
        clusters = db.get_unnamed_clusters(species=species)
        return templates.TemplateResponse(
            name="clusters.html",
            context={"clusters": clusters, "species": species},
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
        return templates.TemplateResponse(
            name="cluster_detail.html",
            context={
                "cluster_id": cluster_id,
                "faces": faces,
                "face_paths": face_paths,
                "total": total,
                "ranked_persons": ranked,
            },
            request=request,
        )

    @app.get("/singletons", response_class=HTMLResponse)
    def singletons_page(request: Request, species: str = "human") -> HTMLResponse:
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
                "species": species,
            },
            request=request,
        )

    @app.get("/persons", response_class=HTMLResponse)
    def persons_page(request: Request, species: str = "human") -> HTMLResponse:
        persons = db.get_persons_by_species(species)
        return templates.TemplateResponse(
            name="persons.html",
            context={"persons": persons, "species": species},
            request=request,
        )

    @app.get("/persons/{person_id}", response_class=HTMLResponse)
    def person_detail(request: Request, person_id: int) -> HTMLResponse:
        person = db.get_person(person_id)
        if not person:
            raise HTTPException(404, "Person not found")
        faces = db.get_person_faces(person_id, limit=200)
        photos = db.get_person_photos(person_id)
        all_persons = db.get_persons()
        return templates.TemplateResponse(
            name="person_detail.html",
            context={
                "person": person,
                "faces": faces,
                "photos": photos,
                "all_persons": all_persons,
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
        return templates.TemplateResponse(
            name="photo.html",
            context={
                "photo": photo,
                "faces_data": faces_data,
                "persons": persons,
            },
            request=request,
        )

    @app.get("/merge-suggestions", response_class=HTMLResponse)
    def merge_suggestions_page(
        request: Request, min_sim: float = 40.0, species: str = "human"
    ) -> HTMLResponse:
        persons_map = {p.id: p.name for p in db.get_persons()}
        return templates.TemplateResponse(
            name="merge_suggestions.html",
            context={
                "persons_map": persons_map,
                "min_sim": min_sim,
                "species": species,
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

    @app.get("/compare", response_class=HTMLResponse)
    def compare_page(
        request: Request,
        a: int | None = None,
        b: int | None = None,
    ) -> HTMLResponse:
        persons = db.get_persons()
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
        return templates.TemplateResponse(
            name="search.html", context={"query": q, "results": results}, request=request
        )

    @app.get("/persons/{person_id}/find-similar", response_class=HTMLResponse)
    def find_similar_page(request: Request, person_id: int, min_sim: float = 55.0) -> HTMLResponse:
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

    # ── API: pagination ─────────────────────────────────────

    @app.get("/api/clusters/{cluster_id}/faces")
    def cluster_faces_api(cluster_id: int, offset: int = 0, limit: int = 200) -> JSONResponse:
        rows = db.query(
            "SELECT id, photo_id FROM faces WHERE cluster_id = ? LIMIT ? OFFSET ?",
            (cluster_id, limit, offset),
        )
        return JSONResponse([{"id": r[0], "photo_id": r[1]} for r in rows])

    @app.get("/api/persons/{person_id}/faces")
    def person_faces_api(person_id: int, offset: int = 0, limit: int = 200) -> JSONResponse:
        rows = db.query(
            "SELECT id, photo_id FROM faces WHERE person_id = ? LIMIT ? OFFSET ?",
            (person_id, limit, offset),
        )
        return JSONResponse([{"id": r[0], "photo_id": r[1]} for r in rows])

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
    def photo_image(photo_id: int, max_size: int = 1600) -> StreamingResponse:
        max_size = min(max_size, 2000)
        photo = db.get_photo(photo_id)
        if not photo:
            raise HTTPException(404)
        resolved = db.resolve_path(photo.file_path)
        if not resolved.exists():
            raise HTTPException(404)

        with Image.open(resolved) as raw_img:
            oriented = ImageOps.exif_transpose(raw_img)
            oriented.thumbnail((max_size, max_size))
            buf = io.BytesIO()
            oriented.save(buf, "JPEG", quality=90)

        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")

    # ── API: actions ───────────────────────────────────────

    def _next_similar_cluster(person_id: int) -> str:
        """Find the unnamed cluster most similar to this person, return redirect URL."""
        next_cluster = find_similar_cluster(db, person_id)
        if next_cluster:
            return f"/clusters/{next_cluster}?suggested_person={person_id}"
        return "/clusters"

    @app.post("/api/clusters/{cluster_id}/name")
    def name_cluster(cluster_id: int, name: str = Form(...)) -> RedirectResponse:
        person_id = db.create_person(name)
        db.assign_cluster_to_person(cluster_id, person_id)
        return RedirectResponse(_next_similar_cluster(person_id), status_code=303)

    @app.post("/api/clusters/{cluster_id}/assign")
    def assign_cluster_to_existing(cluster_id: int, person_id: int = Form(...)) -> RedirectResponse:
        person = db.get_person(person_id)
        if not person:
            raise HTTPException(404, "Person not found")
        db.assign_cluster_to_person(cluster_id, person_id)
        return RedirectResponse(_next_similar_cluster(person_id), status_code=303)

    @app.post("/api/clusters/{cluster_id}/dismiss")
    def dismiss_cluster(cluster_id: int) -> JSONResponse:
        """Dismiss all faces in a cluster as non-faces."""
        face_ids = db.get_cluster_face_ids(cluster_id)
        if face_ids:
            db.dismiss_faces(face_ids)
        return JSONResponse({"ok": True, "dismissed": len(face_ids)})

    @app.post("/api/clusters/merge")
    def merge_clusters_api(
        source_cluster: int = Body(...), target_cluster: int = Body(...)
    ) -> JSONResponse:
        """Move all faces from source cluster into target cluster."""
        db.merge_clusters(source_cluster, target_cluster)
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
        db.rename_person(person_id, name)
        return RedirectResponse(f"/persons/{person_id}", status_code=303)

    @app.post("/api/persons/merge")
    def merge_persons(source_id: int = Form(...), target_id: int = Form(...)) -> RedirectResponse:
        if source_id == target_id:
            raise HTTPException(400, "Cannot merge person with themselves")
        db.merge_persons(source_id, target_id)
        return RedirectResponse(f"/persons/{target_id}", status_code=303)

    @app.post("/api/persons/{person_id}/delete")
    def delete_person(person_id: int) -> RedirectResponse:
        """Unassign all faces and delete the person."""
        db.delete_person(person_id)
        return RedirectResponse("/persons", status_code=303)

    @app.get("/api/export")
    def export_db() -> JSONResponse:
        data = json.loads(db.export_json())
        return JSONResponse(content=data)

    return app
