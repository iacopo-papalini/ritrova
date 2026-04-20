"""Image-serving endpoints: finding thumbnails, source images, and metadata.

Everything here is read-only and content-addressed, so responses carry
``Cache-Control: immutable`` (``helpers.IMMUTABLE_IMAGE_HEADERS``). Source
files are read-only per design principle #5 ("Respect the archive"), which
is what makes immutable caching safe — the bytes for a given URL never
change.

The finding-metadata endpoint (``/info``) is grouped with the image
endpoints rather than with the finding-mutation endpoints in
``findings.py`` because the consumer (the lightbox) fetches it alongside
the image bytes — they share one aggregate from the client's perspective.
"""

from __future__ import annotations

import io

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ...images import crop_face_thumbnail, resize_photo
from ..deps import get_db, get_thumbnails_dir
from ..helpers import IMMUTABLE_IMAGE_HEADERS

router = APIRouter()


@router.get("/api/findings/{finding_id}/thumbnail")
def finding_thumbnail(finding_id: int, size: int = 150) -> StreamingResponse:
    db = get_db()
    size = min(size, 500)
    cache_path = get_thumbnails_dir() / f"{finding_id}_{size}.jpg"
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


@router.get("/api/sources/{source_id}/image")
def source_image(source_id: int, max_size: int = 1600) -> StreamingResponse:
    db = get_db()
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


@router.get("/api/sources/{source_id}/original")
def source_original(source_id: int) -> FileResponse:
    """Serve the unmodified original file from disk with a download disposition."""
    db = get_db()
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


@router.get("/api/sources/{source_id}/info")
def source_info(source_id: int) -> JSONResponse:
    db = get_db()
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


@router.get("/api/findings/{finding_id}/frame")
def finding_frame(finding_id: int, max_size: int = 1600) -> StreamingResponse:
    """Serve the image for a finding — the extracted frame for video findings,
    or the source image for photo findings. `resolve_finding_image` dispatches.
    """
    db = get_db()
    finding = db.get_finding(finding_id)
    if not finding:
        raise HTTPException(404)
    resolved = db.resolve_finding_image(finding)
    if resolved is None or not resolved.exists():
        raise HTTPException(404)
    buf = resize_photo(resolved, min(max_size, 2000))
    return StreamingResponse(buf, media_type="image/jpeg", headers=IMMUTABLE_IMAGE_HEADERS)


@router.get("/api/findings/{finding_id}/info")
def finding_info(finding_id: int) -> JSONResponse:
    """Metadata for the finding's underlying source. `file_path` is always the
    source's path (not the frame path) so the lightbox shows the real filename.
    """
    db = get_db()
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
