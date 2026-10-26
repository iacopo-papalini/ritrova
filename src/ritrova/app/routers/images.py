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

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from ...db import FaceDB, Finding
from ...images import crop_face_thumbnail, resize_photo
from .. import imaging
from ..deps import get_db, get_thumbnails_dir
from ..helpers import IMMUTABLE_IMAGE_HEADERS
from ..responses import JSONResponse

router = APIRouter()


async def _restore_video_frame_cache(db: FaceDB, finding: Finding, expected: Path) -> bool:
    """Recreate a missing cached frame JPEG at the path already stored in the DB."""
    if not finding.frame_path:
        return False
    source = await db.get_source(finding.source_id)
    if source is None or source.type != "video":
        return False
    video_path = db.resolve_path(source.file_path)
    if not video_path.exists():
        return False

    return await imaging.run(_extract_frame, video_path, finding.frame_number, expected)


def _extract_frame(video_path: Path, frame_number: int, destination: Path) -> bool:
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, bgr_frame = cap.read()
    finally:
        cap.release()
    if not ok:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    rgb = bgr_frame[:, :, ::-1]
    Image.fromarray(rgb).save(str(destination), "JPEG", quality=85)
    return destination.exists()


async def _resolve_finding_image_or_restore(db: FaceDB, finding: Finding) -> Path | None:
    resolved = await db.resolve_finding_image(finding)
    if resolved is None or resolved.exists():
        return resolved
    if await _restore_video_frame_cache(db, finding, resolved):
        return resolved
    return resolved


@router.get("/api/findings/{finding_id}/thumbnail")
async def finding_thumbnail(finding_id: UUID, size: int = 150) -> Response:
    db = get_db()
    size = min(size, 500)
    cache_path = get_thumbnails_dir() / f"{finding_id}_{size}.jpg"
    if cache_path.exists():
        return Response(
            cache_path.read_bytes(),
            media_type="image/jpeg",
            headers=IMMUTABLE_IMAGE_HEADERS,
        )

    finding = await db.get_finding(finding_id)
    if not finding:
        raise HTTPException(404)
    resolved = await _resolve_finding_image_or_restore(db, finding)
    if resolved is None or not resolved.exists():
        raise HTTPException(404)

    buf = await imaging.run(
        crop_face_thumbnail,
        resolved,
        finding.bbox_x,
        finding.bbox_y,
        finding.bbox_w,
        finding.bbox_h,
        size,
    )
    cache_path.write_bytes(buf.getvalue())
    return Response(buf.getvalue(), media_type="image/jpeg", headers=IMMUTABLE_IMAGE_HEADERS)


def _view_cache_path(source_id: UUID, size: int) -> Path:
    views = get_thumbnails_dir().parent / "views"
    views.mkdir(parents=True, exist_ok=True)
    return views / f"{source_id}_{size}.jpg"


@router.get("/api/sources/{source_id}/image")
async def source_image(source_id: UUID, max_size: int = 1600) -> Response:
    db = get_db()
    size = min(max_size, 2000)
    cache_path = _view_cache_path(source_id, size)
    if cache_path.exists():
        return Response(
            cache_path.read_bytes(),
            media_type="image/jpeg",
            headers=IMMUTABLE_IMAGE_HEADERS,
        )
    source = await db.get_source(source_id)
    if not source:
        raise HTTPException(404)
    if source.type == "video":
        # No raw image for videos; fall back to the first finding's extracted
        # frame so callers (Together grid, subject detail photo tab) can still
        # render a representative thumbnail. 404 only if truly no frames.
        findings = await db.get_source_findings(source_id)
        frame_finding = next((f for f in findings if f.frame_path), None)
        if frame_finding is None:
            raise HTTPException(404, "No frames available for this video")
        resolved = await _resolve_finding_image_or_restore(db, frame_finding)
        if resolved is None or not resolved.exists():
            raise HTTPException(404)
        buf = await imaging.run(resize_photo, resolved, size)
        cache_path.write_bytes(buf.getvalue())
        return Response(buf.getvalue(), media_type="image/jpeg", headers=IMMUTABLE_IMAGE_HEADERS)
    resolved = db.resolve_path(source.file_path)
    if not resolved.exists():
        raise HTTPException(404)
    buf = await imaging.run(resize_photo, resolved, size)
    cache_path.write_bytes(buf.getvalue())
    return Response(buf.getvalue(), media_type="image/jpeg", headers=IMMUTABLE_IMAGE_HEADERS)


@router.get("/api/sources/{source_id}/original")
async def source_original(source_id: UUID) -> FileResponse:
    """Serve the unmodified original file from disk with a download disposition."""
    db = get_db()
    source = await db.get_source(source_id)
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
async def source_info(source_id: UUID) -> JSONResponse:
    db = get_db()
    source = await db.get_source(source_id)
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
async def finding_frame(finding_id: UUID, max_size: int = 1600) -> Response:
    """Serve the image for a finding — the extracted frame for video findings,
    or the source image for photo findings. `resolve_finding_image` dispatches.
    """
    db = get_db()
    finding = await db.get_finding(finding_id)
    if not finding:
        raise HTTPException(404)
    resolved = await _resolve_finding_image_or_restore(db, finding)
    if resolved is None or not resolved.exists():
        raise HTTPException(404)
    buf = await imaging.run(resize_photo, resolved, min(max_size, 2000))
    return Response(buf.getvalue(), media_type="image/jpeg", headers=IMMUTABLE_IMAGE_HEADERS)


@router.get("/api/findings/{finding_id}/info")
async def finding_info(finding_id: UUID) -> JSONResponse:
    """Metadata for the finding's underlying source. `file_path` is always the
    source's path (not the frame path) so the lightbox shows the real filename.
    """
    db = get_db()
    finding = await db.get_finding(finding_id)
    if not finding:
        raise HTTPException(404)
    source = await db.get_source(finding.source_id)
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
