"""Ritrova — web application for browsing, naming, and searching faces.

This module is the assembly point: it builds the ``FaceDB`` + ``UndoStore``
singletons, hands them to the router package via ``deps.configure``, and
mounts the routers in the order the CLAUDE.md route-ordering invariant
requires.

**Router ordering is load-bearing.** The ``pages`` router owns the
``/{kind}/...`` catch-all pattern. FastAPI / Starlette matches routes in
the order they were registered, so an ``/api/...`` route registered after
``pages`` would be shadowed by ``/{kind}/clusters`` or similar. See
``CLAUDE.md:16`` and the regression test in
``tests/routers/test_route_ordering.py``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..db import FaceDB
from ..undo import UndoStore
from . import deps
from .routers import (
    circles,
    clusters,
    findings,
    images,
    legacy_redirects,
    pages,
    subjects,
    together,
    undo,
)

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
STATIC_DIR = Path(__file__).parent.parent / "static"


def create_app(db_path: str, photos_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="Ritrova")
    db = FaceDB(db_path, base_dir=photos_dir)
    undo_store = UndoStore()
    # Expose on app.state so tests can clear/peek between cases.
    app.state.undo_store = undo_store

    thumbnails_dir = Path(db_path).parent / "tmp" / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    deps.configure(db=db, undo_store=undo_store, templates=templates, thumbnails_dir=thumbnails_dir)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Order is load-bearing: every /api/... router is registered before
    # `pages`, whose /{kind}/... catch-alls would otherwise shadow them.
    # See CLAUDE.md:16 and tests/routers/test_route_ordering.py.
    app.include_router(images.router)
    app.include_router(findings.router)
    app.include_router(clusters.router)
    app.include_router(subjects.router)
    app.include_router(circles.router)
    app.include_router(together.router)
    app.include_router(undo.router)
    app.include_router(pages.router)
    # Legacy redirects go last so they don't shadow the canonical /{kind}/...
    # pages; they are the long-tail safety net for old bookmarks.
    app.include_router(legacy_redirects.router)

    return app
