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

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..capabilities import scan_available
from ..db import FaceDB
from ..undo import UndoStore
from . import deps, imaging
from .responses import JSONResponse
from .routers import (
    browse,
    circles,
    clusters,
    findings,
    images,
    pages,
    print_selection,
    subjects,
    together,
    undo,
)

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
STATIC_DIR = Path(__file__).parent.parent / "static"


def create_app(
    dsn: str,
    data_dir: str | Path,
    photos_dir: str | None = None,
    *,
    imaging_workers: int = 4,
) -> FastAPI:
    undo_store = UndoStore()
    thumbnails_dir = Path(data_dir) / "tmp" / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    # ``|tojson`` is how ids reach the JavaScript, and json.dumps has no
    # rendering for UUID or datetime — without this every page that hands
    # Alpine an id raises at render time.
    templates.env.policies["json.dumps_kwargs"] = {"sort_keys": True, "default": str}
    # Templates hide the affordances a base install cannot serve, so a
    # server build offers no Shift+drag the API would only refuse.
    templates.env.globals["scan_models"] = scan_available()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # The pool has to be created inside the running loop, so it belongs
        # here rather than in the constructor.
        db = await FaceDB.connect(dsn, data_dir=data_dir, base_dir=photos_dir)
        imaging.configure(imaging_workers)
        deps.configure(
            db=db,
            undo_store=undo_store,
            templates=templates,
            thumbnails_dir=thumbnails_dir,
        )
        try:
            yield
        finally:
            imaging.shutdown()
            await db.close()

    app = FastAPI(title="Ritrova", lifespan=lifespan)
    # Expose on app.state so tests can clear/peek between cases.
    app.state.undo_store = undo_store

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Order is load-bearing: every /api/... router is registered before
    # `pages`, whose /{kind}/... catch-alls would otherwise shadow them.
    # See CLAUDE.md:16 and tests/routers/test_route_ordering.py.
    app.include_router(images.router)
    app.include_router(findings.router)
    app.include_router(clusters.router)
    app.include_router(subjects.router)
    app.include_router(circles.router)
    app.include_router(browse.router)
    app.include_router(print_selection.router)
    app.include_router(together.router)
    app.include_router(undo.router)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        """Liveness plus a real query, so a pool that cannot reach Postgres
        reports unhealthy instead of serving 500s to every page."""
        try:
            await deps.get_db().fetchval("SELECT 1")
        except (OSError, asyncpg.PostgresError) as exc:
            return JSONResponse(
                {"status": "unhealthy", "database": type(exc).__name__}, status_code=503
            )
        return JSONResponse({"status": "ok", "scan_models": scan_available()})

    # Registered last: ``/{kind}`` matches any single segment, so it would
    # shadow every literal route declared after it — /healthz included.
    app.include_router(pages.router)

    return app
