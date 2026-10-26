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
from ..db.auth import AllowedUser
from ..undo import UndoStore
from . import deps, imaging, sessions
from .responses import JSONResponse
from .routers import (
    auth,
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


def _photo_archive_present(base_dir: Path | None) -> bool:
    """An empty photo directory means a lost mount, not an empty archive.

    The server reaches the archive through a bind mount over a separate
    disk. If that disk does not come back, the bind leaves an empty
    directory behind: every page still renders and every image 404s. Report
    it rather than look healthy.
    """
    if base_dir is None:
        return True
    try:
        return any(base_dir.iterdir())
    except OSError:
        return False


async def _resolve_logged_user(db: FaceDB, email: str) -> AllowedUser:
    allowed = await db.find_allowed_user(email)
    if allowed is None:
        msg = (
            f"{email} is not on the allowlist, so there is no identity to run as. "
            f"Add it with: ritrova auth allow {email}"
        )
        raise RuntimeError(msg)
    return allowed


def create_app(
    dsn: str,
    data_dir: str | Path,
    photos_dir: str | None = None,
    *,
    imaging_workers: int = 4,
    logged_user: str | None = None,
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
        # Local mode stands in for a completed Google login, so it may only
        # adopt an identity the allowlist already grants — otherwise it would
        # invent users and roles that no real sign-in could produce.
        if logged_user is not None:
            _app.state.logged_user = await _resolve_logged_user(db, logged_user)
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

    # Registered before any router so it sees every request, including
    # routes added later. Default is deny; see sessions.EXEMPT_PREFIXES.
    sessions.install(app)

    # Order is load-bearing: every /api/... router is registered before
    # `pages`, whose /{kind}/... catch-alls would otherwise shadow them.
    # See CLAUDE.md:16 and tests/routers/test_route_ordering.py.
    app.include_router(auth.router)
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
        """Liveness plus the two dependencies that fail silently: a pool that
        cannot reach Postgres, and a photo archive that is not actually
        mounted. Both otherwise leave a server that looks fine and serves
        nothing."""
        db = deps.get_db()
        try:
            await db.fetchval("SELECT 1")
        except (OSError, asyncpg.PostgresError) as exc:
            return JSONResponse(
                {"status": "unhealthy", "database": type(exc).__name__}, status_code=503
            )
        if not _photo_archive_present(db.base_dir):
            return JSONResponse(
                {"status": "unhealthy", "photos": "archive empty or unreadable"},
                status_code=503,
            )
        return JSONResponse({"status": "ok", "scan_models": scan_available()})

    # Registered last: ``/{kind}`` matches any single segment, so it would
    # shadow every literal route declared after it — /healthz included.
    app.include_router(pages.router)

    return app
