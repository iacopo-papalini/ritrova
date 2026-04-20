"""Shared singletons for router modules.

``create_app`` is the only caller that sets these — tests and production
both go through the same entry point, so the module-level state is
initialised exactly once per application instance. Routers import the
*accessors* (``get_db`` / ``get_undo_store`` / ``get_templates``) rather
than the module globals so the "must be configured first" check runs on
every call; attempting to use a router against an uninitialised process
yields a clear ``RuntimeError`` instead of an ``AttributeError`` on
``None``.

There is deliberately no FastAPI ``Depends``-style injection here. The
app is single-process, single-DB, single-photo-root; a ``Depends`` wiring
would add plumbing without gaining anything we use.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates

from ..db import FaceDB
from ..undo import UndoStore

_db: FaceDB | None = None
_undo_store: UndoStore | None = None
_templates: Jinja2Templates | None = None
_thumbnails_dir: Path | None = None


def configure(
    *,
    db: FaceDB,
    undo_store: UndoStore,
    templates: Jinja2Templates,
    thumbnails_dir: Path,
) -> None:
    """Called once from ``create_app`` at startup."""
    global _db, _undo_store, _templates, _thumbnails_dir
    _db = db
    _undo_store = undo_store
    _templates = templates
    _thumbnails_dir = thumbnails_dir


def get_db() -> FaceDB:
    if _db is None:
        raise RuntimeError("deps.configure(...) has not been called — use create_app()")
    return _db


def get_undo_store() -> UndoStore:
    if _undo_store is None:
        raise RuntimeError("deps.configure(...) has not been called — use create_app()")
    return _undo_store


def get_templates() -> Jinja2Templates:
    if _templates is None:
        raise RuntimeError("deps.configure(...) has not been called — use create_app()")
    return _templates


def get_thumbnails_dir() -> Path:
    if _thumbnails_dir is None:
        raise RuntimeError("deps.configure(...) has not been called — use create_app()")
    return _thumbnails_dir
