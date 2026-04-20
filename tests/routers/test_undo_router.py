"""Smoke test for the undo router in isolation (ADR-012 §M1 step 7).

Demonstrates the pattern: construct a ``FaceDB`` + ``UndoStore`` fixture,
call ``deps.configure(...)``, mount only the router under test, and
exercise its endpoints. Full-stack tests live alongside the mutation
tests in ``tests/test_undo.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest import TestCase

import pytest
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from ritrova.app import deps
from ritrova.app.routers import undo as undo_router
from ritrova.db import FaceDB
from ritrova.undo import UndoStore


class TestUndoRouterIsolated(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db")
        self.undo_store = UndoStore()
        templates = Jinja2Templates(directory=str(tmp_path))
        thumbnails_dir = tmp_path / "tmp" / "thumbnails"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        deps.configure(
            db=self.db,
            undo_store=self.undo_store,
            templates=templates,
            thumbnails_dir=thumbnails_dir,
        )
        self.app = FastAPI()
        self.app.include_router(undo_router.router)
        self.client = TestClient(self.app)

    def test_peek_empty_when_no_pending_undo(self) -> None:
        resp = self.client.get("/api/undo/peek")
        assert resp.status_code == 200
        assert resp.json() == {"pending": False}

    def test_apply_unknown_token_is_404(self) -> None:
        resp = self.client.post("/api/undo/nonexistent-token")
        assert resp.status_code == 404
