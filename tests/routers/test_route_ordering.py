"""Route-ordering regression test (CLAUDE.md:16 + ADR-012 §M1 acceptance).

The ``/{kind}/...`` catch-all in ``pages`` would silently shadow specific
``/api/...`` endpoints if any of them were registered after ``pages`` in
``create_app``. That would not show up as an import error — the app would
just start returning HTML where the client expects JSON, and a handful of
paginated htmx / JS calls would break silently.

This test walks every registered route and asserts:

1. Every ``/api/...`` route appears before the first ``/{kind}...`` route
   in the registration order.
2. A concrete GET to ``/api/findings/<missing-id>`` returns JSON 404, not
   the HTML 404 the ``/{kind}/...`` branch would produce.
"""

from __future__ import annotations

from pathlib import Path
from unittest import TestCase

import pytest
from fastapi.testclient import TestClient
from starlette.routing import Route

from ritrova.app import create_app


class TestRouteOrdering(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.app = create_app(str(tmp_path / "test.db"))
        self.client = TestClient(self.app)

    def test_api_routes_registered_before_kind_catchall(self) -> None:
        first_kind_idx: int | None = None
        last_api_idx: int | None = None
        for i, route in enumerate(self.app.router.routes):
            if not isinstance(route, Route):
                continue
            path = route.path
            if path.startswith("/api/"):
                last_api_idx = i
            elif path.startswith("/{kind}") and first_kind_idx is None:
                first_kind_idx = i
        assert first_kind_idx is not None, "expected /{kind}/... routes to be registered"
        assert last_api_idx is not None, "expected /api/... routes to be registered"
        assert last_api_idx < first_kind_idx, (
            f"/api/ route at index {last_api_idx} registered after /{{kind}}/... "
            f"at index {first_kind_idx} — this breaks CLAUDE.md:16."
        )

    def test_api_finding_thumbnail_missing_returns_json_not_html(self) -> None:
        """The /{kind}/... catch-all returns the HTML 404 page — if it were
        shadowing /api/findings/{id}/thumbnail we'd see text/html here."""
        resp = self.client.get("/api/findings/999999/thumbnail")
        assert resp.status_code == 404
        # FastAPI's default JSON error handler returns application/json.
        # The shadow path would be HTMLResponse('…', status_code=404).
        assert resp.headers["content-type"].startswith("application/json")

    def test_api_findings_mutation_body_validation_runs(self) -> None:
        """If `/api/findings/dismiss` were shadowed by `/{kind}/...`, the
        GET 405 vs JSON-422 body-validation distinction would be lost."""
        # POST with empty body hits the registered route; it validates the
        # body and either returns `{ok:true, dismissed:0}` or a 422. A 404
        # would mean the route isn't reachable at all.
        resp = self.client.post("/api/findings/dismiss", json={"face_ids": []})
        assert resp.status_code == 200, resp.text
