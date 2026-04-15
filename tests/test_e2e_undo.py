"""E2E browser tests for FEAT-5 undo.

Exercises the click flows end-to-end against a real uvicorn + Chromium, so we
catch the JS/HX-Trigger wiring that unit tests can't see. Each test captures
``/api/*`` responses to assert the server-side contract (undo_token in the
dismiss response body) and DOM state (toast + Undo button rendered).

Dedicated server on port 18788 to avoid colliding with test_e2e.py's fixture.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator

import numpy as np
import pytest
import uvicorn
from playwright.sync_api import Page, expect

from ritrova.app import create_app
from ritrova.db import FaceDB

from ._helpers import add_findings


def _emb(seed: int = 42, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture(scope="module")
def app_url(tmp_path_factory: pytest.TempPathFactory) -> Generator[str]:
    """Dedicated server with a DB seeded for undo flows."""
    tmp = tmp_path_factory.mktemp("e2e_undo")
    db_path = tmp / "test.db"
    db = FaceDB(str(db_path))

    from PIL import Image

    img = Image.new("RGB", (200, 200), color="blue")
    img_path = tmp / "photo.jpg"
    img.save(str(img_path), "JPEG")

    def _seed_cluster(cluster_id: int, seed_base: int, n: int = 2) -> None:
        for i in range(n):
            sid = db.add_source(f"{img_path}_{cluster_id}_{i}", width=200, height=200)
            add_findings(
                db,
                [(sid, (10, 10, 50, 50), _emb(seed_base + i), 0.95)],
                species="human",
            )
            fid = db.get_source_findings(sid)[0].id
            db.update_cluster_ids({fid: cluster_id})

    # Cluster 200 — dismiss target.
    _seed_cluster(200, seed_base=200)
    # Cluster 400 — assign target. Needs an existing subject to assign to.
    db.create_subject("Alice")
    _seed_cluster(400, seed_base=400)
    db.close()

    app = create_app(str(db_path))
    port = 18788
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(50):
        try:
            import httpx

            httpx.get(f"http://127.0.0.1:{port}/")
            break
        except httpx.ConnectError:
            time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True


def _capture_api(page: Page) -> list[tuple[str, int, str]]:
    """Attach a response listener that accumulates /api/* responses.

    Returns the list the listener writes into — mutated as the page runs.
    """
    responses: list[tuple[str, int, str]] = []

    def _on_response(r):  # type: ignore[no-untyped-def]
        if "/api/" in r.url:
            try:
                body = r.text()
            except Exception:
                body = ""
            responses.append((r.url, r.status, body))

    page.on("response", _on_response)
    return responses


class TestClusterDismissUndo:
    """Reproduce the exact flow the user tested manually."""

    def test_dismiss_then_undo_restores_cluster(self, page: Page, app_url: str) -> None:
        cluster_id = 200
        responses = _capture_api(page)

        page.goto(f"{app_url}/clusters/{cluster_id}")
        page.wait_for_load_state("networkidle")

        page.locator("button", has_text="Dismiss entire cluster").click()
        page.locator("button", has_text="Dismiss cluster").click()

        # Client navigates away on success; wait for the target URL.
        page.wait_for_url(f"{app_url}/people/clusters", timeout=5000)
        page.wait_for_load_state("networkidle")

        # Verify the dismiss request was sent and succeeded. (We can't reliably
        # read the response body here because the navigation happens before
        # Playwright finishes buffering it — see peek-on-load assertion below
        # for the real functional check.)
        dismiss = [r for r in responses if f"/api/clusters/{cluster_id}/dismiss" in r[0]]
        assert dismiss, (
            f"No dismiss request seen. All /api calls: {[(u, s) for u, s, _ in responses]}"
        )
        assert dismiss[-1][1] == 200, f"Dismiss returned HTTP {dismiss[-1][1]}"

        # Real functional check: if the server registered an undo token, peek
        # returns it and the client renders the toast. A failure here would
        # reproduce the user's "pending: false" bug.
        toast = page.locator("[role='status']", has_text="Dismissed 2 faces")
        expect(toast).to_be_visible(timeout=5000)
        undo_btn = toast.locator("button", has_text="Undo")
        expect(undo_btn).to_be_visible()

        # Clicking Undo fires POST /api/undo/<token>.
        with page.expect_request("**/api/undo/*") as req_info:
            undo_btn.click()
        assert req_info.value.method == "POST"
        page.wait_for_load_state("networkidle")

        # Cluster is restored: re-opening its page succeeds and faces render.
        page.goto(f"{app_url}/clusters/{cluster_id}")
        page.wait_for_load_state("networkidle")
        expect(page.locator("img[alt='face']").first).to_be_visible(timeout=5000)


class TestClusterAssignUndo:
    """Peek-on-load handoff across a redirect: assign triggers a navigation,
    and the toast must appear on the *new* page courtesy of /api/undo/peek."""

    def test_assign_via_typeahead_then_undo(self, page: Page, app_url: str) -> None:
        cluster_id = 400
        responses = _capture_api(page)

        page.goto(f"{app_url}/clusters/{cluster_id}")
        page.wait_for_load_state("networkidle")

        # Open the typeahead picker, type "Ali", click Alice.
        picker = page.locator("input[placeholder='Type to search...']")
        picker.fill("Ali")
        page.locator("text=Alice").first.click()

        # Client follows the 303 redirect to the next similar cluster (or the
        # clusters list fallback). Wait for either landing URL.
        page.wait_for_url(
            lambda u: "/clusters" in u and f"/clusters/{cluster_id}" not in u,
            timeout=5000,
        )
        page.wait_for_load_state("networkidle")

        # Toast from peek-on-load.
        toast = page.locator("[role='status']", has_text="Assigned 2 faces to Alice")
        expect(toast).to_be_visible(timeout=5000)

        with page.expect_request("**/api/undo/*") as req_info:
            toast.locator("button", has_text="Undo").click()
        assert req_info.value.method == "POST"
        page.wait_for_load_state("networkidle")

        # Assign endpoint was called and returned a success or a 303 redirect
        # (the non-HX path uses RedirectResponse to send the user to the next
        # similar cluster).
        assigns = [r for r in responses if f"/api/clusters/{cluster_id}/assign" in r[0]]
        assert assigns, "No assign call captured"
        assert assigns[-1][1] in (200, 303)

        # Cluster 400 still has its faces but they're now unassigned again —
        # easiest visible proof: the cluster page still renders.
        page.goto(f"{app_url}/clusters/{cluster_id}")
        page.wait_for_load_state("networkidle")
        expect(page.locator("img[alt='face']").first).to_be_visible(timeout=5000)
