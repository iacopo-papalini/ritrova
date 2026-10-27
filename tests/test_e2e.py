"""E2E browser tests for Ritrova using Playwright."""

import asyncio
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
from ._postgres import _TABLES
from .conftest import TEST_DSN


def _emb(seed: int = 42, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


E2E_USER = "e2e-user@example.com"


async def _truncate(db: FaceDB) -> None:
    """These modules seed a server of their own, so they start from empty."""
    await db.execute(f"TRUNCATE {', '.join(_TABLES)} RESTART IDENTITY")


@pytest.fixture(scope="module")
def app_url(tmp_path_factory: pytest.TempPathFactory) -> Generator[str]:
    """Start a real Ritrova server with test data, return its URL."""
    tmp = tmp_path_factory.mktemp("e2e")

    async def _seed() -> None:
        db = await FaceDB.connect(TEST_DSN, data_dir=tmp)
        await _truncate(db)

        # Create test image
        from PIL import Image

        img = Image.new("RGB", (200, 200), color="blue")
        img_path = tmp / "photo.jpg"
        img.save(str(img_path), "JPEG")

        # Seed data: 2 subjects with findings
        sid_alice = await db.create_subject("Alice")
        source_id = await db.add_source(str(img_path), width=200, height=200)
        await add_findings(db, [(source_id, (10, 10, 50, 50), _emb(1), 0.95)], species="human")
        findings = await db.get_source_findings(source_id)
        await db.assign_finding_to_subject(findings[0].id, sid_alice)

        sid_bob = await db.create_subject("Bob")
        source_id2 = await db.add_source(str(img_path) + "2", width=200, height=200)
        await add_findings(db, [(source_id2, (10, 10, 50, 50), _emb(2), 0.95)], species="human")
        findings2 = await db.get_source_findings(source_id2)
        await db.assign_finding_to_subject(findings2[0].id, sid_bob)

        # Subject with gnarly characters: apostrophe, emoji, brackets
        sid_weird = await db.create_subject("Al'ice \U0001f9d1<test>")
        source_id5 = await db.add_source(str(img_path) + "5", width=200, height=200)
        await add_findings(db, [(source_id5, (10, 10, 50, 50), _emb(5), 0.95)], species="human")
        findings5 = await db.get_source_findings(source_id5)
        await db.assign_finding_to_subject(findings5[0].id, sid_weird)

        # An unassigned cluster
        source_id3 = await db.add_source(str(img_path) + "3", width=200, height=200)
        await add_findings(db, [(source_id3, (10, 10, 50, 50), _emb(3), 0.95)], species="human")
        source_id4 = await db.add_source(str(img_path) + "4", width=200, height=200)
        await add_findings(db, [(source_id4, (10, 10, 50, 50), _emb(3), 0.95)], species="human")
        f3 = await db.get_source_findings(source_id3)
        f4 = await db.get_source_findings(source_id4)
        await db.update_cluster_ids({f3[0].id: 100, f4[0].id: 100})

        # The request gate authenticates everything, so the browser needs an
        # identity. Local mode supplies one from the allowlist, exactly as it
        # does on the Mac — which keeps CSRF and the role checks live in these
        # tests rather than bypassed.
        await db.allow_user(E2E_USER, role="admin")
        await db.close()

    # A loop may already be running here, and asyncio.run cannot nest. A thread
    # swallows what it raises, so the failure is carried out explicitly —
    # otherwise seeding errors surface much later as an empty page.
    seed_error: list[BaseException] = []

    def _run_seed() -> None:
        try:
            asyncio.run(_seed())
        except BaseException as exc:  # noqa: BLE001
            seed_error.append(exc)

    seeder = threading.Thread(target=_run_seed)
    seeder.start()
    seeder.join()
    if seed_error:
        raise seed_error[0]

    app = create_app(TEST_DSN, tmp, logged_user=E2E_USER)
    port = 18787
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Wait for server to be ready
    import httpx

    for _ in range(100):
        try:
            httpx.get(f"http://127.0.0.1:{port}/", timeout=2.0)
            break
        except httpx.ConnectError, httpx.ReadError:
            time.sleep(0.1)
    else:
        # Yielding the URL anyway would surface later as an unexplained
        # ERR_ABORTED in the browser rather than as a startup failure.
        raise RuntimeError(f"server on port {port} never became ready")
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True


class TestTypeaheadPicker:
    """Test the person picker typeahead component."""

    def test_picker_shows_results_on_type(self, page: Page, app_url: str) -> None:
        page.goto(f"{app_url}/clusters/100")
        picker_input = page.locator("input[placeholder='Type to search...']")
        picker_input.fill("Ali")
        # Should show Alice in the dropdown list
        dropdown = page.locator("[class*='overflow-y-auto']")
        expect(dropdown.locator("text=Alice")).to_be_visible(timeout=3000)

    def test_picker_shows_create_option(self, page: Page, app_url: str) -> None:
        page.goto(f"{app_url}/clusters/100")
        picker_input = page.locator("input[placeholder='Type to search...']")
        picker_input.fill("NewPerson")
        # Should show create option
        expect(page.locator("text=Create")).to_be_visible(timeout=3000)

    def test_picker_shows_avatar_thumbnails(self, page: Page, app_url: str) -> None:
        page.goto(f"{app_url}/clusters/100")
        picker_input = page.locator("input[placeholder='Type to search...']")
        picker_input.click()
        # Wait for dropdown to appear and check for avatar images
        expect(page.locator("img[class*='rounded-full']").first).to_be_visible(timeout=3000)


class TestSubjectsListFilter:
    """Test inline filter on the subjects list page."""

    def test_list_page_loads_with_special_chars(self, page: Page, app_url: str) -> None:
        """Subject names with apostrophes, emoji, and angle brackets don't break the page."""
        page.goto(f"{app_url}/people")
        page.wait_for_load_state("networkidle")
        # All 3 subjects should be visible (Alice, Bob, and the gnarly one)
        expect(page.locator("h1")).to_contain_text("(3)")
        # No JS errors: the filter input should be functional
        filter_input = page.locator("input[placeholder='Filter...']")
        expect(filter_input).to_be_visible()

    def test_filter_narrows_results(self, page: Page, app_url: str) -> None:
        page.goto(f"{app_url}/people")
        filter_input = page.locator("input[placeholder='Filter...']")
        filter_input.fill("Ali")
        # Both "Alice" subjects should match (Alice and Al'ice...)
        # Bob should be hidden
        expect(page.locator("text=Bob").first).not_to_be_visible(timeout=2000)

    def test_filter_no_matches(self, page: Page, app_url: str) -> None:
        page.goto(f"{app_url}/people")
        filter_input = page.locator("input[placeholder='Filter...']")
        filter_input.fill("zzzznonexistent")
        expect(page.locator("text=No matches")).to_be_visible(timeout=2000)


class TestLightbox:
    """Test the lightbox component."""

    def test_lightbox_opens_on_thumbnail_click(self, page: Page, app_url: str) -> None:
        page.goto(f"{app_url}/people")
        # Go to a person detail page
        page.locator("text=Alice").first.click()
        page.wait_for_load_state("networkidle")
        # Click a face thumbnail image
        face_img = page.locator("img[alt='face']").first
        face_img.click()
        # Lightbox should be visible
        expect(page.locator("[aria-label='Close lightbox']")).to_be_visible(timeout=3000)

    def test_lightbox_closes_on_escape(self, page: Page, app_url: str) -> None:
        page.goto(f"{app_url}/people")
        page.locator("text=Alice").first.click()
        page.wait_for_load_state("networkidle")
        face_img = page.locator("img[alt='face']").first
        face_img.click()
        expect(page.locator("[aria-label='Close lightbox']")).to_be_visible(timeout=3000)
        page.keyboard.press("Escape")
        expect(page.locator("[aria-label='Close lightbox']")).not_to_be_visible(timeout=3000)

    def test_lightbox_rotate_button(self, page: Page, app_url: str) -> None:
        page.goto(f"{app_url}/people")
        page.locator("text=Alice").first.click()
        page.wait_for_load_state("networkidle")
        page.locator("img[alt='face']").first.click()
        expect(page.locator("[aria-label='Rotate image']")).to_be_visible(timeout=3000)
        page.locator("[aria-label='Rotate image']").click()

    def test_lightbox_arrow_navigation(self, page: Page, app_url: str) -> None:
        """FEAT-18: arrow keys navigate between findings in a multi-item lightbox.

        The fixture seeds cluster 100 with two findings on distinct sources, so the
        lightbox opens with a 2-item list and ArrowRight should swap the image src.
        """
        page.goto(f"{app_url}/clusters/100")
        page.wait_for_load_state("networkidle")

        # Open the lightbox on the first thumbnail. The full-resolution <img> has
        # alt="Full resolution photo" — distinct from the small "face" thumbnail.
        page.locator("img[alt='face']").first.click()
        full_img = page.locator("img[alt='Full resolution photo']")
        expect(full_img).to_be_visible(timeout=3000)

        # Wait for src to be populated (Alpine resolves it on open).
        page.wait_for_function(
            "() => document.querySelector('img[alt=\"Full resolution photo\"]')?.getAttribute('src')",
            timeout=3000,
        )
        first_src = full_img.get_attribute("src")
        assert first_src and "/api/findings/" in first_src, (
            f"Expected finding-keyed src, got {first_src!r}"
        )

        # Right arrow → second item; src should change.
        page.keyboard.press("ArrowRight")
        page.wait_for_function(
            f"() => document.querySelector('img[alt=\"Full resolution photo\"]')"
            f"?.getAttribute('src') !== {first_src!r}",
            timeout=3000,
        )
        second_src = full_img.get_attribute("src")
        assert second_src != first_src

        # Left arrow → back to first.
        page.keyboard.press("ArrowLeft")
        page.wait_for_function(
            f"() => document.querySelector('img[alt=\"Full resolution photo\"]')"
            f"?.getAttribute('src') === {first_src!r}",
            timeout=3000,
        )


class TestNavigation:
    """Test kind-based navigation."""

    def test_dashboard_loads(self, page: Page, app_url: str) -> None:
        page.goto(app_url)
        expect(page.locator("text=Ritrova")).to_be_visible()
        expect(page.locator("text=Dashboard")).to_be_visible()

    def test_people_clusters_loads(self, page: Page, app_url: str) -> None:
        page.goto(f"{app_url}/people/clusters")
        expect(page.locator("text=Unnamed Clusters")).to_be_visible()

    def test_people_directory_loads(self, page: Page, app_url: str) -> None:
        page.goto(f"{app_url}/people")
        expect(page.locator("h1", has_text="Persons")).to_be_visible()
        expect(page.locator("text=Alice").first).to_be_visible()
        expect(page.locator("text=Bob").first).to_be_visible()

    def test_toggle_pills_switch_kind(self, page: Page, app_url: str) -> None:
        page.goto(f"{app_url}/people/clusters")
        page.locator("a", has_text="Pets").click()
        expect(page).to_have_url(f"{app_url}/pets/clusters")

    def test_nav_links_carry_kind(self, page: Page, app_url: str) -> None:
        """Nav links should point to /{kind}/... paths."""
        page.goto(f"{app_url}/people/clusters")
        persons_link = page.locator("nav a", has_text="Persons")
        expect(persons_link).to_have_attribute("href", "/people")
