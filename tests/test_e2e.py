"""E2E browser tests for Ritrova using Playwright."""

import threading
import time

import numpy as np
import pytest
import uvicorn
from playwright.sync_api import Page, expect

from ritrova.app import create_app
from ritrova.db import FaceDB


def _emb(seed: int = 42, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture(scope="module")
def app_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Start a real Ritrova server with test data, return its URL."""
    tmp = tmp_path_factory.mktemp("e2e")
    db_path = tmp / "test.db"
    db = FaceDB(str(db_path))

    # Create test image
    from PIL import Image

    img = Image.new("RGB", (200, 200), color="blue")
    img_path = tmp / "photo.jpg"
    img.save(str(img_path), "JPEG")

    # Seed data: 2 subjects with faces
    sid_alice = db.create_subject("Alice")
    photo_id = db.add_photo(str(img_path), 200, 200)
    db.add_faces_batch([(photo_id, (10, 10, 50, 50), _emb(1), 0.95)], species="human")
    faces = db.get_photo_faces(photo_id)
    db.assign_face_to_subject(faces[0].id, sid_alice)

    sid_bob = db.create_subject("Bob")
    photo_id2 = db.add_photo(str(img_path) + "2", 200, 200)
    db.add_faces_batch([(photo_id2, (10, 10, 50, 50), _emb(2), 0.95)], species="human")
    faces2 = db.get_photo_faces(photo_id2)
    db.assign_face_to_subject(faces2[0].id, sid_bob)

    # Subject with gnarly characters: apostrophe, emoji, brackets
    sid_weird = db.create_subject("Al'ice \U0001f9d1<test>")
    photo_id5 = db.add_photo(str(img_path) + "5", 200, 200)
    db.add_faces_batch([(photo_id5, (10, 10, 50, 50), _emb(5), 0.95)], species="human")
    faces5 = db.get_photo_faces(photo_id5)
    db.assign_face_to_subject(faces5[0].id, sid_weird)

    # An unassigned cluster
    photo_id3 = db.add_photo(str(img_path) + "3", 200, 200)
    db.add_faces_batch([(photo_id3, (10, 10, 50, 50), _emb(3), 0.95)], species="human")
    photo_id4 = db.add_photo(str(img_path) + "4", 200, 200)
    db.add_faces_batch([(photo_id4, (10, 10, 50, 50), _emb(3), 0.95)], species="human")
    f3 = db.get_photo_faces(photo_id3)
    f4 = db.get_photo_faces(photo_id4)
    db.update_cluster_ids({f3[0].id: 100, f4[0].id: 100})

    db.close()

    app = create_app(str(db_path))
    port = 18787
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Wait for server to be ready
    for _ in range(50):
        try:
            import httpx

            httpx.get(f"http://127.0.0.1:{port}/")
            break
        except httpx.ConnectError:
            time.sleep(0.1)
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
