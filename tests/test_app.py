"""Tests for ritrova.app routes and API endpoints."""

from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ritrova.app import create_app
from ritrova.db import FaceDB


def _emb(seed: int = 42, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _add_face(db: FaceDB, path: str, seed: int = 42, species: str = "human") -> tuple[int, int]:
    """Add photo + face, return (photo_id, face_id)."""
    pid = db.add_photo(path, 100, 100)
    db.add_faces_batch([(pid, (10, 10, 50, 50), _emb(seed), 0.95)], species=species)
    faces = db.get_photo_faces(pid)
    return pid, faces[0].id


class TestImageEndpoints(TestCase):
    """Test thumbnail and photo image API endpoints."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db")
        self.app = create_app(str(tmp_path / "test.db"))
        self.client = TestClient(self.app)
        self.tmp = tmp_path

        # Create a real JPEG image for testing
        from PIL import Image

        img = Image.new("RGB", (200, 200), color="red")
        self.image_path = tmp_path / "test_photo.jpg"
        img.save(str(self.image_path), "JPEG")

    def test_face_thumbnail_returns_jpeg(self) -> None:
        photo_id, face_id = _add_face(self.db, str(self.image_path), seed=1)
        resp = self.client.get(f"/api/faces/{face_id}/thumbnail?size=100")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert len(resp.content) > 0

    def test_face_thumbnail_cached(self) -> None:
        photo_id, face_id = _add_face(self.db, str(self.image_path), seed=1)
        # First request generates thumbnail
        resp1 = self.client.get(f"/api/faces/{face_id}/thumbnail?size=100")
        assert resp1.status_code == 200
        # Second request serves from cache
        resp2 = self.client.get(f"/api/faces/{face_id}/thumbnail?size=100")
        assert resp2.status_code == 200
        assert resp1.content == resp2.content

    def test_face_thumbnail_404_unknown_face(self) -> None:
        resp = self.client.get("/api/faces/99999/thumbnail")
        assert resp.status_code == 404

    def test_photo_image_returns_jpeg(self) -> None:
        photo_id, _ = _add_face(self.db, str(self.image_path), seed=1)
        resp = self.client.get(f"/api/photos/{photo_id}/image")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"

    def test_photo_image_404_unknown(self) -> None:
        resp = self.client.get("/api/photos/99999/image")
        assert resp.status_code == 404


class TestMergeSuggestionsAPI(TestCase):
    """Test merge suggestions API endpoint."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db")
        self.app = create_app(str(tmp_path / "test.db"))
        self.client = TestClient(self.app)

    def test_empty_suggestions(self) -> None:
        resp = self.client.get("/api/merge-suggestions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["suggestions"] == []

    def test_cluster_pair_suggestion(self) -> None:
        """Two unnamed clusters with identical embeddings should be suggested."""
        _, fid1 = _add_face(self.db, "/a1.jpg", seed=1)
        _, fid2 = _add_face(self.db, "/a2.jpg", seed=1)
        _, fid3 = _add_face(self.db, "/b1.jpg", seed=1)
        _, fid4 = _add_face(self.db, "/b2.jpg", seed=1)
        self.db.update_cluster_ids({fid1: 10, fid2: 10, fid3: 20, fid4: 20})

        resp = self.client.get("/api/merge-suggestions?min_sim=50")
        data = resp.json()
        assert data["total"] >= 1
        s = data["suggestions"][0]
        assert s["similarity_pct"] > 90
        # Both are unnamed clusters — names should be None
        assert s["name_a"] is None
        assert s["name_b"] is None

    def test_subject_cluster_suggestion_shows_subject_name(self) -> None:
        """A subject-cluster pair should show the subject's name on the subject side only."""
        sid = self.db.create_subject("Alice")
        _, fid1 = _add_face(self.db, "/a1.jpg", seed=1)
        self.db.assign_face_to_subject(fid1, sid)

        _, fid2 = _add_face(self.db, "/b1.jpg", seed=1)
        _, fid3 = _add_face(self.db, "/b2.jpg", seed=1)
        self.db.update_cluster_ids({fid2: 10, fid3: 10})

        resp = self.client.get("/api/merge-suggestions?min_sim=50")
        data = resp.json()
        assert data["total"] >= 1
        s = data["suggestions"][0]
        # Exactly one side should have a name
        names = [s["name_a"], s["name_b"]]
        assert "Alice" in names
        assert None in names

    def test_subject_to_subject_excluded(self) -> None:
        """Two named subjects should NOT appear in suggestions."""
        sid_a = self.db.create_subject("Alice")
        _, fid1 = _add_face(self.db, "/a1.jpg", seed=1)
        self.db.assign_face_to_subject(fid1, sid_a)

        sid_b = self.db.create_subject("Alice2")
        _, fid2 = _add_face(self.db, "/b1.jpg", seed=1)
        self.db.assign_face_to_subject(fid2, sid_b)

        resp = self.client.get("/api/merge-suggestions?min_sim=50")
        data = resp.json()
        assert data["total"] == 0


class TestSubjectsAPI(TestCase):
    """Test /api/subjects/* endpoints used by the typeahead picker."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db")
        self.app = create_app(str(tmp_path / "test.db"))
        self.client = TestClient(self.app)

    def test_all_subjects_empty(self) -> None:
        resp = self.client.get("/api/subjects/all")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_all_subjects_returns_face_id(self) -> None:
        sid = self.db.create_subject("Alice")
        _, fid = _add_face(self.db, "/a.jpg", seed=1)
        self.db.assign_face_to_subject(fid, sid)

        resp = self.client.get("/api/subjects/all")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == sid
        assert data[0]["name"] == "Alice"
        assert data[0]["face_count"] == 1
        assert data[0]["face_id"] == fid

    def test_all_subjects_includes_pets(self) -> None:
        sid_human = self.db.create_subject("Alice", kind="person")
        _, fid_h = _add_face(self.db, "/a.jpg", seed=1, species="human")
        self.db.assign_face_to_subject(fid_h, sid_human)

        sid_pet = self.db.create_subject("Figaro", kind="pet")
        _, fid_p = _add_face(self.db, "/b.jpg__pets", seed=2, species="dog")
        self.db.assign_face_to_subject(fid_p, sid_pet)

        resp = self.client.get("/api/subjects/all")
        data = resp.json()
        names = {d["name"] for d in data}
        assert names == {"Alice", "Figaro"}

    def test_create_subject(self) -> None:
        resp = self.client.post("/api/subjects/create", json={"name": "Bob"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Bob"
        assert data["id"] > 0
        assert data["face_count"] == 0

    def test_create_subject_returns_existing(self) -> None:
        """Creating with an existing name returns the existing subject."""
        self.db.create_subject("Alice")
        resp = self.client.post("/api/subjects/create", json={"name": "Alice"})
        data = resp.json()
        assert data["name"] == "Alice"

    def test_photo_info_includes_gps(self) -> None:
        pid = self.db.add_photo("/test.jpg", 100, 100, latitude=43.77, longitude=11.25)
        resp = self.client.get(f"/api/photos/{pid}/info")
        data = resp.json()
        assert data["latitude"] == pytest.approx(43.77)
        assert data["longitude"] == pytest.approx(11.25)

    def test_photo_info_null_gps(self) -> None:
        pid = self.db.add_photo("/test.jpg", 100, 100)
        resp = self.client.get(f"/api/photos/{pid}/info")
        data = resp.json()
        assert data["latitude"] is None
        assert data["longitude"] is None


class TestTogetherAPI(TestCase):
    """Test /api/together endpoint for multi-subject photo search."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db")
        self.app = create_app(str(tmp_path / "test.db"))
        self.client = TestClient(self.app)

    def test_empty_query(self) -> None:
        resp = self.client.get("/api/together")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_finds_photo_with_both_subjects(self) -> None:
        sid_a = self.db.create_subject("Alice")
        sid_b = self.db.create_subject("Bob")
        # One photo with both faces
        photo_id = self.db.add_photo("/group.jpg", 100, 100)
        self.db.add_faces_batch([(photo_id, (10, 10, 30, 30), _emb(1), 0.9)], species="human")
        self.db.add_faces_batch([(photo_id, (50, 50, 30, 30), _emb(2), 0.9)], species="human")
        faces = self.db.get_photo_faces(photo_id)
        self.db.assign_face_to_subject(faces[0].id, sid_a)
        self.db.assign_face_to_subject(faces[1].id, sid_b)

        resp = self.client.get(f"/api/together?person_ids={sid_a},{sid_b}")
        data = resp.json()
        assert data["total"] == 1
        assert data["photos"][0]["id"] == photo_id

    def test_excludes_photo_with_only_one(self) -> None:
        sid_a = self.db.create_subject("Alice")
        sid_b = self.db.create_subject("Bob")
        # Photo with only Alice
        photo_id = self.db.add_photo("/solo.jpg", 100, 100)
        self.db.add_faces_batch([(photo_id, (10, 10, 30, 30), _emb(1), 0.9)], species="human")
        faces = self.db.get_photo_faces(photo_id)
        self.db.assign_face_to_subject(faces[0].id, sid_a)

        resp = self.client.get(f"/api/together?person_ids={sid_a},{sid_b}")
        assert resp.json()["total"] == 0

    def test_cross_kind_human_and_pet(self) -> None:
        sid_human = self.db.create_subject("Eva", kind="person")
        sid_pet = self.db.create_subject("Figaro", kind="pet")
        photo_id = self.db.add_photo("/family.jpg", 100, 100)
        self.db.add_faces_batch([(photo_id, (10, 10, 30, 30), _emb(1), 0.9)], species="human")
        self.db.add_faces_batch([(photo_id, (50, 50, 30, 30), _emb(2), 0.9)], species="dog")
        faces = self.db.get_photo_faces(photo_id)
        self.db.assign_face_to_subject(faces[0].id, sid_human)
        self.db.assign_face_to_subject(faces[1].id, sid_pet)

        resp = self.client.get(f"/api/together?person_ids={sid_human},{sid_pet}")
        data = resp.json()
        assert data["total"] == 1


class TestNamespaceCollision(TestCase):
    """Verify subject IDs and cluster IDs don't collide in merge suggestions."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db")
        self.app = create_app(str(tmp_path / "test.db"))
        self.client = TestClient(self.app)

    def test_cluster_id_matching_subject_id_not_mislabeled(self) -> None:
        """If cluster_id == subject_id, the cluster should NOT show the subject's name."""
        # Create subject with id=1
        sid = self.db.create_subject("Alice")
        _, fid_alice = _add_face(self.db, "/alice.jpg", seed=1)
        self.db.assign_face_to_subject(fid_alice, sid)

        # Create unnamed cluster with cluster_id=1 (same as subject id!)
        # Use different seed so it's a different embedding from Alice
        _, fid_c1 = _add_face(self.db, "/c1.jpg", seed=99)
        _, fid_c2 = _add_face(self.db, "/c2.jpg", seed=99)
        self.db.update_cluster_ids({fid_c1: sid, fid_c2: sid})  # cluster_id == subject_id

        # Create another unnamed cluster to get a suggestion pair
        _, fid_c3 = _add_face(self.db, "/c3.jpg", seed=99)
        _, fid_c4 = _add_face(self.db, "/c4.jpg", seed=99)
        self.db.update_cluster_ids({fid_c3: 999, fid_c4: 999})

        resp = self.client.get("/api/merge-suggestions?min_sim=50")
        data = resp.json()

        # Find the cluster-cluster suggestion (not the subject-cluster one)
        for s in data["suggestions"]:
            ids = {s["cluster_a"], s["cluster_b"]}
            if ids == {sid, 999}:
                # cluster_id=1 should NOT have name "Alice"
                if s["cluster_a"] == sid:
                    assert s["name_a"] is None, (
                        f"Cluster {sid} incorrectly labeled as '{s['name_a']}' "
                        f"due to subject-id/cluster-id collision"
                    )
                if s["cluster_b"] == sid:
                    assert s["name_b"] is None, (
                        f"Cluster {sid} incorrectly labeled as '{s['name_b']}' "
                        f"due to subject-id/cluster-id collision"
                    )
                break
        else:
            pytest.fail("Expected cluster-cluster suggestion not found")
