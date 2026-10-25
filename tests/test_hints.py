"""Tests for ritrova.hints module (renamed from services in ADR-012 §M3)."""

import numpy as np

from ritrova.db import FaceDB
from ritrova.hints import (
    compute_cluster_hint,
    compute_singleton_hints,
)

from ._postgres import PostgresTestCase


def _make_embedding(seed: int = 42, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


async def _add_source_with_finding(
    db: FaceDB,
    path: str = "/test/photo.jpg",
    species: str = "human",
    embedding: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[int, int]:
    pid = await db.add_source(path, width=100, height=100)
    # Use 768-dim for pet species to match real SigLIP embeddings
    dim = 768 if species in FaceDB.PET_SPECIES else 512
    emb = embedding if embedding is not None else _make_embedding(seed, dim=dim)
    from ._helpers import add_findings

    await add_findings(db, [(pid, (10, 10, 50, 50), emb, 0.95)], species=species)
    findings = await db.get_source_findings(pid)
    return pid, findings[0].id


class TestComputeClusterHint(PostgresTestCase):
    async def test_best_match(self) -> None:
        emb = _make_embedding(seed=1)
        # Create a subject with similar embedding
        sid = await self.db.create_subject("Alice")
        _, fid = await _add_source_with_finding(self.db, path="/a.jpg", embedding=emb, seed=1)
        await self.db.assign_finding_to_subject(fid, sid)

        # Create a cluster with the same embedding
        _, fid2 = await _add_source_with_finding(self.db, path="/b.jpg", embedding=emb, seed=1)
        _, fid3 = await _add_source_with_finding(self.db, path="/c.jpg", embedding=emb, seed=1)
        await self.db.update_cluster_ids({fid2: 10, fid3: 10})

        result = await compute_cluster_hint(self.db, 10)
        assert result is not None
        assert result["name"] == "Alice"
        assert result["subject_id"] == sid
        assert result["sim"] > 90

    async def test_no_subjects(self) -> None:
        _, fid1 = await _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = await _add_source_with_finding(self.db, path="/b.jpg")
        await self.db.update_cluster_ids({fid1: 10, fid2: 10})
        result = await compute_cluster_hint(self.db, 10)
        assert result is None

    async def test_empty_cluster(self) -> None:
        result = await compute_cluster_hint(self.db, 999)
        assert result is None

    async def test_species_filtering(self) -> None:
        # Subject with human findings
        sid = await self.db.create_subject("Alice", kind="person")
        _, fid = await _add_source_with_finding(self.db, path="/a.jpg", species="human", seed=1)
        await self.db.assign_finding_to_subject(fid, sid)

        # Cluster with dog findings — should NOT match human subject
        _, fid2 = await _add_source_with_finding(self.db, path="/b.jpg", species="dog", seed=1)
        _, fid3 = await _add_source_with_finding(self.db, path="/c.jpg", species="dog", seed=1)
        await self.db.update_cluster_ids({fid2: 10, fid3: 10})

        result = await compute_cluster_hint(self.db, 10)
        assert result is None


class TestComputeSingletonHints(PostgresTestCase):
    async def test_compute_hints(self) -> None:
        emb = _make_embedding(seed=1)
        sid = await self.db.create_subject("Alice")
        _, fid1 = await _add_source_with_finding(self.db, path="/a.jpg", embedding=emb)
        await self.db.assign_finding_to_subject(fid1, sid)

        # Singleton finding with same embedding
        _, fid2 = await _add_source_with_finding(self.db, path="/b.jpg", embedding=emb)
        finding = await self.db.get_finding(fid2)
        assert finding is not None
        subject = await self.db.get_subject(sid)
        assert subject is not None

        hints = await compute_singleton_hints(self.db, [finding], "person")
        assert fid2 in hints
        assert hints[fid2]["name"] == "Alice"
        assert hints[fid2]["sim"] > 90

    async def test_no_subjects(self) -> None:
        _, fid = await _add_source_with_finding(self.db)
        finding = await self.db.get_finding(fid)
        assert finding is not None
        hints = await compute_singleton_hints(self.db, [finding], "person")
        assert hints == {}


class TestGetSubjectsByKind(PostgresTestCase):
    """Tests for db.get_subjects_by_kind."""

    async def test_person_filter(self) -> None:
        sid_human = await self.db.create_subject("Alice", kind="person")
        _, fid = await _add_source_with_finding(self.db, path="/a.jpg", species="human")
        await self.db.assign_finding_to_subject(fid, sid_human)

        sid_dog = await self.db.create_subject("Figaro", kind="pet")
        _, fid2 = await _add_source_with_finding(self.db, path="/b.jpg", species="dog")
        await self.db.assign_finding_to_subject(fid2, sid_dog)

        person_subjects = await self.db.get_subjects_by_kind("person")
        assert len(person_subjects) == 1
        assert person_subjects[0].name == "Alice"

    async def test_pet_filter(self) -> None:
        sid_human = await self.db.create_subject("Alice", kind="person")
        _, fid = await _add_source_with_finding(self.db, path="/a.jpg", species="human")
        await self.db.assign_finding_to_subject(fid, sid_human)

        sid_dog = await self.db.create_subject("Figaro", kind="pet")
        _, fid2 = await _add_source_with_finding(self.db, path="/b.jpg", species="dog")
        await self.db.assign_finding_to_subject(fid2, sid_dog)

        pet_subjects = await self.db.get_subjects_by_kind("pet")
        assert len(pet_subjects) == 1
        assert pet_subjects[0].name == "Figaro"
