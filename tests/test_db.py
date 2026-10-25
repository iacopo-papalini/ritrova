"""Comprehensive tests for ritrova.db module."""

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from uuid import UUID

import asyncpg
import numpy as np
import pytest

from ritrova.db import FaceDB
from ritrova.db.path_metadata import parse_source_path_metadata

# A syntactically valid id that no row can have.
from ._helpers import MISSING_ID as _MISSING_ID
from ._postgres import PostgresTestCase


class TestSourceOperations(PostgresTestCase):
    async def test_add_and_get_source(self) -> None:
        pid = await self.db.add_source(
            "/test/photo.jpg", width=1920, height=1080, taken_at=datetime(2024, 1, 1)
        )
        source = await self.db.get_source(pid)
        assert source is not None
        assert source.file_path == "/test/photo.jpg"
        assert source.width == 1920
        assert source.height == 1080
        assert source.taken_at == datetime(2024, 1, 1)

    async def test_get_nonexistent_source(self) -> None:
        assert await self.db.get_source(_MISSING_ID) is None

    async def test_is_scanned_human(self) -> None:
        assert not await self.db.is_scanned("/test/photo.jpg", "human")
        sid = await self.db.add_source("/test/photo.jpg", width=100, height=100)
        await self.db.record_scan(sid, "human")
        assert await self.db.is_scanned("/test/photo.jpg", "human")

    async def test_is_scanned_pet(self) -> None:
        assert not await self.db.is_scanned("/test/photo.jpg", "pet")
        sid = await self.db.add_source("/test/photo.jpg", width=100, height=100)
        await self.db.record_scan(sid, "pet")
        assert await self.db.is_scanned("/test/photo.jpg", "pet")

    async def test_is_scanned_video(self) -> None:
        assert not await self.db.is_scanned("/test/video.mp4", "human")
        sid = await self.db.add_source(
            "/test/video.mp4", source_type="video", width=100, height=100
        )
        await self.db.record_scan(sid, "human")
        assert await self.db.is_scanned("/test/video.mp4", "human")

    async def test_source_count(self) -> None:
        assert await self.db.get_source_count() == 0
        await self.db.add_source("/test/a.jpg", width=100, height=100)
        await self.db.add_source("/test/b.jpg", width=100, height=100)
        assert await self.db.get_source_count() == 2

    async def test_duplicate_source_path_raises(self) -> None:
        await self.db.add_source("/test/photo.jpg", width=100, height=100)
        with pytest.raises(Exception):  # noqa: B017
            await self.db.add_source("/test/photo.jpg", width=200, height=200)


class TestPathMetadata(PostgresTestCase):
    def test_parser_prefers_directory_when_filename_date_is_placeholder_conflict(self) -> None:
        meta = parse_source_path_metadata("2004/2004-07.Laurea/2002_01_01_14_04_46.jpg")

        assert meta.date_text == "2004-07"
        assert meta.date_precision == "month"
        assert meta.date_source == "directory"
        assert meta.date_conflict is True
        assert meta.filename_date == "2002-01-01"
        assert meta.directory_date == "2004-07"
        assert meta.path_tags == {"laurea"}

    def test_parser_uses_precise_filename_when_it_agrees_with_directory(self) -> None:
        meta = parse_source_path_metadata("2021/2021-06-19.Stiappa/2021_06_19_12_00_00.jpg")

        assert meta.date_text == "2021-06-19"
        assert meta.date_precision == "day"
        assert meta.date_source == "filename"
        assert meta.date_conflict is False
        assert "stiappa" in meta.path_tags

    def test_parser_uses_filename_year_when_no_directory_date_exists(self) -> None:
        meta = parse_source_path_metadata("_preistoria/Michele - Foto/2006 - BiscioVSmostro.jpg")

        assert meta.date_text == "2006"
        assert meta.date_precision == "year"
        assert meta.date_source == "filename"
        assert {"preistoria", "michele", "bisciovsmostro"} <= meta.path_tags

    def test_parser_uses_compact_directory_date_for_legacy_video_paths(self) -> None:
        meta = parse_source_path_metadata(
            "_preistoria/Michele - foto 'nuove'/20081226 - Cena Pizzeria Castella/MVI_7904.AVI"
        )

        assert meta.date_text == "2008-12-26"
        assert meta.date_precision == "day"
        assert meta.date_source == "directory"
        assert {"preistoria", "michele", "nuove", "cena", "pizzeria", "castella"} <= (
            meta.path_tags
        )
        assert "mvi" not in meta.path_tags

    def test_parser_falls_back_to_exif_when_path_has_no_date(self) -> None:
        meta = parse_source_path_metadata(
            "Domotz/portrait.jpg", taken_at=datetime(2018, 4, 3, 10, 11, 12)
        )

        assert meta.date_text == "2018-04-03"
        assert meta.date_source == "exif"
        assert meta.path_tags == {"domotz", "portrait"}

    async def test_add_source_caches_path_metadata(self) -> None:
        source_id = await self.db.add_source("2022/2022-10_LuccaComics/PXL_20221031_120000000.jpg")

        meta = await self.db.get_source_path_metadata(source_id)

        assert meta is not None
        assert meta.date_text == "2022-10-31"
        assert meta.date_source == "filename"
        assert "luccacomics" in meta.path_tags
        assert "pxl" not in meta.path_tags

    async def test_backfill_source_path_metadata(self) -> None:
        source_id = await self.db.add_source("varie/nonni_1.jpeg", taken_at=datetime(2010, 5, 1))
        await self.db.execute("DELETE FROM source_path_metadata WHERE source_id = ?", source_id)

        assert await self.db.backfill_source_path_metadata(source_type="photo") == 1
        meta = await self.db.get_source_path_metadata(source_id)

        assert meta is not None
        assert meta.date_text == "2010-05-01"
        assert meta.date_source == "exif"
        assert meta.path_tags == {"varie", "nonni"}


def _make_embedding(dim: int = 512) -> np.ndarray:
    """Create a random normalized embedding."""
    v = np.random.default_rng(42).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


async def _add_source_with_finding(
    db: FaceDB,
    path: str = "/test/photo.jpg",
    species: str = "human",
    embedding: np.ndarray | None = None,
) -> tuple[int, int]:
    """Helper: add a source + one finding, return (source_id, finding_id)."""
    from ._helpers import add_findings

    pid = await db.add_source(path, width=100, height=100)
    dim = 768 if species in FaceDB.PET_SPECIES else 512
    emb = embedding if embedding is not None else _make_embedding(dim=dim)
    await add_findings(db, [(pid, (10, 10, 50, 50), emb, 0.95)], species=species)
    findings = await db.get_source_findings(pid)
    return pid, findings[0].id


class TestSourceSearch(PostgresTestCase):
    async def test_search_sources_filters_by_path_tags_and_path_date(self) -> None:
        matching = await self.db.add_source("2022/2022-10_LuccaComics/PXL_20221031_120000000.jpg")
        await self.db.add_source("2022/2022-11_Roma/IMG_20221102.jpg")
        await self.db.add_source("2021/2021-10_LuccaComics/IMG_20211031.jpg")

        sources = await self.db.search_sources(
            path_tags={"luccaComics"},
            date_from="2022-01-01",
            date_to="2022-12-31",
        )

        assert [s.id for s in sources] == [matching]
        assert (
            await self.db.count_search_sources(
                path_tags={"luccaComics"},
                date_from="2022-01-01",
                date_to="2022-12-31",
            )
            == 1
        )

    async def test_search_sources_sorts_unknown_dates_last(self) -> None:
        unknown = await self.db.add_source("varie/nonni_1.jpeg")
        newest = await self.db.add_source("2026/2026-04-25_Yuki/PXL_20260425_083901244.jpg")
        older = await self.db.add_source("2020/2020-12-29.Libera/20201229_165654.jpg")

        sources = await self.db.search_sources()

        assert [s.id for s in sources] == [newest, older, unknown]

    async def test_search_sources_combines_subject_and_path_tag_filters(self) -> None:
        alice = await self.db.create_subject("Alice")
        matching_source, matching_finding = await _add_source_with_finding(
            self.db, "2023/2023-08_Amsterdam/IMG_20230801.jpg"
        )
        other_source, other_finding = await _add_source_with_finding(
            self.db, "2023/2023-08_Roma/IMG_20230802.jpg"
        )
        await self.db.assign_finding_to_subject(matching_finding, alice)
        await self.db.assign_finding_to_subject(other_finding, alice)

        sources = await self.db.search_sources(subject_ids=[alice], path_tags={"amsterdam"})

        assert [s.id for s in sources] == [matching_source]


class TestPrintSelection(PostgresTestCase):
    async def test_add_remove_and_reorder_print_selection(self) -> None:
        first = await self.db.add_source("2024/2024-01-01.A/a.jpg")
        second = await self.db.add_source("2024/2024-01-02.B/b.jpg")

        assert await self.db.add_to_print_selection(first) == 1
        assert await self.db.add_to_print_selection(second) == 2
        assert await self.db.add_to_print_selection(first) == 1
        assert await self.db.get_print_selection_ids() == [first, second]

        await self.db.reorder_print_selection([second, first])
        assert await self.db.get_print_selection_ids() == [second, first]

        await self.db.remove_from_print_selection(second)
        assert await self.db.get_print_selection_ids() == [first]
        assert (await self.db.list_print_selection())[0].position == 1

    async def test_print_selection_rejects_videos(self) -> None:
        video = await self.db.add_source("2024/video.mp4", source_type="video")

        with pytest.raises(ValueError, match="Only photos"):
            await self.db.add_to_print_selection(video)


class TestFindingOperations(PostgresTestCase):
    async def test_add_and_get_finding(self) -> None:
        from ._helpers import add_findings

        pid = await self.db.add_source("/test/photo.jpg", width=100, height=100)
        emb = _make_embedding()
        await add_findings(self.db, [(pid, (10, 20, 30, 40), emb, 0.95)])
        findings = await self.db.get_source_findings(pid)
        assert len(findings) == 1
        f = findings[0]
        assert f.bbox_x == 10
        assert f.bbox_y == 20
        assert f.bbox_w == 30
        assert f.bbox_h == 40
        assert f.confidence == pytest.approx(0.95)
        assert f.species == "human"

    async def test_get_finding_by_id(self) -> None:
        _, fid = await _add_source_with_finding(self.db)
        finding = await self.db.get_finding(fid)
        assert finding is not None
        assert finding.id == fid

    async def test_get_nonexistent_finding(self) -> None:
        assert await self.db.get_finding(_MISSING_ID) is None

    async def test_embedding_round_trip(self) -> None:
        emb = _make_embedding()
        _, fid = await _add_source_with_finding(self.db, embedding=emb)
        finding = await self.db.get_finding(fid)
        assert finding is not None
        np.testing.assert_array_almost_equal(finding.embedding, emb)

    async def test_finding_count(self) -> None:
        assert await self.db.get_finding_count() == 0
        await _add_source_with_finding(self.db, path="/a.jpg")
        await _add_source_with_finding(self.db, path="/b.jpg")
        assert await self.db.get_finding_count() == 2

    async def test_get_all_embeddings_excludes_dismissed(self) -> None:
        _, fid1 = await _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = await _add_source_with_finding(self.db, path="/b.jpg")
        assert len(await self.db.get_all_embeddings()) == 2
        await self.db.dismiss_findings([fid1])
        assert len(await self.db.get_all_embeddings()) == 1

    async def test_get_all_embeddings_species_filter(self) -> None:
        await _add_source_with_finding(self.db, path="/a.jpg", species="human")
        await _add_source_with_finding(self.db, path="/b.jpg", species="dog")
        assert len(await self.db.get_all_embeddings(species="human")) == 1
        assert len(await self.db.get_all_embeddings(species="dog")) == 1

    async def test_cluster_operations(self) -> None:
        _, fid1 = await _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = await _add_source_with_finding(self.db, path="/b.jpg")
        _, fid3 = await _add_source_with_finding(self.db, path="/c.jpg")

        await self.db.update_cluster_ids({fid1: 1, fid2: 1, fid3: 2})
        assert sorted(await self.db.get_cluster_ids()) == [1, 2]
        assert await self.db.get_cluster_finding_count(1) == 2
        assert await self.db.get_cluster_finding_count(2) == 1

        findings = await self.db.get_cluster_findings(1)
        assert len(findings) == 2

        await self.db.clear_clusters()
        assert await self.db.get_cluster_ids() == []

    async def test_clear_clusters_species_scoped(self) -> None:
        _, fid_human = await _add_source_with_finding(self.db, path="/h.jpg", species="human")
        _, fid_dog = await _add_source_with_finding(self.db, path="/d.jpg", species="dog")
        await self.db.update_cluster_ids({fid_human: 1, fid_dog: 2})

        # Clear only human clusters
        await self.db.clear_clusters(species="human")
        human_finding = await self.db.get_finding(fid_human)
        dog_finding = await self.db.get_finding(fid_dog)
        assert human_finding is not None
        assert human_finding.cluster_id is None
        assert dog_finding is not None
        assert dog_finding.cluster_id == 2

    async def test_get_cluster_finding_ids(self) -> None:
        _, fid1 = await _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = await _add_source_with_finding(self.db, path="/b.jpg")
        await self.db.update_cluster_ids({fid1: 5, fid2: 5})
        ids = await self.db.get_cluster_finding_ids(5)
        assert sorted(ids) == sorted([fid1, fid2])

    async def test_singleton_findings(self) -> None:
        _, fid1 = await _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = await _add_source_with_finding(self.db, path="/b.jpg")
        # Both unclustered — both are singletons
        singletons = await self.db.get_singleton_findings()
        assert len(singletons) == 2

        # Put one in a cluster of size 2 — not a singleton anymore
        _, fid3 = await _add_source_with_finding(self.db, path="/c.jpg")
        await self.db.update_cluster_ids({fid1: 1, fid3: 1})
        singletons = await self.db.get_singleton_findings()
        # fid2 is unclustered, fid1/fid3 in cluster of 2 — only fid2 is singleton
        assert len(singletons) == 1
        assert singletons[0].id == fid2

    async def test_singleton_count(self) -> None:
        await _add_source_with_finding(self.db, path="/a.jpg")
        await _add_source_with_finding(self.db, path="/b.jpg")
        assert await self.db.get_singleton_count() == 2

    async def test_dismiss_findings(self) -> None:
        _, fid = await _add_source_with_finding(self.db)
        subject_id = await self.db.create_subject("Test")
        await self.db.assign_finding_to_subject(fid, subject_id)
        await self.db.update_cluster_ids({fid: 1})

        await self.db.dismiss_findings([fid])
        finding = await self.db.get_finding(fid)
        assert finding is not None
        assert finding.subject_id is None
        assert finding.cluster_id is None

    async def test_unassign_finding(self) -> None:
        _, fid = await _add_source_with_finding(self.db)
        subject_id = await self.db.create_subject("Test")
        await self.db.assign_finding_to_subject(fid, subject_id)
        finding = await self.db.get_finding(fid)
        assert finding is not None
        assert finding.subject_id == subject_id

        await self.db.unassign_finding(fid)
        finding = await self.db.get_finding(fid)
        assert finding is not None
        assert finding.subject_id is None

    async def test_exclude_findings(self) -> None:
        _, fid1 = await _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = await _add_source_with_finding(self.db, path="/b.jpg")
        await self.db.update_cluster_ids({fid1: 1, fid2: 1})

        await self.db.exclude_findings([fid1], cluster_id=1)
        finding = await self.db.get_finding(fid1)
        assert finding is not None
        assert finding.cluster_id is None
        # fid2 still in cluster
        finding2 = await self.db.get_finding(fid2)
        assert finding2 is not None
        assert finding2.cluster_id == 1

    async def test_merge_clusters(self) -> None:
        _, fid1 = await _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = await _add_source_with_finding(self.db, path="/b.jpg")
        await self.db.update_cluster_ids({fid1: 1, fid2: 2})

        await self.db.merge_clusters(source_id=2, target_id=1)
        finding = await self.db.get_finding(fid2)
        assert finding is not None
        assert finding.cluster_id == 1

    async def test_split_cluster_findings(self) -> None:
        _, fid1 = await _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = await _add_source_with_finding(self.db, path="/b.jpg")
        await self.db.update_cluster_ids({fid1: 1, fid2: 1})

        result = await self.db.split_cluster_findings(1, [fid1])
        assert result is not None
        new_cluster_id, moved_ids = result
        assert moved_ids == [fid1]
        assert await self.db.get_cluster_membership(fid1) == new_cluster_id
        assert await self.db.get_cluster_membership(fid2) == 1


class TestSubjectOperations(PostgresTestCase):
    async def test_create_subject(self) -> None:
        sid = await self.db.create_subject("Alice")
        subject = await self.db.get_subject(sid)
        assert subject is not None
        assert subject.name == "Alice"
        assert subject.face_count == 0

    async def test_create_subject_idempotent(self) -> None:
        sid1 = await self.db.create_subject("Alice")
        sid2 = await self.db.create_subject("Alice")
        assert sid1 == sid2

    async def test_get_subjects(self) -> None:
        await self.db.create_subject("Bob")
        await self.db.create_subject("Alice")
        subjects = await self.db.get_subjects()
        assert len(subjects) == 2
        # Ordered by name
        assert subjects[0].name == "Alice"
        assert subjects[1].name == "Bob"

    async def test_rename_subject(self) -> None:
        sid = await self.db.create_subject("Alice")
        await self.db.rename_subject(sid, "Alicia")
        subject = await self.db.get_subject(sid)
        assert subject is not None
        assert subject.name == "Alicia"

    async def test_assign_finding_to_subject(self) -> None:
        _, fid = await _add_source_with_finding(self.db)
        sid = await self.db.create_subject("Alice")
        await self.db.assign_finding_to_subject(fid, sid)
        subject = await self.db.get_subject(sid)
        assert subject is not None
        assert subject.face_count == 1

    async def test_assign_cluster_to_subject(self) -> None:
        _, fid1 = await _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = await _add_source_with_finding(self.db, path="/b.jpg")
        await self.db.update_cluster_ids({fid1: 1, fid2: 1})
        sid = await self.db.create_subject("Alice")
        await self.db.assign_cluster_to_subject(1, sid)
        subject = await self.db.get_subject(sid)
        assert subject is not None
        assert subject.face_count == 2

    async def test_merge_subjects(self) -> None:
        sid_a = await self.db.create_subject("Alice")
        sid_b = await self.db.create_subject("Alice2")
        _, fid = await _add_source_with_finding(self.db)
        await self.db.assign_finding_to_subject(fid, sid_a)

        await self.db.merge_subjects(sid_a, sid_b)
        # Source deleted
        assert await self.db.get_subject(sid_a) is None
        # Target has the finding
        subject = await self.db.get_subject(sid_b)
        assert subject is not None
        assert subject.face_count == 1

    async def test_delete_subject(self) -> None:
        sid = await self.db.create_subject("Alice")
        _, fid = await _add_source_with_finding(self.db)
        await self.db.assign_finding_to_subject(fid, sid)

        await self.db.delete_subject(sid)
        assert await self.db.get_subject(sid) is None
        # Finding should be unassigned, not deleted
        finding = await self.db.get_finding(fid)
        assert finding is not None
        assert finding.subject_id is None

    async def test_search_subjects(self) -> None:
        await self.db.create_subject("Alice")
        await self.db.create_subject("Bob")
        await self.db.create_subject("Alicia")
        results = await self.db.search_subjects("Ali")
        assert len(results) == 2

    async def test_get_subject_findings(self) -> None:
        sid = await self.db.create_subject("Alice")
        _, fid = await _add_source_with_finding(self.db)
        await self.db.assign_finding_to_subject(fid, sid)
        findings = await self.db.get_subject_findings(sid)
        assert len(findings) == 1
        assert findings[0].id == fid

    async def test_get_subject_sources(self) -> None:
        sid = await self.db.create_subject("Alice")
        source_id, fid = await _add_source_with_finding(self.db)
        await self.db.assign_finding_to_subject(fid, sid)
        sources = await self.db.get_subject_sources(sid)
        assert len(sources) == 1
        assert sources[0].id == source_id

    async def test_record_scan_returns_id(self) -> None:
        sid = await self.db.add_source("/test/p.jpg", width=10, height=10)
        scan_id = await self.db.record_scan(sid, "human")
        assert isinstance(scan_id, UUID)

    async def test_add_findings_batch_requires_scan_id_keyword(self) -> None:
        sid = await self.db.add_source("/test/p.jpg", width=10, height=10)
        emb = _make_embedding()
        # Positional scan_id is rejected (kwarg-only).
        with pytest.raises(TypeError):
            await self.db.add_findings_batch([(sid, (0, 0, 10, 10), emb, 0.9)], 1)
        # Missing entirely is rejected too.
        with pytest.raises(TypeError):
            await self.db.add_findings_batch([(sid, (0, 0, 10, 10), emb, 0.9)])

    async def test_delete_scan_cascades_to_findings(self) -> None:
        sid = await self.db.add_source("/test/p.jpg", width=10, height=10)
        scan_id = await self.db.record_scan(sid, "human")
        await self.db.add_findings_batch(
            [(sid, (0, 0, 10, 10), _make_embedding(), 0.9)], scan_id=scan_id
        )
        assert len(await self.db.get_source_findings(sid)) == 1

        result = await self.db.delete_scan(scan_id)
        assert result["deleted_findings"] == 1
        assert result["deleted_with_assignments"] == 0
        assert await self.db.get_source_findings(sid) == []
        # Scan row is gone.
        assert await self.db.fetchrow("SELECT 1 FROM scans WHERE id = ?", scan_id) is None

    async def test_delete_scan_counts_manual_assignments(self) -> None:
        subject = await self.db.create_subject("Alice")
        sid = await self.db.add_source("/test/p.jpg", width=10, height=10)
        scan_id = await self.db.record_scan(sid, "human")
        await self.db.add_findings_batch(
            [
                (sid, (0, 0, 10, 10), _make_embedding(), 0.9),
                (sid, (5, 5, 10, 10), _make_embedding(), 0.9),
            ],
            scan_id=scan_id,
        )
        findings = await self.db.get_source_findings(sid)
        await self.db.assign_finding_to_subject(findings[0].id, subject)

        result = await self.db.delete_scan(scan_id)
        assert result["deleted_findings"] == 2
        assert result["deleted_with_assignments"] == 1

    async def test_delete_scan_unknown_id_raises(self) -> None:
        with pytest.raises(ValueError, match="No such scan"):
            await self.db.delete_scan(_MISSING_ID)

    async def test_find_scans_by_id_and_pattern(self) -> None:
        s1 = await self.db.add_source("/2024/jan/a.jpg", width=10, height=10)
        s2 = await self.db.add_source("/2024/feb/b.jpg", width=10, height=10)
        s3 = await self.db.add_source("/2025/jan/c.jpg", width=10, height=10)
        sc1 = await self.db.record_scan(s1, "human")
        sc2 = await self.db.record_scan(s2, "human")
        await self.db.record_scan(s3, "human")

        # Pattern only — both 2024 sources match.
        rows = await self.db.find_scans(source_pattern="*2024*")
        assert {r["id"] for r in rows} == {sc1, sc2}

        # ID only — the single scan.
        rows = await self.db.find_scans(scan_id=sc1)
        assert len(rows) == 1 and rows[0]["id"] == sc1
        assert rows[0]["finding_count"] == 0

        # Both filters — intersection.
        rows = await self.db.find_scans(scan_id=sc1, source_pattern="*feb*")
        assert rows == []  # sc1 is on /jan/, doesn't match /feb/

        # No filters — everything.
        rows = await self.db.find_scans()
        assert len(rows) == 3

    async def test_get_subject_sources_with_findings_videos(self) -> None:
        """Filter to videos only; return each video paired with the subject's findings on it.

        Regression guard: previously `SELECT s.*, f.*` collided on `id` and `dict(sqlite3.Row)`
        kept only the first one (source's), so every Finding inherited the source's id. The
        guard below asserts the **exact set** of finding ids per source matches what
        `get_source_findings` returns (which is unambiguous), and uses two findings per video
        so an off-by-one or id-substitution bug is detectable even when ids happen to align.
        """
        sid = await self.db.create_subject("Alice")

        # Two video sources, each with TWO findings assigned to Alice. Multiple findings per
        # source means the buggy "all findings inherit source.id" output would collapse to a
        # single finding (set len == 1) instead of the expected 2.
        v1_sid = await self.db.add_source(
            "/test/movies/wedding.mp4", source_type="video", width=100, height=100
        )
        v2_sid = await self.db.add_source(
            "/test/movies/party.mp4", source_type="video", width=100, height=100
        )
        from ._helpers import add_findings

        await add_findings(
            self.db,
            [
                (v1_sid, (0, 0, 10, 10), _make_embedding(), 0.9),
                (v1_sid, (5, 5, 10, 10), _make_embedding(), 0.9),
            ],
            species="human",
            frame_path="tmp/frames/v1.jpg",
        )
        await add_findings(
            self.db,
            [
                (v2_sid, (0, 0, 10, 10), _make_embedding(), 0.9),
                (v2_sid, (5, 5, 10, 10), _make_embedding(), 0.9),
            ],
            species="human",
            frame_path="tmp/frames/v2.jpg",
        )
        for f in await self.db.get_source_findings(v1_sid) + await self.db.get_source_findings(
            v2_sid
        ):
            await self.db.assign_finding_to_subject(f.id, sid)

        # A photo source with the same subject — should be excluded by the type filter.
        photo_sid, photo_fid = await _add_source_with_finding(self.db, path="/test/photos/p.jpg")
        await self.db.assign_finding_to_subject(photo_fid, sid)

        videos = await self.db.get_subject_sources_with_findings(sid, source_type="video")
        assert {src.id for src, _ in videos} == {v1_sid, v2_sid}

        canonical = {
            v1_sid: {f.id for f in await self.db.get_source_findings(v1_sid)},
            v2_sid: {f.id for f in await self.db.get_source_findings(v2_sid)},
        }
        for src, findings in videos:
            assert src.type == "video"
            # Set equality catches both id collapse (would yield fewer findings) and id mixup
            # (would yield ids not present in the canonical per-source lookup).
            assert {f.id for f in findings} == canonical[src.id]
            for f in findings:
                assert f.source_id == src.id
                assert f.frame_path is not None

    async def test_subject_kind_person(self) -> None:
        sid = await self.db.create_subject("Alice", kind="person")
        subject = await self.db.get_subject(sid)
        assert subject is not None
        assert subject.kind == "person"

    async def test_subject_kind_pet(self) -> None:
        sid = await self.db.create_subject("Figaro", kind="pet")
        subject = await self.db.get_subject(sid)
        assert subject is not None
        assert subject.kind == "pet"


class TestSpeciesFilter(PostgresTestCase):
    def test_human_filter(self) -> None:
        clause, params = self.db.species_filter("human")
        assert "species = ?" in clause
        assert params == ("human",)

    def test_pet_filter(self) -> None:
        clause, params = self.db.species_filter("pet")
        assert "IN" in clause
        assert len(params) == len(self.db.PET_SPECIES)

    async def test_unnamed_clusters_species(self) -> None:
        await _add_source_with_finding(self.db, path="/a.jpg", species="human")
        await _add_source_with_finding(self.db, path="/b.jpg", species="human")
        await _add_source_with_finding(self.db, path="/c.jpg", species="dog")
        await _add_source_with_finding(self.db, path="/d.jpg", species="dog")

        # Get finding ids and cluster them
        human_embs = await self.db.get_all_embeddings(species="human")
        dog_embs = await self.db.get_all_embeddings(species="dog")
        await self.db.update_cluster_ids(
            {human_embs[0][0]: 1, human_embs[1][0]: 1, dog_embs[0][0]: 2, dog_embs[1][0]: 2}
        )

        human_clusters = await self.db.get_unnamed_clusters(species="human")
        assert len(human_clusters) == 1
        assert human_clusters[0]["cluster_id"] == 1

        pet_clusters = await self.db.get_unnamed_clusters(species="pet")
        assert len(pet_clusters) == 1
        assert pet_clusters[0]["cluster_id"] == 2


class TestStats(PostgresTestCase):
    async def test_empty_db_stats(self) -> None:
        stats = await self.db.get_stats()
        assert stats["total_sources"] == 0
        assert stats["total_findings"] == 0
        assert stats["total_subjects"] == 0
        assert stats["named_findings"] == 0
        assert stats["unnamed_clusters"] == 0
        assert stats["unclustered_findings"] == 0
        assert stats["dismissed_findings"] == 0

    async def test_populated_stats(self) -> None:
        _, fid1 = await _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = await _add_source_with_finding(self.db, path="/b.jpg")
        _, fid3 = await _add_source_with_finding(self.db, path="/c.jpg")

        sid = await self.db.create_subject("Alice")
        await self.db.assign_finding_to_subject(fid1, sid)
        await self.db.update_cluster_ids({fid2: 1, fid3: 1})

        stats = await self.db.get_stats()
        assert stats["total_sources"] == 3
        assert stats["total_findings"] == 3
        assert stats["total_subjects"] == 1
        assert stats["named_findings"] == 1
        assert stats["unnamed_clusters"] == 1
        assert stats["unclustered_findings"] == 1  # fid1 assigned but unclustered

    async def test_stats_unnamed_clusters_matches_reviewable_cluster_count(self) -> None:
        _, fid1 = await _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = await _add_source_with_finding(self.db, path="/b.jpg")
        _, fid3 = await _add_source_with_finding(self.db, path="/c.jpg")

        await self.db.update_cluster_ids({fid1: 1, fid2: 1, fid3: 2})

        stats = await self.db.get_stats()
        assert stats["unnamed_clusters"] == await self.db.get_unnamed_cluster_count()
        assert stats["unnamed_clusters"] == len(await self.db.get_unnamed_clusters())
        assert stats["unnamed_clusters"] == 1


class TestGetUnclusteredEmbeddings(PostgresTestCase):
    async def test_returns_unclustered_unassigned(self) -> None:
        _, fid1 = await _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = await _add_source_with_finding(self.db, path="/b.jpg")
        _, fid3 = await _add_source_with_finding(self.db, path="/c.jpg")

        # Cluster fid1, assign fid2 to a subject
        await self.db.update_cluster_ids({fid1: 1})
        sid = await self.db.create_subject("Alice")
        await self.db.assign_finding_to_subject(fid2, sid)

        result = await self.db.get_unclustered_embeddings(species="human")
        assert len(result) == 1
        assert result[0][0] == fid3

    async def test_excludes_dismissed(self) -> None:
        _, fid1 = await _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = await _add_source_with_finding(self.db, path="/b.jpg")
        await self.db.dismiss_findings([fid1])
        result = await self.db.get_unclustered_embeddings(species="human")
        assert len(result) == 1
        assert result[0][0] == fid2

    async def test_species_filter(self) -> None:
        await _add_source_with_finding(self.db, path="/a.jpg", species="human")
        await _add_source_with_finding(self.db, path="/b.jpg", species="dog")
        assert len(await self.db.get_unclustered_embeddings(species="human")) == 1
        assert len(await self.db.get_unclustered_embeddings(species="dog")) == 1
        assert len(await self.db.get_unclustered_embeddings(species="pet")) == 1


class TestExport(PostgresTestCase):
    async def test_export_json_structure(self) -> None:
        import json

        sid = await self.db.create_subject("Alice")
        _, fid = await _add_source_with_finding(self.db)
        await self.db.assign_finding_to_subject(fid, sid)

        data = json.loads(await self.db.export_json())
        assert "subjects" in data
        assert "unnamed_findings" in data
        assert len(data["subjects"]) == 1
        assert data["subjects"][0]["name"] == "Alice"
        assert len(data["subjects"][0]["sources"]) == 1

    async def test_export_empty_db(self) -> None:
        import json

        data = json.loads(await self.db.export_json())
        assert data["subjects"] == []
        assert data["unnamed_findings"] == []


class TestPathResolution(TestCase):
    """``resolve_path`` / ``to_relative`` are pure — no database involved."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = (Path(self._tmp.name) / "photos").resolve()
        self.base.mkdir(parents=True, exist_ok=True)
        self.db = FaceDB(None, data_dir=Path(self._tmp.name), base_dir=self.base)  # type: ignore[arg-type]

    async def tearDown(self) -> None:
        await self._tmp.cleanup()

    def test_resolve_relative_path(self) -> None:
        resolved = self.db.resolve_path("subfolder/photo.jpg")
        assert resolved == self.base / "subfolder" / "photo.jpg"

    def test_resolve_rejects_dotdot(self) -> None:
        with pytest.raises(ValueError, match="\\.\\."):
            self.db.resolve_path("../etc/passwd")

    def test_resolve_absolute_path_passthrough(self) -> None:
        """Legacy absolute paths are returned as-is."""
        resolved = self.db.resolve_path("/absolute/path/photo.jpg")
        assert resolved == Path("/absolute/path/photo.jpg")

    def test_to_relative(self) -> None:
        abs_path = str(self.base / "subfolder" / "photo.jpg")
        rel = self.db.to_relative(abs_path)
        assert rel == "subfolder/photo.jpg"

    def test_to_relative_outside_base(self) -> None:
        """Paths outside base_dir are returned as-is."""
        rel = self.db.to_relative("/somewhere/else/photo.jpg")
        assert rel == "/somewhere/else/photo.jpg"

    def test_no_base_dir_passthrough(self) -> None:
        db_no_base = FaceDB(None, data_dir=self.base, base_dir=None)  # type: ignore[arg-type]
        assert db_no_base.resolve_path("photo.jpg") == Path("photo.jpg")
        assert db_no_base.to_relative("/abs/photo.jpg") == "/abs/photo.jpg"


class TestCircles(PostgresTestCase):
    async def test_create_circle_idempotent(self) -> None:
        cid1 = await self.db.create_circle("family")
        cid2 = await self.db.create_circle("family")
        assert cid1 == cid2

    async def test_create_circle_rejects_blank(self) -> None:
        with pytest.raises(ValueError):
            await self.db.create_circle("   ")

    async def test_list_circles_returns_user_created_only(self) -> None:
        await self.db.create_circle("family")
        names = [c.name for c in await self.db.list_circles()]
        # No auto-seeded circles — the user manages the set.
        assert names == ["family"]

    async def test_membership_add_remove_roundtrip(self) -> None:
        sid = await self.db.create_subject("Alice")
        cid = await self.db.create_circle("family", description="nuclear only")
        assert await self.db.add_subject_to_circle(sid, cid) is True
        # second add is a no-op
        assert await self.db.add_subject_to_circle(sid, cid) is False
        circles = await self.db.get_subject_circles(sid)
        assert [c.name for c in circles] == ["family"]
        assert await self.db.remove_subject_from_circle(sid, cid) is True
        assert await self.db.get_subject_circles(sid) == []

    async def test_member_count_reflects_state(self) -> None:
        cid = await self.db.create_circle("friends")
        await self.db.add_subject_to_circle(await self.db.create_subject("A"), cid)
        await self.db.add_subject_to_circle(await self.db.create_subject("B"), cid)
        assert (await self.db.get_circle(cid)).member_count == 2

    async def test_delete_subject_cascades_to_memberships(self) -> None:
        sid = await self.db.create_subject("Alice")
        cid = await self.db.create_circle("family")
        await self.db.add_subject_to_circle(sid, cid)
        await self.db.delete_subject(sid)
        assert (await self.db.get_circle(cid)).member_count == 0

    async def test_delete_circle_cascades_to_memberships(self) -> None:
        sid = await self.db.create_subject("Alice")
        cid = await self.db.create_circle("acquaintances")
        await self.db.add_subject_to_circle(sid, cid)
        await self.db.delete_circle(cid)
        assert await self.db.get_subject_circles(sid) == []

    async def test_subjects_in_any_circle(self) -> None:
        fam = await self.db.create_circle("family")
        acq = await self.db.create_circle("acquaintances")
        a, b, c = (
            await self.db.create_subject("A"),
            await self.db.create_subject("B"),
            await self.db.create_subject("C"),
        )
        await self.db.add_subject_to_circle(a, fam)
        await self.db.add_subject_to_circle(b, acq)
        assert await self.db.subjects_in_any_circle([fam, acq]) == {a, b}
        assert await self.db.subjects_in_any_circle([fam]) == {a}
        assert await self.db.subjects_in_any_circle([]) == set()
        assert c not in await self.db.subjects_in_any_circle([fam, acq])

    async def test_rename_circle(self) -> None:
        cid = await self.db.create_circle("frineds")  # typo
        await self.db.rename_circle(cid, "friends")
        assert (await self.db.get_circle(cid)).name == "friends"


class TestFindingCuration(PostgresTestCase):
    """``Finding.curation`` reifies the (subject_id, exclusion_reason) XOR."""

    async def test_uncurated_when_both_none(self) -> None:
        from ritrova.db.models import Uncurated

        _, fid = await _add_source_with_finding(self.db, path="/a.jpg")
        finding = await self.db.get_finding(fid)
        assert finding is not None
        assert finding.curation == Uncurated()

    async def test_assigned_to_when_subject_set(self) -> None:
        from ritrova.db.models import AssignedTo

        sid = await self.db.create_subject("Alice")
        _, fid = await _add_source_with_finding(self.db, path="/a.jpg")
        await self.db.assign_finding_to_subject(fid, sid)
        finding = await self.db.get_finding(fid)
        assert finding is not None
        assert finding.curation == AssignedTo(subject_id=sid)

    async def test_excluded_when_exclusion_reason_set(self) -> None:
        from ritrova.db.models import Excluded

        _, fid = await _add_source_with_finding(self.db, path="/a.jpg")
        await self.db.set_exclusions([fid], "stranger")
        finding = await self.db.get_finding(fid)
        assert finding is not None
        assert finding.curation == Excluded(reason="stranger")

        # Flip the same finding to not_a_face via the dismiss path.
        await self.db.dismiss_findings([fid])
        finding = await self.db.get_finding(fid)
        assert finding is not None
        assert finding.curation == Excluded(reason="not_a_face")


class TestFindingAssignmentConstraints(PostgresTestCase):
    """XOR CHECK + enum CHECK on finding_assignment."""

    async def test_check_constraint_blocks_both_null(self) -> None:
        _, fid = await _add_source_with_finding(self.db, path="/a.jpg")
        with pytest.raises(asyncpg.exceptions.IntegrityConstraintViolationError):
            await self.db.execute(
                "INSERT INTO finding_assignment(finding_id, subject_id, exclusion_reason, curated_at) "
                "VALUES (?, NULL, NULL, '2026-04-19T00:00:00+00:00')",
                fid,
            )

    async def test_check_constraint_blocks_both_set(self) -> None:
        _, fid = await _add_source_with_finding(self.db, path="/a.jpg")
        sid = await self.db.create_subject("Alice")
        with pytest.raises(asyncpg.exceptions.IntegrityConstraintViolationError):
            await self.db.execute(
                "INSERT INTO finding_assignment(finding_id, subject_id, exclusion_reason, curated_at) "
                "VALUES (?, ?, 'stranger', '2026-04-19T00:00:00+00:00')",
                fid,
                sid,
            )

    async def test_check_constraint_blocks_bad_reason(self) -> None:
        _, fid = await _add_source_with_finding(self.db, path="/a.jpg")
        with pytest.raises(asyncpg.exceptions.IntegrityConstraintViolationError):
            await self.db.execute(
                "INSERT INTO finding_assignment(finding_id, subject_id, exclusion_reason, curated_at) "
                "VALUES (?, NULL, 'bogus', '2026-04-19T00:00:00+00:00')",
                fid,
            )


class TestAssignmentMixin(PostgresTestCase):
    """New helpers for reading/writing finding_assignment and cluster_findings
    (the target API after the Apr 2026 refactor; existing call sites still
    use the old denormalized columns — they'll be switched in Commit C)."""

    # finding_assignment

    async def test_set_subject_then_get_curation(self) -> None:
        _, fid = await _add_source_with_finding(self.db)
        sid = await self.db.create_subject("Alice")
        await self.db.set_subject(fid, sid)
        assert await self.db.get_curation(fid) == (sid, None)

    async def test_set_exclusion_stranger(self) -> None:
        _, fid = await _add_source_with_finding(self.db)
        await self.db.set_exclusion(fid, "stranger")
        assert await self.db.get_curation(fid) == (None, "stranger")

    async def test_set_exclusion_rejects_bad_reason(self) -> None:
        _, fid = await _add_source_with_finding(self.db)
        with pytest.raises(ValueError):
            await self.db.set_exclusion(fid, "bogus")

    async def test_set_subject_overwrites_exclusion(self) -> None:
        """Picking a name on a previously-marked stranger implicitly unmarks."""
        _, fid = await _add_source_with_finding(self.db)
        sid = await self.db.create_subject("Alice")
        await self.db.set_exclusion(fid, "stranger")
        await self.db.set_subject(fid, sid)
        assert await self.db.get_curation(fid) == (sid, None)

    async def test_set_exclusion_overwrites_subject(self) -> None:
        """Marking as stranger clears a prior assignment."""
        _, fid = await _add_source_with_finding(self.db)
        sid = await self.db.create_subject("Alice")
        await self.db.set_subject(fid, sid)
        await self.db.set_exclusion(fid, "stranger")
        assert await self.db.get_curation(fid) == (None, "stranger")

    async def test_get_curation_returns_none_when_uncurated(self) -> None:
        _, fid = await _add_source_with_finding(self.db)
        assert await self.db.get_curation(fid) is None

    async def test_clear_curation(self) -> None:
        _, fid = await _add_source_with_finding(self.db)
        sid = await self.db.create_subject("Alice")
        await self.db.set_subject(fid, sid)
        await self.db.clear_curation(fid)
        assert await self.db.get_curation(fid) is None

    async def test_clear_curations_batch(self) -> None:
        _, fid1 = await _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = await _add_source_with_finding(self.db, path="/b.jpg")
        await self.db.set_exclusion(fid1, "stranger")
        await self.db.set_exclusion(fid2, "not_a_face")
        await self.db.clear_curations([fid1, fid2])
        assert await self.db.get_curation(fid1) is None
        assert await self.db.get_curation(fid2) is None

    async def test_set_exclusions_batch(self) -> None:
        _, fid1 = await _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = await _add_source_with_finding(self.db, path="/b.jpg")
        await self.db.set_exclusions([fid1, fid2], "stranger")
        assert await self.db.get_curation(fid1) == (None, "stranger")
        assert await self.db.get_curation(fid2) == (None, "stranger")

    # cluster_findings

    async def test_set_and_get_cluster_membership(self) -> None:
        _, fid = await _add_source_with_finding(self.db)
        await self.db.set_cluster_memberships({fid: 42})
        assert await self.db.get_cluster_membership(fid) == 42

    async def test_set_cluster_memberships_bulk(self) -> None:
        _, fid1 = await _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = await _add_source_with_finding(self.db, path="/b.jpg")
        await self.db.set_cluster_memberships({fid1: 7, fid2: 9})
        assert await self.db.get_cluster_membership(fid1) == 7
        assert await self.db.get_cluster_membership(fid2) == 9

    async def test_remove_cluster_memberships(self) -> None:
        _, fid = await _add_source_with_finding(self.db)
        await self.db.set_cluster_memberships({fid: 42})
        await self.db.remove_cluster_memberships([fid])
        assert await self.db.get_cluster_membership(fid) is None

    async def test_clear_species_cluster_memberships(self) -> None:
        """Clearing dogs leaves humans alone."""
        _, human_fid = await _add_source_with_finding(self.db, path="/h.jpg", species="human")
        _, dog_fid = await _add_source_with_finding(self.db, path="/d.jpg", species="dog")
        await self.db.set_cluster_memberships({human_fid: 1, dog_fid: 2})
        await self.db.clear_species_cluster_memberships("dog")
        assert await self.db.get_cluster_membership(human_fid) == 1
        assert await self.db.get_cluster_membership(dog_fid) is None

    async def test_merge_cluster_memberships(self) -> None:
        _, fid1 = await _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = await _add_source_with_finding(self.db, path="/b.jpg")
        await self.db.set_cluster_memberships({fid1: 1, fid2: 1})
        moved = await self.db.merge_cluster_memberships(source_cluster=1, target_cluster=5)
        assert moved == 2
        assert await self.db.get_cluster_membership(fid1) == 5
        assert await self.db.get_cluster_membership(fid2) == 5

    async def test_merge_same_cluster_is_noop(self) -> None:
        _, fid = await _add_source_with_finding(self.db)
        await self.db.set_cluster_memberships({fid: 1})
        assert await self.db.merge_cluster_memberships(1, 1) == 0


class TestSchemaInvariants(PostgresTestCase):
    """What the Alembic baseline guarantees, asserted against the live catalog.

    These replace the old SQLite in-place-migration tests: the migration
    they exercised is gone, but the invariants it existed to establish
    still matter.
    """

    async def test_findings_scan_id_is_not_null(self) -> None:
        nullable = await self.db.fetchval(
            "SELECT is_nullable FROM information_schema.columns "
            "WHERE table_name = 'findings' AND column_name = 'scan_id'"
        )
        assert nullable == "NO"

    async def test_denormalized_finding_columns_are_gone(self) -> None:
        columns = {
            r[0]
            for r in await self.db.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'findings'"
            )
        }
        assert "person_id" not in columns
        assert "cluster_id" not in columns

    async def test_dismissed_findings_table_is_gone(self) -> None:
        present = await self.db.fetchval(
            "SELECT to_regclass('public.dismissed_findings') IS NOT NULL"
        )
        assert present is False

    async def test_surrogate_keys_are_uuid(self) -> None:
        rows = await self.db.fetch(
            "SELECT table_name, data_type FROM information_schema.columns "
            "WHERE column_name = 'id' AND table_schema = 'public' "
            "AND table_name = ANY(?)",
            ["sources", "scans", "subjects", "findings", "descriptions", "circles"],
        )
        assert {r[0] for r in rows} == {
            "sources",
            "scans",
            "subjects",
            "findings",
            "descriptions",
            "circles",
        }
        assert {r[1] for r in rows} == {"uuid"}
