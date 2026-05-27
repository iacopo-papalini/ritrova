"""Tests for HTML analysis reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ritrova.analysis_report import write_analysis_report
from ritrova.db import FaceDB


def _emb(dim: int = 512) -> np.ndarray:
    v = np.ones(dim, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_write_analysis_report_summarizes_sources_species_and_subjects(tmp_path: Path) -> None:
    db = FaceDB(tmp_path / "test.db")
    source_id = db.add_source("2026/photo.jpg", source_type="photo", width=100, height=100)
    scan_id = db.record_scan(source_id, "subjects", detection_strategy="test")
    db.add_findings_batch(
        [
            (source_id, (0, 0, 10, 10), _emb(), 0.9),
            (source_id, (10, 10, 12, 12), _emb(dim=768), 0.8),
        ],
        scan_id=scan_id,
        species="human",
    )
    dog_scan_id = db.record_scan(
        db.add_source("2026/dog.mp4", source_type="video", width=200, height=100),
        "subjects+captions",
        detection_strategy="test",
    )
    dog_source = db.get_source_by_path("2026/dog.mp4")
    assert dog_source is not None
    db.add_findings_batch(
        [(dog_source.id, (0, 0, 20, 20), _emb(dim=768), 0.95)],
        scan_id=dog_scan_id,
        species="dog",
    )
    finding = db.get_source_findings(source_id)[0]
    peer = db.get_source_findings(source_id)[1]
    subject_id = db.create_subject("Alice")
    db.assign_finding_to_subject(finding.id, subject_id)
    db.conn.executemany(
        "INSERT INTO cluster_findings (finding_id, cluster_id) VALUES (?, ?)",
        [(finding.id, 7), (peer.id, 7)],
    )
    db.conn.commit()

    report_path = tmp_path / "report.html"
    report = write_analysis_report(db, [scan_id, dog_scan_id], report_path)

    assert report is not None
    assert report.path == report_path
    assert report.source_count == 2
    assert report.finding_count == 3
    assert report.named_subject_count == 1
    html = report_path.read_text()
    assert "Alice" in html
    assert "Likely Known Subjects" in html
    assert "Clustering" in html
    assert "2026/photo.jpg" in html
    assert "2026/dog.mp4" in html
    assert '<span class="value">2</span><span class="muted">Human detections' in html
    assert '<span class="value">1</span><span class="muted">1 dogs, 0 cats' in html
    db.close()


def test_write_analysis_report_returns_none_for_empty_scan_list(tmp_path: Path) -> None:
    db = FaceDB(tmp_path / "test.db")
    assert write_analysis_report(db, [], tmp_path / "report.html") is None
    assert not (tmp_path / "report.html").exists()
    db.close()
