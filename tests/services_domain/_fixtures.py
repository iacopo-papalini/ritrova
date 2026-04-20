"""Shared fixtures for the services_domain unit tests.

Builds a ``FaceDB`` + ``UndoStore`` pair plus a couple of sources + findings,
so every test can exercise the snapshot-then-mutate-then-undo round-trip
without repeating the seed boilerplate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ritrova.db import FaceDB
from ritrova.undo import UndoStore
from tests._helpers import add_findings


def _unit_embedding(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@dataclass
class Fixture:
    db: FaceDB
    undo: UndoStore
    source_a_id: int
    source_b_id: int
    finding_a_id: int
    finding_b_id: int


def build_fixture(db: FaceDB, *, species: str = "human") -> Fixture:
    """Seed two sources each with one finding of the given species."""
    src_a = db.add_source("/test/a.jpg", width=100, height=100)
    src_b = db.add_source("/test/b.jpg", width=100, height=100)
    dim = 768 if species in FaceDB.PET_SPECIES else 512
    add_findings(
        db,
        [(src_a, (10, 10, 50, 50), _unit_embedding(1, dim=dim), 0.95)],
        species=species,
    )
    add_findings(
        db,
        [(src_b, (10, 10, 50, 50), _unit_embedding(2, dim=dim), 0.90)],
        species=species,
    )
    a = db.get_source_findings(src_a)
    b = db.get_source_findings(src_b)
    return Fixture(
        db=db,
        undo=UndoStore(),
        source_a_id=src_a,
        source_b_id=src_b,
        finding_a_id=a[0].id,
        finding_b_id=b[0].id,
    )
