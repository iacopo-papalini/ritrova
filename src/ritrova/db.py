"""SQLite database for face recognition data."""

import contextlib
import functools
import json
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import numpy as np

P = ParamSpec("P")
R = TypeVar("R")


def _locked(method: Callable[..., R]) -> Callable[..., R]:
    """Decorator: hold the DB lock for the entire method call."""

    @functools.wraps(method)
    def wrapper(self: FaceDB, *args: Any, **kwargs: Any) -> R:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


@dataclass
class Photo:
    id: int
    file_path: str
    width: int
    height: int
    taken_at: str | None
    scanned_at: str
    video_path: str | None = None
    latitude: float | None = None
    longitude: float | None = None


@dataclass
class Face:
    id: int
    photo_id: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    embedding: np.ndarray
    person_id: int | None
    cluster_id: int | None
    confidence: float
    detected_at: str
    species: str = "human"


@dataclass
class Subject:
    id: int
    name: str
    kind: str = "person"
    face_count: int = 0


class FaceDB:
    def __init__(self, db_path: str | Path, base_dir: str | Path | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.base_dir: Path | None = Path(base_dir).resolve() if base_dir else None
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                taken_at TEXT,
                scanned_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS subjects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'person',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
                bbox_x INTEGER NOT NULL,
                bbox_y INTEGER NOT NULL,
                bbox_w INTEGER NOT NULL,
                bbox_h INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                person_id INTEGER REFERENCES subjects(id) ON DELETE SET NULL,
                cluster_id INTEGER,
                confidence REAL NOT NULL DEFAULT 0.0,
                detected_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id);
            CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);
            CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id);
            CREATE INDEX IF NOT EXISTS idx_photos_path ON photos(file_path);

            -- Track dismissed (non-face) entries so they're excluded everywhere
            CREATE TABLE IF NOT EXISTS dismissed_faces (
                face_id INTEGER PRIMARY KEY REFERENCES faces(id) ON DELETE CASCADE
            );
        """)
        # Migrations (idempotent — errors suppressed if already applied)
        for migration in [
            "ALTER TABLE photos ADD COLUMN video_path TEXT",
            "ALTER TABLE faces ADD COLUMN species TEXT NOT NULL DEFAULT 'human'",
            "ALTER TABLE photos ADD COLUMN latitude REAL",
            "ALTER TABLE photos ADD COLUMN longitude REAL",
            # persons -> subjects migration (for DBs created before the rename)
            "ALTER TABLE persons RENAME TO subjects",
            "ALTER TABLE subjects ADD COLUMN kind TEXT NOT NULL DEFAULT 'person'",
        ]:
            with contextlib.suppress(sqlite3.OperationalError):
                self.conn.execute(migration)
        # Unique index on (name, kind) — idempotent via IF NOT EXISTS
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_subjects_name_kind ON subjects(name, kind)"
        )
        self.conn.commit()

    @_locked
    def run(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        """Execute SQL and commit. For ad-hoc queries from outside."""
        self.conn.execute(sql, params)
        self.conn.commit()

    @_locked
    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[Any]:
        """Execute SQL and return all rows."""
        return self.conn.execute(sql, params).fetchall()

    PET_SPECIES = ("dog", "cat", "other_pet")
    KIND_TO_SPECIES: dict[str, str] = {"person": "human", "pet": "pet"}

    def close(self) -> None:
        self.conn.close()

    def resolve_path(self, stored_path: str) -> Path:
        """Resolve a DB-stored path to an absolute filesystem path.

        Strips the __pets suffix, then searches base_dir and db_path.parent
        for the file. Returns the first existing match, or the primary
        candidate for error reporting.
        """
        clean = stored_path.removesuffix("__pets")
        if clean.startswith("/"):
            return Path(clean)
        if ".." in clean.split("/"):
            msg = f"Path contains '..': {clean}"
            raise ValueError(msg)
        # tmp/ paths are app-generated (video frames, etc.) — relative to DB directory
        if clean.startswith("tmp/"):
            return self.db_path.parent / clean
        if self.base_dir is not None:
            return self.base_dir / clean
        return Path(clean)

    def to_relative(self, absolute_path: str) -> str:
        """Convert an absolute path to a relative path (stripping base_dir prefix)."""
        if self.base_dir is None:
            return absolute_path
        try:
            return str(Path(absolute_path).resolve().relative_to(self.base_dir))
        except ValueError:
            return absolute_path

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

    # --- Photos ---

    @_locked
    def add_photo(
        self,
        file_path: str,
        width: int,
        height: int,
        taken_at: str | None = None,
        video_path: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO photos (file_path, width, height, taken_at, scanned_at, "
            "video_path, latitude, longitude) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (file_path, width, height, taken_at, self._now(), video_path, latitude, longitude),
        )
        self.conn.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    @_locked
    def is_photo_scanned(self, file_path: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM photos WHERE file_path = ?", (file_path,)).fetchone()
        return row is not None

    @_locked
    def is_pet_scanned(self, file_path: str) -> bool:
        """Check if a photo has been scanned for pets (separate from face scan)."""
        row = self.conn.execute(
            "SELECT 1 FROM photos WHERE file_path = ?",
            (file_path + "__pets",),
        ).fetchone()
        return row is not None

    @_locked
    def is_video_scanned(self, video_path: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM photos WHERE video_path = ?", (video_path,)
        ).fetchone()
        return row is not None

    @_locked
    def get_photo(self, photo_id: int) -> Photo | None:
        row = self.conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()
        if not row:
            return None
        return Photo(**dict(row))

    @_locked
    def get_photos_batch(self, photo_ids: list[int]) -> dict[int, Photo]:
        """Return {photo_id: Photo} for multiple photos in one query."""
        if not photo_ids:
            return {}
        placeholders = ",".join("?" * len(photo_ids))
        rows = self.conn.execute(
            f"SELECT * FROM photos WHERE id IN ({placeholders})", photo_ids
        ).fetchall()
        return {r["id"]: Photo(**dict(r)) for r in rows}

    @_locked
    def get_photo_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM photos").fetchone()
        return int(row[0]) if row else 0

    # --- Faces ---

    @_locked
    def add_faces_batch(
        self,
        faces_data: list[tuple[int, tuple[int, int, int, int], np.ndarray, float]],
        species: str = "human",
    ) -> None:
        """Add multiple faces in a single transaction."""
        now = self._now()
        self.conn.executemany(
            "INSERT INTO faces (photo_id, bbox_x, bbox_y, bbox_w, bbox_h, "
            "embedding, confidence, detected_at, species) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (pid, *bbox, emb.tobytes(), conf, now, species)
                for pid, bbox, emb, conf in faces_data
            ],
        )
        self.conn.commit()

    @_locked
    def get_face(self, face_id: int) -> Face | None:
        row = self.conn.execute("SELECT * FROM faces WHERE id = ?", (face_id,)).fetchone()
        if not row or row["embedding"] is None:
            return None
        d = dict(row)
        d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
        return Face(**d)

    @_locked
    def get_photo_faces(self, photo_id: int) -> list[Face]:
        rows = self.conn.execute("SELECT * FROM faces WHERE photo_id = ?", (photo_id,)).fetchall()
        result = []
        for row in rows:
            if row["embedding"] is None:
                continue
            d = dict(row)
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
            result.append(Face(**d))
        return result

    @_locked
    def get_face_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM faces").fetchone()
        return int(row[0]) if row else 0

    @_locked
    def get_all_embeddings(self, species: str = "human") -> list[tuple[int, np.ndarray]]:
        """Return all (face_id, embedding) pairs for clustering, excluding dismissed."""
        clause, params = self.species_filter(species)
        rows = self.conn.execute(
            f"SELECT id, embedding FROM faces "
            f"WHERE {clause} AND id NOT IN (SELECT face_id FROM dismissed_faces)",
            params,
        ).fetchall()
        return [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in rows]

    @_locked
    def get_unassigned_embeddings(self, species: str = "human") -> list[tuple[int, np.ndarray]]:
        """Return (face_id, embedding) for unassigned, non-dismissed faces only."""
        clause, params = self.species_filter(species)
        rows = self.conn.execute(
            f"SELECT id, embedding FROM faces "
            f"WHERE person_id IS NULL AND {clause} "
            f"AND id NOT IN (SELECT face_id FROM dismissed_faces)",
            params,
        ).fetchall()
        return [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in rows]

    @_locked
    def update_cluster_ids(self, face_cluster_map: dict[int, int]) -> None:
        """Update cluster_id for multiple faces."""
        self.conn.executemany(
            "UPDATE faces SET cluster_id = ? WHERE id = ?",
            [(cid, fid) for fid, cid in face_cluster_map.items()],
        )
        self.conn.commit()

    @_locked
    def clear_clusters(self, species: str | None = None) -> None:
        """Reset cluster assignments (preserves person assignments).

        If species is given, only clear clusters for that species group.
        """
        if species is None:
            self.conn.execute("UPDATE faces SET cluster_id = NULL")
        else:
            clause, params = self.species_filter(species)
            self.conn.execute(f"UPDATE faces SET cluster_id = NULL WHERE {clause}", params)
        self.conn.commit()

    @_locked
    def get_cluster_ids(self) -> list[int]:
        """All distinct non-null cluster IDs."""
        rows = self.conn.execute(
            "SELECT DISTINCT cluster_id FROM faces WHERE cluster_id IS NOT NULL ORDER BY cluster_id"
        ).fetchall()
        return [row[0] for row in rows]

    def species_filter(self, species: str) -> tuple[str, tuple[str, ...]]:
        """Return SQL clause and params for species filtering."""
        if species == "pet":
            placeholders = ",".join("?" * len(self.PET_SPECIES))
            return f"species IN ({placeholders})", self.PET_SPECIES
        return "species = ?", (species,)

    @_locked
    def get_unnamed_clusters(self, species: str = "human") -> list[dict[str, Any]]:
        """Clusters with no subject assigned, ordered by size."""
        clause, params = self.species_filter(species)
        rows = self.conn.execute(
            f"""
            SELECT cluster_id, COUNT(*) as face_count,
                   GROUP_CONCAT(id) as face_ids
            FROM faces
            WHERE cluster_id IS NOT NULL AND person_id IS NULL AND {clause}
            GROUP BY cluster_id
            HAVING COUNT(*) >= 2
            ORDER BY face_count DESC
        """,
            params,
        ).fetchall()
        result = []
        for row in rows:
            face_ids = [int(x) for x in row["face_ids"].split(",")]
            result.append(
                {
                    "cluster_id": row["cluster_id"],
                    "face_count": row["face_count"],
                    "sample_face_ids": face_ids[:12],
                }
            )
        return result

    @_locked
    def get_singleton_faces(
        self, species: str = "human", limit: int = 200, offset: int = 0
    ) -> list[Face]:
        """Faces in clusters of size 1 or unclustered, unassigned, not dismissed."""
        clause, params = self.species_filter(species)
        rows = self.conn.execute(
            f"""
            SELECT f.* FROM faces f
            WHERE f.person_id IS NULL AND {clause}
            AND f.id NOT IN (SELECT face_id FROM dismissed_faces)
            AND (
                f.cluster_id IS NULL
                OR f.cluster_id IN (
                    SELECT cluster_id FROM faces
                    WHERE cluster_id IS NOT NULL AND person_id IS NULL
                    GROUP BY cluster_id HAVING COUNT(*) = 1
                )
            )
            LIMIT ? OFFSET ?
        """,
            (*params, limit, offset),
        ).fetchall()
        result = []
        for row in rows:
            if row["embedding"] is None:
                continue
            d = dict(row)
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
            result.append(Face(**d))
        return result

    @_locked
    def get_singleton_count(self, species: str = "human") -> int:
        clause, params = self.species_filter(species)
        row = self.conn.execute(
            f"""
            SELECT COUNT(*) FROM faces f
            WHERE f.person_id IS NULL AND {clause}
            AND f.id NOT IN (SELECT face_id FROM dismissed_faces)
            AND (
                f.cluster_id IS NULL
                OR f.cluster_id IN (
                    SELECT cluster_id FROM faces
                    WHERE cluster_id IS NOT NULL AND person_id IS NULL
                    GROUP BY cluster_id HAVING COUNT(*) = 1
                )
            )
        """,
            params,
        ).fetchone()
        return int(row[0]) if row else 0

    @_locked
    def get_cluster_face_count(self, cluster_id: int) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM faces WHERE cluster_id = ?",
            (cluster_id,),
        ).fetchone()
        return int(row[0]) if row else 0

    @_locked
    def get_cluster_faces(self, cluster_id: int, limit: int = 200) -> list[Face]:
        rows = self.conn.execute(
            "SELECT * FROM faces WHERE cluster_id = ? LIMIT ?",
            (cluster_id, limit),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
            result.append(Face(**d))
        return result

    # --- Subjects ---

    def _species_for_kind(self, kind: str) -> str:
        """Map subject kind to the face species filter value."""
        return self.KIND_TO_SPECIES[kind]

    def _is_species_kind_compatible(self, face_species: str, subject_kind: str) -> bool:
        """Check if a face's species is compatible with a subject's kind."""
        if subject_kind == "person":
            return face_species == "human"
        if subject_kind == "pet":
            return face_species in self.PET_SPECIES
        return True

    @_locked
    def create_subject(self, name: str, kind: str = "person") -> int:
        """Create a subject or return existing one if name+kind matches."""
        row = self.conn.execute(
            "SELECT id FROM subjects WHERE name = ? AND kind = ?", (name, kind)
        ).fetchone()
        if row:
            return int(row[0])
        cur = self.conn.execute(
            "INSERT INTO subjects (name, kind, created_at) VALUES (?, ?, ?)",
            (name, kind, self._now()),
        )
        self.conn.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    @_locked
    def get_subject(self, subject_id: int) -> Subject | None:
        row = self.conn.execute(
            """
            SELECT s.id, s.name, s.kind, COUNT(f.id) as face_count
            FROM subjects s LEFT JOIN faces f ON f.person_id = s.id
            WHERE s.id = ? GROUP BY s.id
            """,
            (subject_id,),
        ).fetchone()
        if not row:
            return None
        return Subject(
            id=row["id"], name=row["name"], kind=row["kind"], face_count=row["face_count"]
        )

    @_locked
    def get_subjects(self) -> list[Subject]:
        rows = self.conn.execute("""
            SELECT s.id, s.name, s.kind, COUNT(f.id) as face_count
            FROM subjects s LEFT JOIN faces f ON f.person_id = s.id
            GROUP BY s.id ORDER BY s.name
        """).fetchall()
        return [
            Subject(id=r["id"], name=r["name"], kind=r["kind"], face_count=r["face_count"])
            for r in rows
        ]

    @_locked
    def get_subjects_by_kind(self, kind: str) -> list[Subject]:
        """Return subjects of the given kind with their face counts."""
        rows = self.conn.execute(
            """
            SELECT s.id, s.name, s.kind, COUNT(f.id) as face_count
            FROM subjects s LEFT JOIN faces f ON f.person_id = s.id
            WHERE s.kind = ?
            GROUP BY s.id ORDER BY s.name
            """,
            (kind,),
        ).fetchall()
        return [
            Subject(id=r["id"], name=r["name"], kind=r["kind"], face_count=r["face_count"])
            for r in rows
        ]

    @_locked
    def get_subject_centroids(self, kind: str = "person") -> list[tuple[int, str, np.ndarray]]:
        """Return [(subject_id, name, centroid)] for all subjects of a kind.

        Single query for all face embeddings, grouped by subject in Python.
        """
        species = self._species_for_kind(kind)
        clause, params = self.species_filter(species)
        rows = self.conn.execute(
            f"""
            SELECT f.person_id, s.name, f.embedding
            FROM faces f
            JOIN subjects s ON s.id = f.person_id
            WHERE f.person_id IS NOT NULL AND s.kind = ? AND {clause}
            ORDER BY f.person_id
            """,
            (kind, *params),
        ).fetchall()

        from .embeddings import compute_centroid

        subject_embs: dict[int, tuple[str, list[np.ndarray]]] = {}
        for r in rows:
            sid = r["person_id"]
            if sid not in subject_embs:
                subject_embs[sid] = (r["name"], [])
            subject_embs[sid][1].append(np.frombuffer(r["embedding"], dtype=np.float32))

        result: list[tuple[int, str, np.ndarray]] = []
        for sid, (name, embs) in subject_embs.items():
            centroid = compute_centroid(np.array(embs))
            result.append((sid, name, centroid))
        return result

    @_locked
    def rename_subject(self, subject_id: int, name: str) -> None:
        self.conn.execute("UPDATE subjects SET name = ? WHERE id = ?", (name, subject_id))
        self.conn.commit()

    @_locked
    def assign_face_to_subject(self, face_id: int, subject_id: int) -> None:
        """Assign a face to a subject, validating species/kind compatibility."""
        face = self.get_face(face_id)
        subject = self.get_subject(subject_id)
        if face and subject and not self._is_species_kind_compatible(face.species, subject.kind):
            raise ValueError(f"Cannot assign {face.species} face to a {subject.kind} subject")
        self.conn.execute("UPDATE faces SET person_id = ? WHERE id = ?", (subject_id, face_id))
        self.conn.commit()

    @_locked
    def assign_cluster_to_subject(self, cluster_id: int, subject_id: int) -> None:
        """Assign all unassigned faces in a cluster to a subject, validating compatibility."""
        subject = self.get_subject(subject_id)
        if subject:
            faces = self.get_cluster_faces(cluster_id, limit=1)
            if faces and not self._is_species_kind_compatible(faces[0].species, subject.kind):
                raise ValueError(
                    f"Cannot assign {faces[0].species} face to a {subject.kind} subject"
                )
        self.conn.execute(
            "UPDATE faces SET person_id = ? WHERE cluster_id = ? AND person_id IS NULL",
            (subject_id, cluster_id),
        )
        self.conn.commit()

    @_locked
    def merge_subjects(self, source_id: int, target_id: int) -> None:
        """Merge source subject into target: reassign faces, delete source."""
        self.conn.execute(
            "UPDATE faces SET person_id = ? WHERE person_id = ?",
            (target_id, source_id),
        )
        self.conn.execute("DELETE FROM subjects WHERE id = ?", (source_id,))
        self.conn.commit()

    @_locked
    def get_subject_faces(self, subject_id: int, limit: int = 200, offset: int = 0) -> list[Face]:
        rows = self.conn.execute(
            """SELECT f.* FROM faces f
               LEFT JOIN photos p ON f.photo_id = p.id
               WHERE f.person_id = ?
               ORDER BY p.taken_at DESC, f.id
               LIMIT ? OFFSET ?""",
            (subject_id, limit, offset),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
            result.append(Face(**d))
        return result

    def get_subject_faces_with_paths(
        self, subject_id: int, limit: int = 200, offset: int = 0
    ) -> list[tuple[Face, str]]:
        """Return faces with their photo's file_path, sorted by path (date-based dirs)."""
        rows = self.conn.execute(
            """SELECT f.*, p.file_path AS photo_path FROM faces f
               LEFT JOIN photos p ON f.photo_id = p.id
               WHERE f.person_id = ?
               ORDER BY p.file_path DESC, f.id
               LIMIT ? OFFSET ?""",
            (subject_id, limit, offset),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            path = d.pop("photo_path", "") or ""
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
            result.append((Face(**d), path))
        return result

    def get_random_avatars(self, subject_ids: list[int]) -> dict[int, int]:
        """Pick one random face ID per subject for avatar thumbnails."""
        if not subject_ids:
            return {}
        placeholders = ",".join("?" * len(subject_ids))
        rows = self.conn.execute(
            f"""SELECT person_id, id FROM (
                    SELECT person_id, id, ROW_NUMBER() OVER (
                        PARTITION BY person_id ORDER BY RANDOM()
                    ) AS rn FROM faces WHERE person_id IN ({placeholders})
                ) WHERE rn = 1""",
            tuple(subject_ids),
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_photos_with_all_subjects(
        self, subject_ids: list[int], limit: int = 0, offset: int = 0
    ) -> list[Photo]:
        """Find photos that contain ALL given subjects (intersection).

        Uses faces index: filter by person_id first, group by photo, keep
        only photos matching all requested subjects.
        """
        if not subject_ids:
            return []
        placeholders = ",".join("?" * len(subject_ids))
        pagination = ""
        params: tuple[int | str, ...] = (*subject_ids, len(subject_ids))
        if limit > 0:
            pagination = " LIMIT ? OFFSET ?"
            params = (*params, limit, offset)
        rows = self.conn.execute(
            f"""
            SELECT p.* FROM photos p
            JOIN (
                SELECT photo_id FROM faces
                WHERE person_id IN ({placeholders})
                GROUP BY photo_id
                HAVING COUNT(DISTINCT person_id) = ?
            ) matched ON matched.photo_id = p.id
            ORDER BY p.file_path DESC{pagination}
            """,
            params,
        ).fetchall()
        return [Photo(**dict(r)) for r in rows]

    def count_photos_with_all_subjects(self, subject_ids: list[int]) -> int:
        """Count photos containing ALL given subjects."""
        if not subject_ids:
            return 0
        placeholders = ",".join("?" * len(subject_ids))
        row = self.conn.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT photo_id FROM faces
                WHERE person_id IN ({placeholders})
                GROUP BY photo_id
                HAVING COUNT(DISTINCT person_id) = ?
            )
            """,
            (*subject_ids, len(subject_ids)),
        ).fetchone()
        return int(row[0]) if row else 0

    @_locked
    def get_subject_photos(self, subject_id: int) -> list[Photo]:
        """All unique photos containing a subject, newest first (by path)."""
        rows = self.conn.execute(
            """
            SELECT DISTINCT p.* FROM photos p
            JOIN faces f ON f.photo_id = p.id
            WHERE f.person_id = ?
            ORDER BY p.file_path DESC
            """,
            (subject_id,),
        ).fetchall()
        return [Photo(**dict(r)) for r in rows]

    @_locked
    def dismiss_faces(self, face_ids: list[int]) -> None:
        """Mark faces as non-faces (statues, paintings, etc.)."""
        self.conn.executemany(
            "INSERT OR IGNORE INTO dismissed_faces (face_id) VALUES (?)",
            [(fid,) for fid in face_ids],
        )
        self.conn.executemany(
            "UPDATE faces SET cluster_id = NULL, person_id = NULL WHERE id = ?",
            [(fid,) for fid in face_ids],
        )
        self.conn.commit()

    @_locked
    def unassign_face(self, face_id: int) -> None:
        """Remove subject assignment from a face."""
        self.conn.execute("UPDATE faces SET person_id = NULL WHERE id = ?", (face_id,))
        self.conn.commit()

    @_locked
    def exclude_faces(self, face_ids: list[int], cluster_id: int) -> None:
        """Remove faces from a cluster (set cluster_id to NULL)."""
        placeholders = ",".join("?" * len(face_ids))
        self.conn.execute(
            f"UPDATE faces SET cluster_id = NULL WHERE id IN ({placeholders}) AND cluster_id = ?",
            (*face_ids, cluster_id),
        )
        self.conn.commit()

    @_locked
    def merge_clusters(self, source_id: int, target_id: int) -> None:
        """Move all faces from source cluster to target cluster."""
        self.conn.execute(
            "UPDATE faces SET cluster_id = ? WHERE cluster_id = ?",
            (target_id, source_id),
        )
        self.conn.commit()

    @_locked
    def get_cluster_face_ids(self, cluster_id: int) -> list[int]:
        """Return all face IDs in a cluster."""
        rows = self.conn.execute(
            "SELECT id FROM faces WHERE cluster_id = ?", (cluster_id,)
        ).fetchall()
        return [r[0] for r in rows]

    @_locked
    def delete_subject(self, subject_id: int) -> None:
        """Delete a subject and unassign their faces."""
        self.conn.execute("UPDATE faces SET person_id = NULL WHERE person_id = ?", (subject_id,))
        self.conn.execute("DELETE FROM subjects WHERE id = ?", (subject_id,))
        self.conn.commit()

    @_locked
    def get_unclustered_embeddings(self, species: str = "human") -> list[tuple[int, np.ndarray]]:
        """Return (face_id, embedding) for unclustered, unassigned, non-dismissed faces."""
        clause, params = self.species_filter(species)
        rows = self.conn.execute(
            f"SELECT id, embedding FROM faces "
            f"WHERE person_id IS NULL AND cluster_id IS NULL AND {clause} "
            f"AND id NOT IN (SELECT face_id FROM dismissed_faces)",
            params,
        ).fetchall()
        return [(r[0], np.frombuffer(r[1], dtype=np.float32)) for r in rows]

    @_locked
    def search_subjects(self, query: str, kind: str | None = None) -> list[Subject]:
        if kind:
            rows = self.conn.execute(
                """
                SELECT s.id, s.name, s.kind, COUNT(f.id) as face_count
                FROM subjects s LEFT JOIN faces f ON f.person_id = s.id
                WHERE s.name LIKE ? AND s.kind = ?
                GROUP BY s.id ORDER BY s.name
                """,
                (f"%{query}%", kind),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT s.id, s.name, s.kind, COUNT(f.id) as face_count
                FROM subjects s LEFT JOIN faces f ON f.person_id = s.id
                WHERE s.name LIKE ?
                GROUP BY s.id ORDER BY s.name
                """,
                (f"%{query}%",),
            ).fetchall()
        return [
            Subject(id=r["id"], name=r["name"], kind=r["kind"], face_count=r["face_count"])
            for r in rows
        ]

    # --- Stats ---

    def _count(self, sql: str, params: tuple[str, ...] = ()) -> int:
        row = self.conn.execute(sql, params).fetchone()
        return int(row[0]) if row else 0

    @_locked
    def get_stats(self, species: str = "human") -> dict[str, int]:
        clause, params = self.species_filter(species)
        return {
            "total_photos": self._count("SELECT COUNT(*) FROM photos"),
            "total_faces": self._count(f"SELECT COUNT(*) FROM faces WHERE {clause}", params),
            "total_subjects": self._count("SELECT COUNT(*) FROM subjects"),
            "named_faces": self._count(
                f"SELECT COUNT(*) FROM faces WHERE person_id IS NOT NULL AND {clause}",
                params,
            ),
            "unnamed_clusters": self._count(
                f"SELECT COUNT(DISTINCT cluster_id) FROM faces "
                f"WHERE cluster_id IS NOT NULL AND person_id IS NULL AND {clause}",
                params,
            ),
            "unclustered_faces": self._count(
                f"SELECT COUNT(*) FROM faces WHERE cluster_id IS NULL AND {clause} "
                f"AND id NOT IN (SELECT face_id FROM dismissed_faces)",
                params,
            ),
            "dismissed_faces": self._count("SELECT COUNT(*) FROM dismissed_faces"),
        }

    # --- Export ---

    @_locked
    def export_json(self) -> str:
        """Export as JSON: subject -> photos -> face rectangles."""
        subjects = self.get_subjects()
        data: dict[str, list[Any]] = {"subjects": [], "unnamed_faces": []}

        for subject in subjects:
            faces = self.get_subject_faces(subject.id, limit=10000)
            photos_map: dict[str, list[dict[str, Any]]] = {}
            for face in faces:
                photo = self.get_photo(face.photo_id)
                if photo:
                    photos_map.setdefault(photo.file_path, []).append(
                        {
                            "bbox": [
                                face.bbox_x,
                                face.bbox_y,
                                face.bbox_w,
                                face.bbox_h,
                            ],
                            "confidence": face.confidence,
                        }
                    )
            data["subjects"].append(
                {
                    "id": subject.id,
                    "name": subject.name,
                    "kind": subject.kind,
                    "photos": [
                        {"file_path": fp, "faces": rects} for fp, rects in photos_map.items()
                    ],
                }
            )

        unnamed = self.conn.execute("SELECT * FROM faces WHERE person_id IS NULL").fetchall()
        for row in unnamed:
            photo = self.get_photo(row["photo_id"])
            if photo:
                data["unnamed_faces"].append(
                    {
                        "photo": photo.file_path,
                        "bbox": [
                            row["bbox_x"],
                            row["bbox_y"],
                            row["bbox_w"],
                            row["bbox_h"],
                        ],
                        "cluster_id": row["cluster_id"],
                    }
                )

        return json.dumps(data, indent=2)
