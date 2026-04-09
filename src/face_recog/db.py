"""SQLite database for face recognition data."""

import functools
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _locked(method):
    """Decorator: hold the DB lock for the entire method call."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
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
class Person:
    id: int
    name: str
    face_count: int = 0


class FaceDB:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                taken_at TEXT,
                scanned_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
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
                person_id INTEGER REFERENCES persons(id) ON DELETE SET NULL,
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
        # Migrations
        for migration in [
            "ALTER TABLE photos ADD COLUMN video_path TEXT",
            "ALTER TABLE faces ADD COLUMN species TEXT NOT NULL DEFAULT 'human'",
        ]:
            try:
                self.conn.execute(migration)
            except sqlite3.OperationalError:
                pass  # column already exists
        self.conn.commit()

    @_locked
    def run(self, sql, params=()):
        """Execute SQL and commit. For ad-hoc queries from outside."""
        self.conn.execute(sql, params)
        self.conn.commit()

    @_locked
    def query(self, sql, params=()):
        """Execute SQL and return all rows."""
        return self.conn.execute(sql, params).fetchall()

    PET_SPECIES = ("dog", "cat", "other_pet")

    def close(self):
        self.conn.close()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # --- Photos ---

    @_locked
    def add_photo(
        self,
        file_path: str,
        width: int,
        height: int,
        taken_at: str | None = None,
        video_path: str | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO photos (file_path, width, height, taken_at, scanned_at, video_path) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (file_path, width, height, taken_at, self._now(), video_path),
        )
        self.conn.commit()
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
    def get_photo_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]

    # --- Faces ---

    @_locked
    def add_faces_batch(
        self,
        faces_data: list[tuple[int, tuple, np.ndarray, float]],
        species: str = "human",
    ):
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
        return self.conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]

    @_locked
    def get_all_embeddings(self, species: str = "human") -> list[tuple[int, np.ndarray]]:
        """Return all (face_id, embedding) pairs for clustering, excluding dismissed."""
        rows = self.conn.execute(
            "SELECT id, embedding FROM faces "
            "WHERE species = ? AND id NOT IN (SELECT face_id FROM dismissed_faces)",
            (species,),
        ).fetchall()
        return [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in rows]

    @_locked
    def update_cluster_ids(self, face_cluster_map: dict[int, int]):
        """Update cluster_id for multiple faces."""
        self.conn.executemany(
            "UPDATE faces SET cluster_id = ? WHERE id = ?",
            [(cid, fid) for fid, cid in face_cluster_map.items()],
        )
        self.conn.commit()

    @_locked
    def clear_clusters(self):
        """Reset all cluster assignments (preserves person assignments)."""
        self.conn.execute("UPDATE faces SET cluster_id = NULL")
        self.conn.commit()

    @_locked
    def get_cluster_ids(self) -> list[int]:
        """All distinct non-null cluster IDs."""
        rows = self.conn.execute(
            "SELECT DISTINCT cluster_id FROM faces WHERE cluster_id IS NOT NULL ORDER BY cluster_id"
        ).fetchall()
        return [row[0] for row in rows]

    def _species_filter(self, species: str) -> tuple[str, tuple]:
        """Return SQL clause and params for species filtering."""
        if species == "pet":
            placeholders = ",".join("?" * len(self.PET_SPECIES))
            return f"species IN ({placeholders})", self.PET_SPECIES
        return "species = ?", (species,)

    @_locked
    def get_unnamed_clusters(self, species: str = "human") -> list[dict]:
        """Clusters with no person assigned, ordered by size."""
        clause, params = self._species_filter(species)
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
        clause, params = self._species_filter(species)
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
        clause, params = self._species_filter(species)
        return self.conn.execute(
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
        ).fetchone()[0]

    @_locked
    def get_cluster_face_count(self, cluster_id: int) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM faces WHERE cluster_id = ?",
            (cluster_id,),
        ).fetchone()[0]

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

    # --- Persons ---

    @_locked
    def create_person(self, name: str) -> int:
        """Create a person or return existing one if name matches."""
        row = self.conn.execute("SELECT id FROM persons WHERE name = ?", (name,)).fetchone()
        if row:
            return row[0]
        cur = self.conn.execute(
            "INSERT INTO persons (name, created_at) VALUES (?, ?)",
            (name, self._now()),
        )
        self.conn.commit()
        return cur.lastrowid

    @_locked
    def get_person(self, person_id: int) -> Person | None:
        row = self.conn.execute(
            """
            SELECT p.id, p.name, COUNT(f.id) as face_count
            FROM persons p LEFT JOIN faces f ON f.person_id = p.id
            WHERE p.id = ? GROUP BY p.id
            """,
            (person_id,),
        ).fetchone()
        if not row:
            return None
        return Person(id=row["id"], name=row["name"], face_count=row["face_count"])

    @_locked
    def get_persons(self) -> list[Person]:
        rows = self.conn.execute("""
            SELECT p.id, p.name, COUNT(f.id) as face_count
            FROM persons p LEFT JOIN faces f ON f.person_id = p.id
            GROUP BY p.id ORDER BY p.name
        """).fetchall()
        return [Person(id=r["id"], name=r["name"], face_count=r["face_count"]) for r in rows]

    @_locked
    def rename_person(self, person_id: int, name: str):
        self.conn.execute("UPDATE persons SET name = ? WHERE id = ?", (name, person_id))
        self.conn.commit()

    @_locked
    def assign_face_to_person(self, face_id: int, person_id: int):
        self.conn.execute("UPDATE faces SET person_id = ? WHERE id = ?", (person_id, face_id))
        self.conn.commit()

    @_locked
    def assign_cluster_to_person(self, cluster_id: int, person_id: int):
        self.conn.execute(
            "UPDATE faces SET person_id = ? WHERE cluster_id = ? AND person_id IS NULL",
            (person_id, cluster_id),
        )
        self.conn.commit()

    def merge_persons(self, source_id: int, target_id: int):
        """Merge source person into target: reassign faces, delete source."""
        self.conn.execute(
            "UPDATE faces SET person_id = ? WHERE person_id = ?",
            (target_id, source_id),
        )
        self.conn.execute("DELETE FROM persons WHERE id = ?", (source_id,))
        self.conn.commit()

    @_locked
    def get_person_faces(self, person_id: int, limit: int = 200) -> list[Face]:
        rows = self.conn.execute(
            "SELECT * FROM faces WHERE person_id = ? LIMIT ?",
            (person_id, limit),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
            result.append(Face(**d))
        return result

    @_locked
    def get_person_photos(self, person_id: int) -> list[Photo]:
        """All unique photos containing a person, newest first."""
        rows = self.conn.execute(
            """
            SELECT DISTINCT p.* FROM photos p
            JOIN faces f ON f.photo_id = p.id
            WHERE f.person_id = ?
            ORDER BY p.taken_at DESC, p.file_path
            """,
            (person_id,),
        ).fetchall()
        return [Photo(**dict(r)) for r in rows]

    @_locked
    def dismiss_faces(self, face_ids: list[int]):
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
    def search_persons(self, query: str) -> list[Person]:
        rows = self.conn.execute(
            """
            SELECT p.id, p.name, COUNT(f.id) as face_count
            FROM persons p LEFT JOIN faces f ON f.person_id = p.id
            WHERE p.name LIKE ?
            GROUP BY p.id ORDER BY p.name
            """,
            (f"%{query}%",),
        ).fetchall()
        return [Person(id=r["id"], name=r["name"], face_count=r["face_count"]) for r in rows]

    # --- Stats ---

    @_locked
    def get_stats(self, species: str = "human") -> dict:
        clause, params = self._species_filter(species)
        return {
            "total_photos": self.conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0],
            "total_faces": self.conn.execute(
                f"SELECT COUNT(*) FROM faces WHERE {clause}", params
            ).fetchone()[0],
            "total_persons": self.conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0],
            "named_faces": self.conn.execute(
                f"SELECT COUNT(*) FROM faces WHERE person_id IS NOT NULL AND {clause}",
                params,
            ).fetchone()[0],
            "unnamed_clusters": self.conn.execute(
                f"SELECT COUNT(DISTINCT cluster_id) FROM faces "
                f"WHERE cluster_id IS NOT NULL AND person_id IS NULL AND {clause}",
                params,
            ).fetchone()[0],
            "unclustered_faces": self.conn.execute(
                f"SELECT COUNT(*) FROM faces WHERE cluster_id IS NULL AND {clause} "
                f"AND id NOT IN (SELECT face_id FROM dismissed_faces)",
                params,
            ).fetchone()[0],
            "dismissed_faces": self.conn.execute("SELECT COUNT(*) FROM dismissed_faces").fetchone()[
                0
            ],
        }

    # --- Export ---

    @_locked
    def export_json(self) -> str:
        """Export as JSON: person -> photos -> face rectangles."""
        persons = self.get_persons()
        data: dict = {"persons": [], "unnamed_faces": []}

        for person in persons:
            faces = self.get_person_faces(person.id, limit=10000)
            photos_map: dict[str, list] = {}
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
            data["persons"].append(
                {
                    "id": person.id,
                    "name": person.name,
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
