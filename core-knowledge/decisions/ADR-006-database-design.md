# ADR-006: Database design

## Status

Accepted

## Context

The system needs persistent storage for photos, face detections (with embeddings), person assignments, and cluster assignments. The storage must support:

- Incremental scanning (skip already-processed photos/videos)
- Concurrent access from the web UI (FastAPI with uvicorn workers) and CLI commands
- Efficient queries for clustering, searching, and browsing
- A simple deployment model (no external database server)

## Decision

We use **SQLite** with the following configuration and schema:

### Connection settings

- **WAL journal mode** (`PRAGMA journal_mode=WAL`): allows concurrent reads while a write is in progress, which is essential for the web UI serving thumbnails while a scan or clustering operation is running
- **Foreign keys enabled** (`PRAGMA foreign_keys=ON`)
- **`check_same_thread=False`**: SQLite connection is shared across threads (FastAPI uses a thread pool)
- **`threading.RLock`**: all database methods are wrapped with a reentrant lock via the `@_locked` decorator, ensuring thread-safe access. RLock (rather than Lock) allows methods to call other locked methods without deadlocking.

### Schema

Four tables:

**photos**
- `id` (PK), `file_path` (UNIQUE), `width`, `height`, `taken_at` (EXIF date), `scanned_at`
- `video_path` (nullable): added by migration, stores the source video path for extracted frames

**faces**
- `id` (PK), `photo_id` (FK -> photos), `bbox_x/y/w/h`, `embedding` (BLOB, 512 float32 = 2048 bytes), `person_id` (FK -> persons, nullable), `cluster_id` (nullable), `confidence`, `detected_at`
- `species` (TEXT, default `'human'`): added by migration, separates human faces from pet detections (`dog`/`cat`)

**persons**
- `id` (PK), `name`, `created_at`

**dismissed_faces**
- `face_id` (PK, FK -> faces): faces marked as non-faces (statues, paintings, false detections)
- Dismissing a face also clears its `cluster_id` and `person_id`
- Dismissed faces are excluded from clustering, auto-assign, and the find-similar workflow

### Indexes

- `idx_faces_photo` on `faces(photo_id)` -- photo detail page queries
- `idx_faces_person` on `faces(person_id)` -- person detail/photo queries
- `idx_faces_cluster` on `faces(cluster_id)` -- cluster browsing
- `idx_photos_path` on `photos(file_path)` -- idempotent scan lookups

### Idempotency strategies

- **Photo scanning**: uses `file_path` uniqueness. If the absolute path exists in photos, the file is skipped.
- **Video scanning**: checks `video_path` column. If the video path exists in any photos row, the video is skipped.
- **Pet scanning**: uses `file_path + "__pets"` suffix. The same photo can have separate entries for face scanning and pet scanning.

### Schema migrations

New columns (`video_path`, `species`) are added via `ALTER TABLE ... ADD COLUMN` wrapped in try/except for `OperationalError` (column already exists). This is a simple forward-only migration approach.

## Consequences

**Positive:**

- Zero deployment overhead: SQLite is a single file, no server process needed
- WAL mode provides adequate concurrency for this use case (one writer, multiple readers)
- RLock prevents data corruption from concurrent web UI + CLI access
- The `__pets` suffix trick avoids schema changes for multi-pipeline scanning
- Embedding storage as BLOB is compact and avoids serialization overhead

**Negative:**

- SQLite's single-writer limitation means clustering and scanning block each other's writes. For a single-user tool this is acceptable.
- The database file grows large with many face embeddings (~2 KB per face for the embedding alone). A collection of 50k+ faces produces a database over 100 MB.
- The `__pets` suffix on `file_path` is a hack that conflates the photo identity with the scanning pipeline. A cleaner design would use a separate scans tracking table.
- No migration framework: if the schema needs more complex changes in the future, the manual `ALTER TABLE` approach will not scale. Consider alembic or similar if this becomes a problem.
- `check_same_thread=False` combined with RLock works but is not the most robust threading model. For higher concurrency, a connection pool would be better.
