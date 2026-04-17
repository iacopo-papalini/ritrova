# Ritrova

Face recognition and tagging for photo collections. Detects faces in photos and videos, clusters them by identity using ArcFace embeddings, detects pets (dogs and cats) using YOLO + SigLIP, and provides a web UI for naming and searching people and animals across years of photos.

Built for Apple Silicon (CoreML acceleration) but most scanning features work cross-platform.

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
cd ritrova
uv sync
cp .env.example .env
# Edit .env: set PHOTOS_DIR to your photos root directory
```

### Windows + Nvidia notes

- `uv sync` skips `mlx-vlm` on Windows, because MLX does not publish Windows wheels there.
- Apple Silicon defaults to the MLX backend for image captioning.
- Windows and Linux fall back to a `transformers` VLM backend for image captioning.
- On Windows, `uv sync` pulls `torch` and `torchvision` from the official PyTorch CUDA 12.8 wheel index.
- If you want GPU acceleration for face detection on Windows, the project installs `onnxruntime-gpu` so InsightFace can use `CUDAExecutionProvider`.
- Pet embedding and the transformers VLM prefer CUDA automatically when a CUDA-enabled PyTorch build is installed.

The first run of `scan` downloads the InsightFace `buffalo_l` model (~300 MB) to `~/.insightface/models/`. The first run of `scan-pets` downloads YOLO and SigLIP models.

## Configuration

All configuration is via environment variables (loaded from `.env`):

| Variable | Default | Description |
|---|---|---|
| `PHOTOS_DIR` | (required) | Root directory for photos |
| `FACE_DB` | `./data/faces.db` | Path to SQLite database |

## Recommended workflow

```bash
# 1. Scan photos for human faces
uv run ritrova scan

# 2. Scan videos for human faces
uv run ritrova scan-videos

# 3. Scan photos for pets (dogs and cats)
uv run ritrova scan-pets

# 4. Cluster all faces (humans + dogs + cats)
uv run ritrova cluster

# 5. Launch the web UI to name clusters and review faces
uv run ritrova serve

# 6. After naming some persons in the UI, bulk-assign remaining clusters
uv run ritrova auto-assign

# 7. Re-cluster if needed (adjusting threshold), then auto-assign again
uv run ritrova cluster
uv run ritrova auto-assign
```

## CLI commands

All commands use `PHOTOS_DIR` and `FACE_DB` from `.env` (or `--photos-dir` / `--db` flags).

### scan

Scan the photos directory for human faces.

```bash
uv run ritrova scan
```

Walks the directory tree, finds JPG/JPEG files, detects faces using InsightFace/ArcFace, extracts 512-dimensional embeddings, and stores everything in the SQLite database. Paths are stored relative to `PHOTOS_DIR`.

Options:
- `--min-confidence 0.65` -- minimum detection confidence (filters false positives)

Scanning is incremental: already-processed photos are skipped on re-run.

### scan-videos

Scan the photos directory for videos containing human faces.

```bash
uv run ritrova scan-videos
```

Finds MP4, MOV, AVI, and MKV files. Samples one frame every N seconds, detects faces, and deduplicates per video (keeping the highest-confidence detection for each unique identity). Extracted frames are saved as JPEG in `data/tmp/frames/`.

Options:
- `--min-confidence 0.65` -- minimum detection confidence
- `--interval 2.0` -- seconds between sampled frames

Scanning is incremental: already-processed videos are skipped on re-run.

### scan-pets

Scan the photos directory for dogs and cats.

```bash
uv run ritrova scan-pets
```

Uses YOLO for object detection (COCO classes: dog, cat) and SigLIP for visual embeddings. Pet detections are stored separately from human faces using a species column.

Options:
- `--min-confidence 0.5` -- minimum YOLO detection confidence

Scanning is incremental: already-processed photos are skipped on re-run.

### cluster

Cluster all detected faces by embedding similarity.

```bash
uv run ritrova cluster
```

Clusters humans, dogs, and cats separately in one pass. Uses FAISS-accelerated two-phase clustering: brute-force range search to find candidate neighbor pairs, then exact complete-linkage verification within each connected component. Complete linkage guarantees that ALL members of a cluster are within the distance threshold of each other, avoiding chaining.

Options:
- `--threshold 0.45` -- maximum cosine distance within a cluster (lower = stricter)
- `--min-size 2` -- minimum number of faces to form a cluster

Re-running cluster clears previous cluster assignments for each species independently (human clustering does not affect pet clusters, and vice versa).

### auto-assign

Bulk-assign unnamed clusters to existing named persons by centroid similarity.

```bash
uv run ritrova auto-assign
```

Options:
- `--min-similarity 50.0` -- minimum centroid similarity percentage to accept a match
- `--species human` -- species to process (`human`, `dog`, `cat`)

### auto-merge

Auto-merge unnamed clusters whose centroids are highly similar.

```bash
uv run ritrova auto-merge
```

Options:
- `--min-similarity 70.0` -- minimum centroid similarity percentage
- `--species human` -- species to process

### cleanup

Dismiss tiny and blurry faces from the database.

```bash
uv run ritrova cleanup
```

Options:
- `--min-size 50` -- minimum face width/height in pixels
- `--min-sharpness 30.0` -- minimum Laplacian variance (focus blur)
- `--min-edges 2.0` -- minimum Canny edge density % (motion blur)
- `--dry-run` -- show what would be dismissed without acting

### serve

Start the web UI for browsing and naming faces.

```bash
uv run ritrova serve
```

Opens at [http://localhost:8787](http://localhost:8787).

Options:
- `--host 0.0.0.0` -- bind address
- `--port 8787` -- port number

### scans

Inspect and prune scan records.

```bash
uv run ritrova scans list                                # all scans
uv run ritrova scans list --source-pattern "2024/*"      # filter by GLOB on source path

uv run ritrova scans prune --scan-id 42                  # prune one scan + its findings
uv run ritrova scans prune --source-pattern "2024/*"     # prune every scan on matching sources
uv run ritrova scans prune --scan-id 42 --source-pattern "2024/*"  # intersection of both
```

Both filters are optional but at least one is required. The prune operation prints how many findings will be deleted (and how many of those have manual subject assignments) and asks for confirmation. Pass `-y` to skip the prompt.

### rescan

Re-run scans on a single source. Use this when the original scan was faulty or detection has improved — by default `scan` skips already-scanned sources to protect curation.

```bash
uv run ritrova rescan path/to/photo.jpg                  # re-run every existing scan on this source
uv run ritrova rescan path/to/photo.jpg --scan-type pet  # only re-run the pet scan
uv run ritrova rescan path/to/video.mp4 -y               # skip the confirm
```

The source can be specified as an absolute path or a path relative to `PHOTOS_DIR`. Existing scans (and their findings, including manual subject assignments) on the source are deleted before the new scan runs — confirms before acting.

### migrate-paths

Rewrite absolute paths in the DB to relative (using `PHOTOS_DIR` as base).

```bash
uv run ritrova migrate-paths
```

Run this once after setting `PHOTOS_DIR` if your database was created with absolute paths.

### export

Export database as JSON.

```bash
uv run ritrova export              # JSON to stdout
uv run ritrova export -o data.json # JSON to file
```

### stats

Show database statistics.

```bash
uv run ritrova stats
```

## Web UI

The web UI (started with `serve`) provides:

- **Dashboard** (`/`) -- overview statistics
- **Clusters** (`/clusters`) -- browse unnamed face groups, name/assign/dismiss them
- **Singletons** (`/singletons`) -- unmatched single faces with nearest-person hints
- **Persons** (`/persons`) -- browse named people, rename/delete/merge
- **Compare** (`/compare`) -- find misassigned faces between two persons
- **Merge suggestions** (`/merge-suggestions`) -- auto-detected similar cluster pairs
- **Photo view** (`/photos/{id}`) -- photo with face bounding box overlays
- **Search** (`/search`) -- search persons by name

Species toggle (human/pet) is available in the navigation bar.

## Project structure

```
ritrova/
├── pyproject.toml
├── .env.example           # Configuration template
├── adr/                   # Architecture Decision Records
├── data/                  # Runtime data (gitignored)
│   ├── faces.db           # SQLite database
│   ├── yolo11m.pt         # YOLO model weights
│   └── tmp/               # Thumbnails and video frames
├── src/ritrova/
│   ├── cli.py             # CLI entry points (click)
│   ├── db.py              # SQLite schema and operations (WAL, RLock)
│   ├── detector.py        # InsightFace/ArcFace face detection + embeddings
│   ├── pet_detector.py    # YOLO detection + SigLIP embeddings for pets
│   ├── scanner.py         # Photo/video/pet scanning pipelines
│   ├── cluster.py         # FAISS clustering, auto-assign, merge suggestions
│   ├── embeddings.py      # Pure math: normalize, centroid, cosine similarity
│   ├── services.py        # Business logic (cluster hints, singleton hints)
│   ├── app.py             # FastAPI web application
│   ├── templates/         # Jinja2 HTML templates
│   └── static/            # CSS and JS
└── tests/                 # pytest test suite (131 tests)
```

## Database

SQLite with WAL journaling and four tables:

- **photos** -- file path (relative to `PHOTOS_DIR`), dimensions, EXIF date, video_path
- **faces** -- bounding box, embedding (512-dim for humans, 768-dim for pets), person/cluster assignment, confidence, species (`human`/`dog`/`cat`)
- **persons** -- name, created_at
- **dismissed_faces** -- face IDs marked as non-faces
