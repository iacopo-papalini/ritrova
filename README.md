# FaceRecog

Face recognition and tagging for photo collections. Detects faces in photos and videos, clusters them by identity using ArcFace embeddings, detects pets (dogs and cats) using YOLO + SigLIP, and provides a web UI for naming and searching people and animals across years of photos.

Built for Apple Silicon (CoreML acceleration) but works on any platform via CPU fallback.

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
cd face_recog
uv sync
```

The first run of `scan` downloads the InsightFace `buffalo_l` model (~300 MB) to `~/.insightface/models/`. The first run of `scan-pets` downloads YOLO and SigLIP models.

## Recommended workflow

```bash
# 1. Scan photos for human faces
uv run face-recog scan /path/to/photos

# 2. Scan videos for human faces
uv run face-recog scan-videos /path/to/photos

# 3. Scan photos for pets (dogs and cats)
uv run face-recog scan-pets /path/to/photos

# 4. Cluster human faces
uv run face-recog cluster

# 5. Cluster pets (separately per species)
uv run face-recog cluster --species dog
uv run face-recog cluster --species cat

# 6. Launch the web UI to name clusters and review faces
uv run face-recog serve

# 7. After naming some persons in the UI, bulk-assign remaining clusters
uv run face-recog auto-assign

# 8. Re-cluster if needed (adjusting threshold), then auto-assign again
uv run face-recog cluster --threshold 0.50
uv run face-recog auto-assign
```

## CLI commands

All commands accept a global `--db PATH` option (default: `./faces.db`, env: `FACE_DB`) to specify the database location.

### scan

Scan a directory of photos for human faces.

```bash
uv run face-recog scan /path/to/photos
```

Walks the directory tree, finds JPG/JPEG files, detects faces using InsightFace/ArcFace, extracts 512-dimensional embeddings, and stores everything in the SQLite database.

Options:
- `--min-confidence 0.65` -- minimum detection confidence (filters false positives)

Scanning is incremental: already-processed photos are skipped on re-run.

### scan-videos

Scan a directory of videos for human faces.

```bash
uv run face-recog scan-videos /path/to/photos
```

Finds MP4, MOV, AVI, and MKV files. Samples one frame every N seconds, detects faces, and deduplicates per video (keeping the highest-confidence detection for each unique identity). Extracted frames are saved as JPEG in `tmp/frames/` next to the database.

Options:
- `--min-confidence 0.65` -- minimum detection confidence
- `--interval 2.0` -- seconds between sampled frames

Scanning is incremental: already-processed videos are skipped on re-run.

### scan-pets

Scan a directory of photos for dogs and cats.

```bash
uv run face-recog scan-pets /path/to/photos
```

Uses YOLO for object detection (COCO classes: dog, cat) and SigLIP for visual embeddings. Pet detections are stored separately from human faces using a species column, and pet photos use a `__pets` suffix for idempotent scanning.

Options:
- `--min-confidence 0.5` -- minimum YOLO detection confidence

Scanning is incremental: already-processed photos are skipped on re-run.

### cluster

Cluster detected faces by embedding similarity.

```bash
uv run face-recog cluster
```

Uses agglomerative clustering with complete linkage on cosine distances. Complete linkage requires ALL members of a cluster to be within the distance threshold of each other, which avoids the chaining problem common with DBSCAN.

Options:
- `--threshold 0.45` -- maximum cosine distance within a cluster (lower = stricter, fewer false merges)
- `--min-size 2` -- minimum number of faces to form a cluster (smaller groups become noise)
- `--species human` -- species to cluster (`human`, `dog`, `cat`); each species is clustered independently

Re-running cluster clears previous cluster assignments but **preserves** person assignments.

### auto-assign

Bulk-assign unnamed clusters to existing named persons by centroid similarity.

```bash
uv run face-recog auto-assign
```

Computes a centroid embedding for each named person and each unnamed cluster, then matches clusters to the most similar person. This is O(clusters x persons) rather than requiring a full re-clustering.

Options:
- `--min-similarity 50.0` -- minimum centroid similarity percentage to accept a match
- `--species human` -- species to process (`human`, `dog`, `cat`)

### serve

Start the web UI for browsing and naming faces.

```bash
uv run face-recog serve
```

Opens at [http://localhost:8787](http://localhost:8787).

Options:
- `--host 0.0.0.0` -- bind address
- `--port 8787` -- port number

### export

Export database as JSON.

```bash
uv run face-recog export              # JSON to stdout
uv run face-recog export -o data.json # JSON to file
```

Options:
- `--output / -o` -- output file path (default: `-` for stdout)

### stats

Show database statistics.

```bash
uv run face-recog stats
```

Displays: photos scanned, faces detected, persons named, named faces, unnamed clusters, unclustered faces.

## Web UI

The web UI (started with `serve`) provides the following pages:

### Dashboard (`/`)

Overview statistics: photos scanned, faces detected, persons named, named faces, unnamed clusters, unclustered faces, dismissed faces.

### Clusters (`/clusters`)

Browse unnamed face groups ordered by size. Each cluster shows thumbnail samples of the faces it contains. Click into a cluster to see all its faces and take action.

**Cluster detail** (`/clusters/{id}`):
- **Name** -- create a new person with a name and assign the cluster to them. After naming, you are automatically redirected to the next most-similar unnamed cluster for that person.
- **Assign** -- assign the cluster to an existing named person. Existing persons are ranked by centroid similarity to help you pick.
- **Dismiss** -- mark all faces in the cluster as non-faces (statues, paintings, etc.). Dismissed faces are excluded from clustering and all views.
- **Exclude** -- remove selected faces from the cluster (sets their cluster_id to NULL) without dismissing them, so they can be re-clustered later.

### Persons (`/persons`)

Browse all named people with their face counts. Click into a person for details.

**Person detail** (`/persons/{id}`):
- **Rename** -- change the person's name
- **Delete** -- remove the person (unassigns all their faces, does not delete face data)
- **Merge** -- merge this person into another (all faces are reassigned to the target, this person is deleted)
- **Find similar unclustered** -- search for unclustered faces that are similar to this person's centroid and claim them
- **Photo gallery** -- all photos containing this person, newest first

### Compare persons (`/compare`)

Select two persons and find misassigned faces. Shows faces from person A that are closer to person B's centroid (and vice versa), sorted by the size of the gap. You can then swap selected faces to the correct person.

### Merge suggestions (`/merge-suggestions`)

Automatically identifies pairs of clusters/persons whose centroids are highly similar, suggesting they might be the same identity. Results are paginated and sorted by similarity. Configurable minimum similarity threshold.

### Photo view (`/photos/{id}`)

Display a photo with bounding box overlays on all detected faces. Each face shows its confidence score and person assignment. You can assign individual faces to existing persons or unassign them.

### Search (`/search`)

Search for persons by name (case-insensitive substring match). Results link to person detail pages.

## Tuning

### Siblings / look-alikes getting merged

Lower the threshold for stricter clustering:

```bash
uv run face-recog cluster --threshold 0.40
```

### Too many small clusters / same person split across clusters

Raise the threshold to allow more variation:

```bash
uv run face-recog cluster --threshold 0.55
```

### Children growing up

Children's faces change significantly over years. The algorithm handles moderate aging well, but young children (toddler vs. teenager) will likely end up in separate clusters. Use the **merge** feature in the web UI to combine them under one person.

### Pets being confused

Pets of the same species can look similar. Use a stricter threshold and review clusters carefully:

```bash
uv run face-recog cluster --species dog --threshold 0.35
```

## Project structure

```
face_recog/
├── pyproject.toml
├── adr/                   # Architecture Decision Records
├── src/face_recog/
│   ├── cli.py             # CLI entry points (click)
│   ├── db.py              # SQLite schema and operations (WAL, RLock)
│   ├── detector.py        # InsightFace/ArcFace face detection + embeddings
│   ├── pet_detector.py    # YOLO detection + SigLIP embeddings for pets
│   ├── scanner.py         # Photo/video/pet scanning pipelines
│   ├── cluster.py         # Agglomerative clustering, auto-assign, merge suggestions
│   ├── app.py             # FastAPI web application
│   ├── templates/         # Jinja2 HTML templates
│   └── static/            # CSS and JS
```

## Database

SQLite with WAL journaling and four tables:

- **photos** -- file path, dimensions, EXIF date, video_path (for frames extracted from videos)
- **faces** -- bounding box (pixel coords), embedding (512-dim float32 blob), person/cluster assignment, confidence, species (`human`/`dog`/`cat`)
- **persons** -- name, created_at
- **dismissed_faces** -- face IDs marked as non-faces (excluded from clustering and views)

The JSON export produces:

```json
{
  "persons": [
    {
      "id": 1,
      "name": "Alice",
      "photos": [
        {
          "file_path": "/path/to/photo.jpg",
          "faces": [{"bbox": [x, y, w, h], "confidence": 0.89}]
        }
      ]
    }
  ],
  "unnamed_faces": [
    {
      "photo": "/path/to/photo.jpg",
      "bbox": [x, y, w, h],
      "cluster_id": 42
    }
  ]
}
```
