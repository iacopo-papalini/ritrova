# ADR-009: Unified Source Analysis Pipeline

## Context

The current architecture runs **separate passes** per analysis type: one for human faces, one for pets, one for captions. Each pass iterates all sources independently, each scan function talks to the DB inline, and the return types/patterns are inconsistent despite the refactoring we just completed (db/ package split, ScanPipeline template, Detection dataclasses).

The real problem: adding a new analysis type means writing a new scan loop, a new CLI command, and new DB interaction code. The architecture couples "what to iterate" with "what analysis to run" with "where to store results."

**Goal**: A single pipeline that iterates sources once, applies N composable analysis steps, accumulates results in a `SourceAnalysis` value object, and persists at the end. The DB becomes the persistence layer, not the accumulator.

## Decisions from ADR grilling

| Decision | Choice | Why |
|----------|--------|-----|
| Processing model | Sequential, one source at a time | Parallelisation via queue/workers later |
| Scan type | New `composite` type; old types coexist during migration | Sources without `composite` get re-analysed |
| Finding→Scan relationship | Keep 1-to-many FK; update scan_id to latest composite scan | Simpler than M2M junction table; achieves same goal |
| SourceAnalysis mutability | API returns distinct objects; implementation may mutate for efficiency | Implementation detail, not contract |
| frame_path | Keep as regenerable cache hint; add `frame_number` to Finding | 0 for photos, N for video frames |
| Video model | Source → frame iterator → each frame enters pipeline → final collapse/dedup step | Process video once, not per-species |
| CLI | Unified `scan` command with `--ignore-pets`, `--ignore-people`, `--no-caption` | Old commands become thin wrappers |
| DB interaction | Analysis steps never touch DB; persistence is a final pipeline step | SRP, enables --dry-run, testability |

## Architecture

```
Source ──→ FrameIterator ──→ [AnalysisStep, ...] ──→ SourceAnalysis ──→ PersistStep ──→ DB
               │                    │                       │
               │                    ├─ FaceDetectionStep    ├─ findings[]
               │                    ├─ PetDetectionStep     ├─ description
               │                    ├─ CaptionStep          ├─ tags
               │                    └─ DeduplicationStep    └─ metadata
               │
               ├─ photo: yields 1 frame (the image itself)
               └─ video: yields every Nth frame
```

### Core types

```python
# src/ritrova/analysis.py

@dataclass
class FrameRef:
    """A single frame from a source (photo = 1 frame, video = many)."""
    source_path: Path
    frame_number: int          # 0 for photos
    image: Image.Image         # already loaded, EXIF-corrected
    width: int
    height: int

@dataclass
class Analysisfinding:
    """A detection not yet persisted — no DB ids."""
    bbox: tuple[int, int, int, int]
    embedding: np.ndarray
    confidence: float
    species: str
    frame_number: int

@dataclass
class SourceAnalysis:
    """Accumulated knowledge about a source, flowing through the pipeline."""
    source_path: str           # relative path (DB key)
    source_type: str           # "photo" | "video"
    width: int
    height: int
    taken_at: str | None
    latitude: float | None
    longitude: float | None
    findings: list[AnalysisFinding]
    caption: str
    tags: set[str]

class AnalysisStep(ABC):
    """One stage in the analysis pipeline."""
    @abstractmethod
    def analyse(self, frame: FrameRef, state: SourceAnalysis) -> SourceAnalysis: ...
    
    @property
    @abstractmethod
    def name(self) -> str: ...
```

### Builder

```python
class AnalysisPipelineBuilder:
    def with_face_detection(self, detector, min_confidence) -> Self: ...
    def with_pet_detection(self, detector, min_confidence) -> Self: ...
    def with_captioning(self, describer, translator=None) -> Self: ...
    def with_deduplication(self, threshold) -> Self: ...
    def build(self) -> AnalysisPipeline: ...

class AnalysisPipeline:
    """Iterates frames from a source, runs steps, returns SourceAnalysis."""
    def analyse_source(self, source_path: Path) -> SourceAnalysis: ...
```

### Persistence step (separate from pipeline)

```python
class AnalysisPersister:
    """Flushes a SourceAnalysis to the DB. Not an AnalysisStep — runs after."""
    def persist(self, analysis: SourceAnalysis, scan_type: str = "composite") -> None: ...
    # Creates/updates source, records scan, inserts findings + description
```

## Implementation phases

### Phase A: Core data model + step ABC
**Files**: `src/ritrova/analysis.py` (new), `tests/test_analysis.py` (new)

- `FrameRef`, `AnalysisFinding`, `SourceAnalysis` dataclasses
- `AnalysisStep` ABC with `analyse(frame, state) -> SourceAnalysis`
- `AnalysisPipeline` that composes steps and iterates frames
- `AnalysisPipelineBuilder`
- Frame iterators: `photo_frames(path) -> Iterator[FrameRef]`, `video_frames(path, interval) -> Iterator[FrameRef]`
- Tests with fake steps (no real detectors)

### Phase B: Concrete analysis steps wrapping existing code
**Files**: `src/ritrova/analysis_steps.py` (new), `tests/test_analysis_steps.py` (new)

- `FaceDetectionStep` — wraps `FaceDetector.detect()`, appends to `state.findings`
- `PetDetectionStep` — wraps `PetDetector.detect()`, appends to `state.findings`
- `CaptionStep` — wraps `Describer.describe()` + optional `Translator.translate()`, sets `state.caption` + `state.tags`
- `DeduplicationStep` — collapses duplicate findings across frames (same embedding within threshold → keep best confidence)
- Tests with mock detectors (reuse existing `_emb()` helper)

### Phase C: Persistence + unified CLI
**Files**: `src/ritrova/analysis.py` (add `AnalysisPersister`), `src/ritrova/cli.py` (modify), `src/ritrova/db/models.py` (add `frame_number`), `src/ritrova/db/findings.py` (update), `src/ritrova/db/connection.py` (migration)

- `AnalysisPersister.persist(analysis)` — creates source, records composite scan, inserts findings + description
- Add `frame_number` column to findings table (default 0, migration)
- New unified `scan` CLI command with `--ignore-pets`, `--ignore-people`, `--no-caption`
- Old `scan`, `scan-pets`, `describe` commands become thin wrappers that build single-step pipelines
- `--dry-run` flag: run pipeline, print results, skip persistence

### Phase D: Video frame integration
**Files**: `src/ritrova/analysis.py` (extend frame iterator), `src/ritrova/scanner.py` (reuse `_extract_video_faces` logic)

- `video_frames()` iterator: opens video, yields `FrameRef` every N frames
- `DeduplicationStep` handles cross-frame dedup (same person in frame 10 and 50)
- frame_path becomes cache: `AnalysisPersister` saves extracted frame JPEGs, stores path as cache hint
- `scan-videos` CLI command becomes wrapper for pipeline with video sources

## Files summary

| File | Action | Phase |
|------|--------|-------|
| `src/ritrova/analysis.py` | New — core types, pipeline, builder, persister | A, C |
| `src/ritrova/analysis_steps.py` | New — concrete steps | B |
| `src/ritrova/db/models.py` | Modify — add frame_number to Finding | C |
| `src/ritrova/db/findings.py` | Modify — frame_number in queries | C |
| `src/ritrova/db/connection.py` | Modify — migration for frame_number column | C |
| `src/ritrova/cli.py` | Modify — unified scan command | C |
| `src/ritrova/scanner.py` | Modify — old functions become wrappers | C |
| `tests/test_analysis.py` | New — pipeline + builder tests | A |
| `tests/test_analysis_steps.py` | New — step tests with mocks | B |

## What stays unchanged

- `FaceDetector`, `PetDetector` — wrapped by steps, not modified
- `Describer`, `Translator` — wrapped by CaptionStep
- `Detection`, `DetectionResult` — used inside steps
- `db/` package structure — only models.py and findings.py get minor additions
- Existing scan functions — become thin wrappers, not deleted (backwards compat)
- Web UI (`app.py`) — reads from DB, doesn't care how data got there

## Verification

After each phase:
```bash
uv run pytest tests/ -x --tb=short -q --ignore=tests/test_e2e.py --ignore=tests/test_e2e_undo.py
```

Phase C additionally: run `ritrova scan --photos-dir <path>` on a small test directory and verify findings + descriptions appear in the web UI.
