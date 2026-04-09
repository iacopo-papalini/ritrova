# TDD Refactoring Plan for face_recog

## Context

The face_recog project (~2500 lines, 7 modules) has grown organically with no tests, no linting, and several SOLID violations:
- Centroid computation duplicated 13 times across cluster.py and app.py
- Business logic embedded in presentation layers (app.py routes compute centroids, cli.py cleanup has quality filtering)
- Raw SQL scattered in app.py and cluster.py bypassing db.py abstraction
- Private db method accessed from cluster.py (`db._species_filter()`)
- No type checking, no linting, no tests

Goal: incremental TDD refactoring across 6 cycles — each leaves the code working, committed, and better structured. Uses `uv` for all Python/tool invocations.

---

## Cycle 0: Project Scaffolding (~30 min)

**Commit: "Initialize git, add pytest/ruff/mypy, create test scaffold"**

1. `git init` in project root
2. Create `.gitignore`: `.venv/`, `__pycache__/`, `*.db`, `tmp/`, `*.pt`, `.DS_Store`, `*.pyc`
3. Add dev dependencies via `uv add --dev pytest ruff mypy`
4. Add tool config to `pyproject.toml`:
   ```toml
   [tool.ruff]
   line-length = 100
   target-version = "py314"
   [tool.ruff.lint]
   select = ["E", "F", "W"]

   [tool.mypy]
   python_version = "3.14"
   warn_return_any = false
   warn_unused_configs = true
   ignore_missing_imports = true
   ```
5. Create `tests/__init__.py` and `tests/conftest.py` with db fixture:
   ```python
   @pytest.fixture
   def db(tmp_path):
       db_instance = FaceDB(tmp_path / "test.db")
       yield db_instance
       db_instance.close()
   ```
6. Run `uv run ruff check src/` for baseline
7. Git commit

---

## Cycle 1: Test and Harden db.py (~1.5 hr)

**Why first**: standalone (no internal imports), no ML deps, everything depends on it.

**Tests** (`tests/test_db.py`, TestCase classes):
- `TestPhotoOperations`: add/get/is_scanned/count/duplicate constraint
- `TestFaceOperations`: add_batch/get_face/get_photo_faces/embeddings round-trip/dismissed exclusion/cluster ops/singletons
- `TestPersonOperations`: create/idempotent name/get/rename/assign/merge/dismiss/search
- `TestStats`: empty db, populated db
- `TestSpeciesFilter`: human filter, pet filter
- `TestExport`: JSON structure

**Refactor db.py**:
1. Rename `_species_filter()` → `species_filter()` (public). Update call in cluster.py
2. Add missing methods to eliminate raw SQL in app.py/cluster.py:
   - `unassign_face(face_id)`
   - `delete_person(person_id)`
   - `exclude_faces(face_ids)`
   - `merge_clusters(source_id, target_id)`
   - `get_cluster_face_ids(cluster_id) -> list[int]`
   - `has_person_species(person_id, species) -> bool`
   - `get_unclustered_embeddings(species) -> list[tuple[int, ndarray]]`
3. Verify `@_locked` on all public methods

**ruff**: Enable `I` (isort). Fix db.py only.
**Commit: "Add comprehensive db.py tests; harden FaceDB public API"**

---

## Cycle 2: Extract Embedding Utilities + Test cluster.py Pure Functions (~1.5 hr)

**Why second**: centroid pattern duplicated 13 times — highest-value extraction.

**New file**: `src/face_recog/embeddings.py` (~30 lines)
- `normalize(v: ndarray) -> ndarray`
- `compute_centroid(embeddings: ndarray) -> ndarray`
- `cosine_similarity(a: ndarray, b: ndarray) -> float`
- `rank_by_similarity(query: ndarray, candidates: list[tuple[any, ndarray]]) -> list[tuple[any, float]]`

**Tests** (`tests/test_embeddings.py`):
- `TestNormalize`: unit vector, zero vector, random vector
- `TestComputeCentroid`: identical vectors, two vectors, single vector
- `TestCosineSimilarity`: identical, orthogonal, opposite
- `TestRankBySimilarity`: sorted desc, empty candidates

**Tests** (`tests/test_cluster.py`, @patch on db):
- `TestClusterFaces`: empty db, identical embeddings, distant embeddings
- `TestFindSimilarUnclustered`: finds match, empty person
- `TestComparePersons`: finds swaps, empty person

**Refactor cluster.py**: replace all 13 centroid+normalize patterns with `embeddings.*` calls.

**ruff**: Add `UP` (pyupgrade), `B` (bugbear). Run mypy on embeddings.py.
**Commit: "Extract embeddings.py; deduplicate centroid computation in cluster.py"**

---

## Cycle 3: Extract Service Layer from app.py and cli.py (~2 hr)

**Why third**: with db hardened and embeddings extracted, move business logic out of presentation.

**New file**: `src/face_recog/services.py` (~120 lines)
- `compute_cluster_hint(db, cluster_id) -> dict | None` — from app.py cluster_hint()
- `compute_singleton_hints(db, faces, species) -> dict[int, dict]` — from app.py singletons_page()
- `filter_persons_by_species(db, persons, species) -> list[Person]` — from app.py persons_page()
- `classify_face_quality(db, face_ids, min_size, min_sharpness, min_edges) -> tuple[list[int], list[int]]` — from cli.py cleanup()
- `find_next_similar_cluster(db, person_id) -> str` — from app.py _next_similar_cluster()

**Tests** (`tests/test_services.py`, @patch on db):
- `TestClusterHintService`: best match, no persons, species filtering
- `TestSingletonHintsService`: compute hints, no persons
- `TestPersonSpeciesFilter`: human, pet
- `TestCleanupService`: tiny faces, blurry faces, dry run

**Refactor app.py**: replace inline logic with services.* calls. Replace all `db.run()`/`db.query()` with db methods from cycle 1.
**Refactor cli.py**: replace cleanup() inline logic with `services.classify_face_quality()`.

**After-state**: app.py = route handlers only. cli.py = Click decorators + output only. Zero `db.run()`/`db.query()` in presentation layer.

**ruff**: Add `SIM`, `RET`.
**Commit: "Extract services.py; remove business logic from app.py and cli.py"**

---

## Cycle 4: Harden cluster.py + Fix Private Access (~1.5 hr)

**Tests** (`tests/test_cluster_integration.py`, real db fixture — no mocking):
- `TestAutoAssign`: matches cluster, below threshold skips, sweeps singletons, no persons
- `TestAutoMergeClusters`: similar clusters merge, skips named, keeps larger
- `TestSuggestMerges`: finds pair, respects threshold, species filter
- `TestFindSimilarCluster`: best match, none below threshold

**Refactor cluster.py**:
1. Replace `db._species_filter()` → `db.species_filter()` (renamed in cycle 1)
2. Replace `db.run()/db.query()` → proper db methods
3. Replace `print()` → `logging.info()`

**ruff**: Add `T20` (flag print in non-CLI), `LOG`.
**Commit: "Harden cluster.py; remove private DB access, raw SQL, print statements"**

---

## Cycle 5: Test scanner.py + app.py Routes (~1.5 hr)

**Tests** (`tests/test_scanner.py`, @patch on detectors):
- `TestFindImages`: discovers jpg, case insensitive, deduplicates
- `TestFindVideos`: discovers mp4, deduplicates
- `TestScanPhotos`: skips scanned, detects and stores, handles error
- `TestIsDuplicate`: above/below threshold, empty
- `TestScanPets`: stores species correctly

**Tests** (`tests/test_app.py`, TestClient + real in-memory db):
- `TestAppRoutes`: index 200, clusters 200, persons 200, person 404, name cluster, assign face, dismiss, export JSON

**Refactor**:
- scanner.py: replace `print()` → `logging.info()`
- app.py: verify every route body ≤ ~15 lines
- Add `db.get_cluster_faces_paginated()` / `db.get_person_faces_paginated()` for remaining db.query() calls

**ruff**: Add `ANN` (annotations for public functions).
**Commit: "Add scanner and app tests; finalize presentation layer cleanup"**

---

## Cycle 6: Final Cleanup, Type Annotations, Full Lint (~1 hr)

**What to do**:
1. Type annotations on all public function signatures
2. `uv run ruff check --fix src/ tests/` with full rules
3. `uv run mypy src/` — fix all errors
4. Add `py.typed` marker
5. Final `pyproject.toml` config:
   ```toml
   [tool.ruff.lint]
   select = ["E", "F", "W", "I", "UP", "B", "SIM", "RET", "T20", "LOG"]

   [tool.mypy]
   disallow_untyped_defs = true
   ```
6. Update README with new project structure

**Final architecture**:
```
src/face_recog/
  embeddings.py    # Pure math utilities (new)
  db.py            # Data access layer (hardened)
  services.py      # Business logic layer (new)
  cluster.py       # Clustering algorithms (cleaned)
  scanner.py       # Scanning pipelines (cleaned)
  detector.py      # Face detection (unchanged)
  pet_detector.py  # Pet detection (unchanged)
  cli.py           # CLI presentation (thinned)
  app.py           # Web presentation (thinned)
```

**Commit: "Final cleanup: type annotations, full lint compliance"**

---

## Verification

After each cycle:
1. `uv run pytest tests/ -v` — all green
2. `uv run ruff check src/ tests/` — no errors
3. `uv run face-recog stats` — CLI works
4. `uv run face-recog serve` — web UI works (manual spot-check)

After all cycles:
1. `uv run mypy src/` — zero errors
2. `uv run pytest tests/ -v --tb=short` — full green
3. Full manual test: scan a small photo set, cluster, browse web UI, name clusters, compare persons
