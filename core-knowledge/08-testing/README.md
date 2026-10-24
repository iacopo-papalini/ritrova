# Testing

> The conventions, fixtures and regression guards a new Ritrova test must fit into: real SQLite everywhere, no model weights, and one app instance per test case.

## Modules

- **Pure unit layer**²³ — embedding math, VLM response parsing, pipeline composition, `UndoStore` put/pop/peek/TTL semantics, file discovery and EXIF guards. No database, no HTTP, no models. Boundaries: in scope only where a function has no I/O; anything touching `FaceDB` belongs a layer down.

- **Database layer**¹² — exercises `FaceDB` and its mixins directly against a real SQLite file: sources, findings, subjects, clusters, curation, circles, scans, path metadata, print selection, stats, export. Boundaries: owns schema behaviour and constraints, not business rules.
  - Notable: migration tests hand-write a pre-migration schema with raw `sqlite3`, then open `FaceDB` on it and assert the migration, backfill and `NOT NULL` promotion all ran¹². The same raw-`sqlite3` trick with `PRAGMA foreign_keys=OFF` is how orphan rows are manufactured for the `doctor` tests¹³.

- **Service layer**³ — one `TestCase` per `services_domain` operation. Every test follows the same three beats: seed, call the service, then pop the returned receipt's token and call `entry.payload.undo(db)` and assert the original state is back. Boundaries: owns receipt messages, `None`-on-noop returns, `SpeciesMismatch`, and undo inversion; owns no HTTP.
  - Notable: `UndoStore` is single-slot, so a test that mutates twice must `undo.clear()` between the two calls or the second `put` clobbers the receipt it wants.

- **Transaction and concurrency layer**⁸ — the only tests that run threads. Two threads race `dismiss_findings_as_cluster` on one cluster and assert exactly one receipt and one `None`, never a half-dismissed cluster; the race is then repeated ten times to catch rare interleavings. Three more cases cover the reentrant `RLock` under `transaction()`, nested `transaction()` reusing the outer one, and rollback on exception¹⁷.
  - Notable: assertions are wall-clock based — a 200 ms hold, a 5 s deadlock threshold. A loaded machine can fail these without a code change.

- **Router layer**²² ⁷ ¹⁶ — `TestClient` over a real app. Two shapes: the whole app from `create_app`²² ⁵, or a bare `FastAPI` with a single router mounted after calling `deps.configure` by hand⁷. Boundaries: owns status codes, response shape, content types, cache headers and FastAPI body validation; owns no DOM.

- **CLI layer**¹³ — `click.testing.CliRunner` with `--db` pointed at a `tmp_path` database. Plain module-level test functions here, not `TestCase` classes — the one layer that departs from the project convention. Boundaries: argument handling, confirmation prompts, exit codes, and the DB-side cascade.

- **E2E layer**¹⁰ ¹¹ — Playwright against a real uvicorn server. Covers the typeahead picker, the subjects filter, the lightbox and its keyboard navigation, kind-toggle navigation, and the two undo flows whose toast must survive a redirect. Boundaries: owns the JS, htmx and `HX-Trigger` wiring that no other layer can see.

- **`tests/_helpers.add_findings`**² — the one way tests insert findings. Production wires one scan row per source and scan type; the helper finds-or-records that row for the `(source, species)` pair and then batches the insert. Calling `db.add_findings_batch` directly means supplying `scan_id` yourself.

- **`tests/conftest.db`**¹ — the only global fixture: a `FaceDB` on `tmp_path`, closed on teardown. Tests that need the file path as well as the handle build `FaceDB` inline instead and usually do not close it.

- **`services_domain/_fixtures.build_fixture`**³ — seeds two sources, one finding each, plus a fresh `UndoStore`, and returns them as a `Fixture` dataclass. Takes `species`, and picks the embedding width from it.

### Conventions

- `unittest.TestCase` subclass, with pytest fixtures reached through `@pytest.fixture(autouse=True) def _setup(self, tmp_path)` that assigns onto `self`. Plain `setUp` only where no fixture is needed⁹.
- Real SQLite always. The database is never mocked, never patched, never in-memory.
- Embeddings are deterministic unit vectors from `np.random.default_rng(seed)` — 512 dimensions for `human`, 768 for pet species. Each module defines its own one-line `_emb` helper rather than importing one.
- Model weights never load. Three ways out, by layer: stub `service._embed` with a lambda returning a fixed vector¹⁶, inject `MagicMock` detectors into the analysis steps¹⁵ or hand-written fake `AnalysisStep` subclasses into the pipeline¹⁴, or pass `--no-faces --no-pets` to the CLI¹³. No test needs the `scan` or `caption` extras installed.
- Real JPEGs are written with PIL into `tmp_path` whenever an endpoint or service will open the file.

### Dependency and undo isolation

`deps` holds module-level globals, not FastAPI `Depends`⁴. `create_app` rebinds all of them — database, templates, thumbnails directory, undo store, and the five cached domain services⁵. Isolation therefore comes from calling `create_app` in each test case's setup, not from any teardown: the last `create_app` in the process owns the globals. A router test that reaches `deps.get_findings_service()` to stub its `_embed`¹⁶ is relying on that binding being the app it just built, so it must not build a second one.

The undo store is created inside `create_app`, so a fresh app means a fresh single undo slot. Service-layer tests construct `UndoStore()` themselves³.

Router tests commonly hold two `FaceDB` instances on the same file: their own, for seeding and asserting, and the app's, created inside `create_app`²² ⁷ ¹⁶. These are separate connections with separate locks; the database runs in WAL mode¹⁷.

### Invariants with a dedicated guard

- **Route ordering**⁶ — every `/api/...` route registers before the first `/{kind}/...` route. The guard walks `app.router.routes` by index, then proves the point concretely: `GET /api/findings/<missing>/thumbnail` must answer `application/json` 404, not the HTML 404 the catch-all would render, and `POST /api/findings/dismiss` must still reach body validation.
- **`findings.scan_id` migration and backfill** from a legacy schema, including the synthetic scan created for an orphan¹².
- **The `finding_assignment` XOR check constraint** — both columns null, both set, or a bad exclusion reason must all be rejected¹².
- **Orphaned findings 404 rather than 500** when their source row is gone²².
- **Immutable `Cache-Control` on every image endpoint**, on both the generate and the on-disk-cache path²².
- **Cluster ids and subject ids do not collide** in the UI namespace²².
- **Zero-byte files are skipped** by discovery, and malformed EXIF GPS rationals never raise¹⁸.
- **Transaction atomicity, reentrancy, nesting and rollback**⁸.

## Operational notes

- `make test` runs `uv run pytest tests/ -v`²¹. `testpaths = ["tests"]` is the whole pytest configuration¹⁹ — no `addopts`, no registered markers, no coverage settings, although `pytest-cov` is a dev dependency.
- There are no markers or skip conditions anywhere in the suite, so a plain run includes the E2E layer. Running only the fast layers means naming paths on the command line.
- The E2E layer needs the `pytest-playwright` `page` fixture and an installed Chromium browser. Each E2E module starts its own uvicorn in a daemon thread on a module-scoped fixture, then polls `/` with httpx until it answers¹⁰ ¹¹.
- E2E ports are hard-coded: 18787 and 18788. Two suite runs at once, or anything already bound there, will collide. This also rules out running the suite in parallel.
- Slowest cases, in order: the two E2E modules, then the ten-iteration thread race and the timing assertions beside it⁸.
- The pre-commit hook runs ruff, ruff-format, and mypy over `src/` only²⁰ — test code is not type-checked by the hook. `make lint` covers `tests/` as well²¹.

## Related decisions

- [`../decisions/ADR-012-architecture-refactor.md`](../decisions/ADR-012-architecture-refactor.md) — the router split, the domain-service layer, and the single-transaction snapshot+mutate the concurrency tests pin down
- [`../decisions/ADR-006-database-design.md`](../decisions/ADR-006-database-design.md) — the schema the database layer asserts against

## References

¹ `tests/conftest.py` — `db` fixture
² `tests/_helpers.py` — `add_findings`
³ `tests/services_domain/_fixtures.py` — `Fixture`, `build_fixture`
⁴ `src/ritrova/app/deps.py` — `configure` and the `get_*` accessors over module globals
⁵ `src/ritrova/app/__init__.py` — `create_app`, router registration order
⁶ `tests/routers/test_route_ordering.py` — `TestRouteOrdering`
⁷ `tests/routers/test_undo_router.py` — single-router mount with a hand-called `deps.configure`
⁸ `tests/services_domain/test_transaction_concurrency.py` — thread races, deadlock and rollback guards
⁹ `tests/test_undo.py` — `TestUndoStore`, `TestUndoEndpoints`
¹⁰ `tests/test_e2e.py` — `app_url` uvicorn-on-a-thread fixture
¹¹ `tests/test_e2e_undo.py` — undo flows across a redirect
¹² `tests/test_db.py` — `_seed_legacy_db`, migration and constraint classes
¹³ `tests/test_cli.py` — `CliRunner` harness, `_orphan_a_source`
¹⁴ `tests/test_analysis.py` — fake `AnalysisStep` subclasses, `AnalysisPersister` tests
¹⁵ `tests/test_analysis_steps.py` — `MagicMock` detectors and describers per step
¹⁶ `tests/services_domain/test_findings_service.py` — `_embed` stubbing
¹⁷ `src/ritrova/db/connection.py` — `_TransactionalConnection`, `transaction`, WAL setup
¹⁸ `tests/test_scanner.py` — discovery and `get_exif_gps` guards
¹⁹ `pyproject.toml` — `[tool.pytest.ini_options]`, dev dependency group
²⁰ `.pre-commit-config.yaml` — ruff, ruff-format, mypy on `src/`
²¹ `Makefile` — `test` and `lint` targets
²² `tests/test_app.py` — whole-app router tests
²³ `tests/test_embeddings.py` — pure numeric unit tests

---

_Last updated: 2026-10-24 by Kipi_
