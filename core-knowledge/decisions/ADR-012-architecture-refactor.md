# ADR-012: Architecture Refactor — Routers, Vocabulary, Services

## Context

The Software Architect audit (2026-04-20) identified three top architectural issues holding the codebase back:

1. **`app.py` is a 1378-line closure** — every route, every helper, every undo payload lives in `create_app`. Nothing is importable, route ordering is load-bearing, `static/app.js` (743 lines) is the client-side mirror.
2. **Finding state vocabulary is scattered** — `Finding.person_id` / `cluster_id` live on the dataclass but are LEFT-JOIN aliases of `finding_assignment.subject_id` / `cluster_findings.cluster_id`. Five vocabularies for one concept: UI `kind` ∈ {people, pets}, subject `kind` ∈ {person, pet}, finding `species` ∈ {human, dog, cat, other_pet}, dataclass `person_id`, table `subject_id`. The XOR invariant is enforced in SQL but not in Python types.
3. **12-mixin FaceDB with no domain layer** — cross-mixin calls need `# type: ignore`; MRO order is load-bearing; snapshot-then-mutate undo pattern is reimplemented 10+ times; two persistence paths (pipeline + inline scanner).

This ADR records the five-milestone plan the architect proposed. Milestones are strictly sequenced — each unblocks the next.

## Principles

- **No backwards-compatibility cruft.** Rename in one commit, delete the old name. Zero users = zero reason to keep legacy surfaces.
- **No speculative abstractions.** Thin coordinators over SQL, not a repository pattern.
- **Tests must not regress.** After every milestone, `uv run pytest tests/ --ignore=tests/test_e2e_undo.py -q` stays green.
- **Route-ordering invariant preserved.** `CLAUDE.md:16`: `/{kind}/…` page routes registered **after** all `/api/…` routes.
- **Work on `main` in small commits** — pre-commit hooks (ruff, ruff-format, mypy) must pass on each commit. No `--no-verify`.

---

## M0 — Delete legacy/compat surfaces  (Effort: **S**)

**Outcome.** The codebase reflects what we want today, not a migration halfway to it. No users exist; there is no one to stay compatible with.

### Concrete steps

1. **Delete `src/ritrova/app/routers/legacy_redirects.py`** and its inclusion in `create_app`. Old bare URLs (`/clusters`, `/persons`, `/singletons`, `/merge-suggestions`, `/compare`) simply 404. Delete any redirect tests.
2. **Delete pre-composite scan-type handling.** ADR-009 said "Old types coexist during migration" — migration is done. Grep for `"human"` / `"pet_yolo"` / `"caption"` as scan_type literals and replace with `"composite"`. Drop any `scan_type != 'composite'` branching code in pipeline / cleanup / prune paths.
3. **Delete empty root-level `faces.db`** (0-byte artifact at project root, not the real DB in `data/`).
4. **Audit for `# deprecated` / `# TODO: remove` / `# legacy` / `# migration`** comments and their guarded code paths. Remove each whose guard condition is now always-true or always-false.
5. **Drop any dataclass fields / DB columns marked as pre-refactor remnants.** (Mostly cleaned by the Apr refactor; this step is a final sweep.)

### Acceptance

- `rg -i 'legacy|deprecated|migration|compat' src/` returns only historic ADR references, no live guards.
- No scan_type other than `"composite"` (and `"tune-*"` A/B variants from ADR-010) is queried.
- Tests still green.

### Risk

Very low. If a legacy path is still load-bearing for some code path we don't know about, tests will catch it. Rollback = revert commits.

---

## M0.5 — Stop vocabulary leakage across layers  (Effort: **S**)

**Outcome.** Plural/singular kind each stay in their own layer. No HTTP code ever holds a singular subject_kind; no DB code ever holds a plural URL-kind. `species` is a separate concept and flows through both layers unchanged.

### The principle

Different vocabularies per layer is fine — **when they encode different concepts**:

| Layer | Concept | Vocabulary |
|---|---|---|
| DB row `subjects.kind` | "this single row is a…" | `"person"` / `"pet"` (singular) |
| URL path + Jinja `kind` var + UI nav | "browse the collection of…" | `"people"` / `"pets"` (plural) |
| Finding `species` (detector output) | the actual animal class | `"human"` / `"dog"` / `"cat"` / `"other_pet"` |

Singular vs. plural carry different meaning (row-level identity vs. collection-level browsing), so keeping both is defensible. **What's not defensible:** HTTP code that holds singular values, or DB code that holds plural ones. One vocabulary per layer; translate once at the boundary.

### Current violations

- `src/ritrova/app/helpers.py:48-52` `subject_kind_for_species(species) -> "person"|"pet"` — HTTP helper that **returns** singular. Singular has no business existing inside HTTP.
- `src/ritrova/app/helpers.py:55-57` `kind_for_subject(subject_kind) -> "people"|"pets"` — HTTP helper that **accepts** singular. Same violation, inbound.
- `src/ritrova/app/routers/clusters.py:49, 78, 114` — HTTP routers call `subject_kind_for_species(species)` to produce singular and feed it to `cluster.py` domain functions. HTTP is the wrong place to compute singular.

### Concrete steps

1. **Change `cluster.py` domain functions to accept `species` directly** where HTTP currently hands them singular kind:
   - `find_similar_cluster(db, subject_id, kind=)` → `find_similar_cluster(db, subject_id, species=)` (or add a sibling).
   - `suggest_merges(db, min_similarity=, kind=)` → `suggest_merges(db, min_similarity=, species=)`.
   - `auto_merge_clusters(db, min_similarity=, kind=)` → internal lookup from species; CLI caller is already singular so add a species overload or convert there.
   - Inside `cluster.py` the singular `kind` still exists for internal `KIND_TO_SPECIES` lookups — that's fine, it's the domain layer.
2. **Delete `subject_kind_for_species`** from `src/ritrova/app/helpers.py:48-52`. All HTTP call sites now pass `species` directly to domain functions.
3. **Delete `kind_for_subject`** from `src/ritrova/app/helpers.py:55-57`. Any HTTP code that currently converts a singular subject kind to plural should instead read the plural URL kind from context, or convert the subject's species via `kind_for_species` (HTTP already has species, which is a legitimate DB→HTTP crossing value).
4. **Audit `cluster.py` callers:** `cli.py` uses singular `kind` (fine — CLI is a domain-adjacent layer). HTTP routers must not pass `kind=subject_kind_for_species(...)`. Grep to confirm post-change.
5. **Audit `db/paths.py:68-74`** `_is_species_kind_compatible(species, subject_kind)` — already lives in the DB layer, stays singular, no change.

### Acceptance

- `rg 'subject_kind_for_species|kind_for_subject' src/` returns zero hits.
- Every `cluster.py` call from `src/ritrova/app/routers/*.py` passes `species=` not `kind=`.
- All tests green. No DB migration.

### Risk

Very low — it's a mechanical refactor of ~10 call sites plus two signature changes. The domain internal `KIND_TO_SPECIES` lookup is unchanged. Main gotcha: signature changes in `cluster.py` functions that are also called from `cli.py` — either rename the param and update CLI too, or add a species overload. Prefer the single-rename path.

---

## M1 — Split `app.py` into domain routers  (Effort: **M**)

**Outcome.** `app.py` becomes a ~150-line assembly point. Routes live in `src/ritrova/app/routers/{findings,clusters,subjects,circles,together,images,undo,pages}.py`.

### Concrete steps

1. Create `src/ritrova/app/` package. Move `app.py` to `src/ritrova/app/__init__.py`, rename `create_app` but keep its public signature.
2. Add `src/ritrova/app/deps.py` with `get_db()`, `get_undo_store()`, `get_templates()` — all returned from module-level singletons set by `create_app` at startup.
3. Extract the `_kind_for_species`, `_subject_kind_for_species`, `_describe_*` helpers (currently `app.py:51-77` and neighbouring) into `src/ritrova/app/helpers.py`. Delete duplicated versions in `src/ritrova/db/paths.py:14-75` where they overlap.
4. Move routes into routers by aggregate, using `APIRouter`:
   - `routers/findings.py` — `/api/findings/*`
   - `routers/clusters.py` — `/api/clusters/*`, `/api/cluster-stats`
   - `routers/subjects.py` — `/api/subjects/*`
   - `routers/circles.py` — `/api/circles/*`, `/api/subjects/{id}/circles`
   - `routers/together.py` — `/api/together/*`, `/together` page
   - `routers/images.py` — `/api/sources/*`, `/api/findings/*/thumbnail`
   - `routers/undo.py` — `/api/undo/*`
   - `routers/pages.py` — every Jinja page route (`/{kind}/...`, `/`, `/search`, `/circles`, …)
5. `create_app` includes routers **in this exact order**: all `/api` routers first, `pages` router last. Add a module-level assertion/comment tying the order to the `CLAUDE.md:16` rule.
6. Split `src/ritrova/static/app.js` (743 lines) into ES modules:
   - `static/js/fetch-wrapper.js` — global fetch interceptor + `_safeToast`
   - `static/js/undo.js` — `_applyUndo`, `window.showUndoToast`, `z` keybinding
   - `static/js/dialogs.js` — `window.confirmDialog`, `window.claimFaces`
   - `static/js/alpine-components.js` — every `faceSelection`, `subjectPicker`, etc.
   - Keep `static/app.js` as a thin bootstrap that `import`s them, served with `type="module"` in `base.html`. Bump `?v=` in `base.html`.
7. Tests: split `tests/test_app.py` into `tests/routers/test_{findings,clusters,…}.py`. Each router test constructs only its router + a FaceDB fixture, no full app.

### Acceptance

- `app.py` (or `app/__init__.py`) ≤ 200 lines.
- No single router > 400 lines. No helper module > 200 lines.
- All existing tests pass. New per-router tests added.
- Manual smoke: `uv run ritrova serve` → every page loads, claim-faces / dismiss / undo work.

### Risk

Route-ordering bugs (a page route shadowing an API route). Mitigation: mount order is enforced by `include_router` order in `create_app`; add a unit test that asserts `/api/findings/1` returns JSON, not HTML.

---

## M2 — Rename `Finding.person_id` → `subject_id`; drop aliased JOINs; type the XOR invariant  (Effort: **M**)

**Depends on:** M1 (routers already split → fewer merge-conflict surfaces).

**Outcome.** Code speaks the DB's vocabulary. `Finding` carries a typed curation field; the XOR invariant (at most one of `subject_id` / `exclusion_reason`) is a type-level truth.

### Concrete steps

1. In `src/ritrova/db/findings.py:27-29`, change the aliased SELECT to:
   ```
   fa.subject_id AS subject_id,
   fa.exclusion_reason AS exclusion_reason,
   cf.cluster_id AS cluster_id
   ```
2. `src/ritrova/db/models.py:31-32` — rename `person_id` → `subject_id` on the `Finding` dataclass.
3. Introduce in `src/ritrova/db/models.py` a typed curation union:
   ```python
   @dataclass(frozen=True)
   class Uncurated: pass
   @dataclass(frozen=True)
   class AssignedTo:
       subject_id: int
   @dataclass(frozen=True)
   class Excluded:
       reason: Literal["stranger", "not_a_face"]
   Curation = Uncurated | AssignedTo | Excluded
   ```
   Add `Finding.curation: Curation` as a computed property that reads `subject_id` + `exclusion_reason`. Callers should prefer `match`-ing on `curation`. (Keep the raw fields for now — M3 will consume them via curation.)
4. Grep-and-rename across the codebase: 104 occurrences of `person_id` in 21 files (per audit). Exceptions: undo-payload field names (`FindingPersonSnapshot`, `RestorePersonIdsPayload`) should also rename → `FindingSubjectSnapshot`, `RestoreSubjectIdsPayload`. Class renames: no migration needed — undo store is in-memory, single-slot.
5. Jinja templates: rename `data-person-id` → `data-subject-id`, any `{{ finding.person_id }}` → `{{ finding.subject_id }}` (BUG-21 area — double-check each template).
6. `static/app.js` (and its ES-module split from M1): rename any `person_id` refs in JS.
7. Delete the ad-hoc SQL in `app.py:389-406, 421-427` that skipped `_FINDING_FROM` — replace with macro use.
8. Single commit for the rename, small follow-up commits for the typed curation union.

### Acceptance

- `rg -w person_id src/` returns zero hits except the DB column in the (deleted) migration logic of `connection.py` (historic migration code).
- mypy passes. All tests pass.
- Manual smoke: photo page renders, cluster page renders, stranger-marking still works.

### Risk

Forgotten template attribute rename → broken Alpine binding. Mitigation: Grep templates for `person[-_]id` before commit; visual pass in browser.

---

## M3 — Thin domain-service layer above FaceDB  (Effort: **L**)

**Depends on:** M1 + M2.

**Outcome.** Routes call services, services orchestrate snapshot + mutate + register-undo in one method. FaceDB stays the SQL layer; no repository pattern.

### Concrete steps

1. Create `src/ritrova/services_domain/` (name avoids clash with existing `services.py` which will be absorbed):
   - `curation_service.py` — `CurationService(db, undo)` with methods `dismiss_findings_as_cluster(cluster_id) → UndoToken`, `mark_strangers(finding_ids) → UndoToken`, `unassign_findings(finding_ids) → UndoToken`, `restore_from_stranger(cluster_id) → UndoToken`.
   - `cluster_service.py` — `ClusterService(db)` for cluster-level reads/merges that don't need undo orchestration, plus `merge_clusters(source, target) → UndoToken`.
   - `subject_service.py` — `SubjectService(db, undo)` for `claim_faces(subject_id, finding_ids, force=False) → UndoToken`, `create_subject`, `rename_subject`, `delete_subject`.
   - `circles_service.py` — membership mutations with undo.
2. Each service method that mutates state takes a `db.transaction()` context (new primitive — see M4). For now, do snapshot + mutate back-to-back; M4 wraps them in the transaction.
3. Migrate routes: replace inline `db.snapshot_findings_fields(…) + db.X(…) + undo_store.put(…)` blocks (10+ sites in `app.py`, now in per-router files after M1) with one service call.
4. **Kill the duplicate persistence path.** `src/ritrova/scanner.py:144-319` writes findings inline. Route `rescan` through `AnalysisPipeline` + `AnalysisPersister`. If there's a real reason scanner writes inline (streaming progress?), document it; otherwise delete the inline writes. *(Superseded April 2026: the `rescan` command itself was deleted — `analyse <path> --force` now covers single-source re-analysis via the same pipeline.)*
5. `src/ritrova/services.py` (badly named — holds `compute_cluster_hint`, `compute_singleton_hints`) — rename to `ritrova/hints.py` and keep as pure functions (they don't mutate).
6. Tests: per-service unit tests under `tests/services_domain/test_{curation,cluster,subject,circles}_service.py`. Each constructs a FaceDB + UndoStore fixture, asserts before/after state + undo token replay restores state.

### Acceptance

- No `snapshot_findings_fields` call appears in any router file. Every undo-bearing mutation is a service call.
- `scanner.py` no longer imports `FaceDB.add_findings_batch` directly.
- At least 8 new service-level tests.
- Cross-mixin `# type: ignore[attr-defined]` count stays flat or decreases (this milestone doesn't yet eliminate them — that's follow-up).

### Risk

Scope creep toward a repository pattern. Mitigation: services are dumb coordinators, ≤ 200 lines each. Every method either: (a) reads DB and returns a DTO, (b) mutates DB and returns an undo token. No caching, no ORM, no query builders.

---

## M4 — Single-transaction snapshot+mutate inside services  (Effort: **S**)

**Depends on:** M3.

**Outcome.** The race window at `src/ritrova/undo.py:244-249` closes. Snapshot + mutate are atomic under one lock acquisition.

### Concrete steps

1. Add `FaceDB.transaction()` context manager in `src/ritrova/db/connection.py` that (a) acquires the RLock, (b) begins a SQLite transaction, (c) commits on normal exit, rolls back on exception. It must compose with the existing `@_locked` decorator (reentrant RLock).
2. In each service method that currently does snapshot-then-mutate, wrap both in `with db.transaction():`.
3. Remove the race comment at `undo.py:244-249`.
4. Stress test: a new test that spawns two threads each calling `dismiss_findings_as_cluster(X)` on the same cluster and asserts exactly one wins, the loser sees an already-excluded state.

### Acceptance

- `undo.py` no longer carries the race-acceptance comment.
- Multi-threaded stress test passes.

### Risk

Transaction scoping around methods that already hold `@_locked` on sub-calls — RLock is reentrant so this should be fine, but verify with a deadlock-detection test (thread blocked > 5s counts as a deadlock).

---

## M5 (optional, YAGNI) — Persistent undo store  (Effort: **S**)

**Only do this** if multi-tab support or server-restart undo survival becomes a real need. Today the user is single-session on a single-user desktop — in-memory, single-slot is sufficient.

### Concrete steps (when triggered)

1. Add `undo_entries(id TEXT PRIMARY KEY, created_at TEXT, description TEXT, payload_type TEXT, payload_json TEXT, expires_at TEXT)` table.
2. `UndoStore.put` inserts; `UndoStore.pop` deletes-and-returns. Cleanup pass on app startup purges expired.
3. Keep 60s TTL (`undo.py:252`).

### Acceptance

- Undo token survives `uv run ritrova serve` restart.

---

## Verification across milestones

After each milestone:

```bash
uv run pytest tests/ -q --ignore=tests/test_e2e_undo.py
uv run mypy src/
uv run ruff check src/
uv run ruff format --check src/
```

Plus manual smoke against `data/faces.db` (or a copy):

1. Page load: `/people/clusters`, `/pets/clusters`, `/circles`.
2. Claim-faces with cross-species (expect confirm dialog).
3. Mark-stranger on a cluster (expect next-unnamed redirect).
4. Dismiss from selection bar (expect undo toast, `z` restores).
5. `ritrova cluster --auto-merge-threshold 100` — still produces reasonable pet clusters.

## Out of scope

- Further optimisation of auto-merge for pets (tracked separately — SigLIP isn't identity-discriminative).
- VLM / caption pipeline changes (ADR-010 covers that).
- Auth/session (`docs/auth-session-design-home-server.md` — after M3 when services become the permission boundary).
- Any DB schema change. This refactor is Python-side only.
