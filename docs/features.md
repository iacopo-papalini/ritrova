# Feature Backlog

## Open

### FEAT-27: Circles of people/pets for view filtering
**Reported:** 2026-04-18 | **Priority:** Medium
A subject (named person or pet) can belong to zero, one, or several **circles** — user-defined labelled groups like `family`, `close-friends`, `acquaintances`, `parenti lontani`, `colleagues`, `strangers`. Circles are filters applied to any view that lists photos or subjects: "exclude members of *acquaintances* and *strangers*", or conversely "only show *family*". Primary use case is **exclusion** — sweeping acquaintances + strangers out of year/together/photo views to keep the archive feeling like a family album without having to delete anything.

**Schema (new tables, minimal):**
```sql
CREATE TABLE circles (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  name        TEXT NOT NULL UNIQUE,   -- UI label
  description TEXT,                   -- optional freeform note
  created_at  TEXT NOT NULL
);
CREATE TABLE subject_circles (
  subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
  circle_id  INTEGER NOT NULL REFERENCES circles(id)  ON DELETE CASCADE,
  added_at   TEXT NOT NULL,
  PRIMARY KEY (subject_id, circle_id)
);
CREATE INDEX idx_subject_circles_circle ON subject_circles(circle_id);
```
Many-to-many by design: a subject can be in several circles at once (e.g. `family` + `everyday`), and a circle spans both humans and pets.

**UI surface:**
- New `/circles` page: list circles, counts of members, create / rename / delete.
- Subject detail pages (person/pet): a chip row showing current circles, add/remove inline (typeahead like the subject-assign picker).
- View filters: every listing page (year, together, subjects, clusters, photo) gains a compact "circles" dropdown with **include** and **exclude** multi-selects. Exclude wins over include when both are set on the same circle.
- Bulk action on subject list: "add selected to circle…" / "remove from circle…".

**CLI peers:**
- `ritrova circles list` / `create <name>` / `rename <old> <new>` / `delete <name>`.
- `ritrova circles add <circle> <subject…>` / `remove <circle> <subject…>`.
- `ritrova circles members <circle>`.

**Exclusion semantics in views:**
A photo is "hidden" by an `exclude=acquaintances` filter if **every named subject** on it is in `acquaintances` — otherwise it stays visible. A photo of a family member + an acquaintance stays visible (the family member is the reason you want the photo). Unnamed / unclustered findings don't count — they can't be "in a circle" until they're named. This avoids losing random strangers-in-the-background photos just because one circle-tagged person happens to be in them.

**Relationship to existing features:**
- Subjects table (named people/pets) is the membership source — circles reference it, don't duplicate it.
- FEAT-5 undo: every circle membership change (add / remove / delete-circle) goes through the existing `UndoPayload` ABC. Deleting a circle captures its members in the payload so undo restores both the circle row and its memberships.
- FEAT-7 (generic search): once circles exist, they become another metadata axis — `circle:family -circle:acquaintances` as search syntax.
- Together view / year view: both gain the include/exclude filter naturally.

**MVP scope (first PR):**
1. Migration: create the two tables.
2. `DB` mixin: circle CRUD + add/remove membership, with `UndoPayload` integrations.
3. `/circles` page: create, rename, delete circles.
4. Subject detail: list circles of subject + add/remove.
5. One view filter wired up end-to-end (e.g. Subjects page `?exclude=acquaintances`).
6. CLI subcommands above.

**Phase 2:**
- Filter surface expanded to Year / Together / Photo / Cluster pages.
- Bulk-add UI on Subjects list.
- Search syntax integration (deferred to FEAT-7).

### FEAT-20: Together — filter by source type (photos / videos / either)
**Reported:** 2026-04-13 | **Closed:** 2026-04-15
**Shipped:**
- `db.get_sources_with_all_subjects` and `count_sources_with_all_subjects` accept an optional `source_type` arg (`"photo"` / `"video"`). Filter is applied on the outer join (`WHERE s.type = ?`) so the together inner subquery stays species-agnostic.
- `/api/together` and `/api/together-html` accept `source_type=either|photo|video` (default `either`); anything else normalises to `either`. The htmx pagination URL in `together_results.html` now threads `source_type` through alongside `alone`.
- 3-way pill selector on `/together` next to the "Just them" checkbox. Alpine state: `sourceType` ∈ `{'either', 'photo', 'video'}`.
- Bonus: `/api/sources/{id}/image` now falls back to a representative finding's extracted frame for **video** sources (previously 404'd), so video thumbnails in Together results and subject_detail's Photos tab render cleanly. Play-icon overlay on video cards in Together results makes the type obvious.
- Tests: 5 new cases (either/photo/video filter, invalid source_type, video thumbnail fallback, video with no frames → 404).

### FEAT-16: Lightbox → open source detail page in new tab
**Reported:** 2026-04-11 | **Closed:** 2026-04-13
**Shipped:** External-link button added to the lightbox toolbar (left of Rotate). Links to `/photos/{photoId}` with `target="_blank" rel="noopener"`. Hidden via `x-show` when no photo is loaded.

### FEAT-17: Download original file from lightbox and source detail page
**Reported:** 2026-04-11 | **Closed:** 2026-04-13
**Shipped:** `GET /api/sources/{id}/original` returns the raw file via `FileResponse(filename=…)` — Starlette emits `Content-Disposition: attachment` and content-type is inferred from the extension. Works for both photo and video sources. Download buttons:
- Lightbox toolbar — icon-only, left of the "Open details" link.
- `/photos/{id}` header — labelled button at the right end of the header row.
Client uses the `download` HTML attribute as a belt-and-suspenders on top of the server disposition. 3 tests cover: successful download (body == raw bytes, filename in Content-Disposition), 404 on unknown source, 404 when the file is missing from disk.

### FEAT-18: Arrow key navigation in lightbox
**Reported:** 2026-04-11 | **Closed:** 2026-04-13
**Shipped:** Lightbox store refactored to hold an `items[]` list with an `index` cursor and `next()`/`prev()` methods (no wrap-around — arrow at end is a no-op). Sibling list is **derived from the DOM at click time** via the new `openFromGrid($el, mode)` helper: it walks to the nearest `[data-lightbox-group]` ancestor and collects all siblings carrying `data-finding-id` (or `data-source-id` for source-mode grids like Together). This works correctly with htmx-paginated grids because the DOM is authoritative when the user clicks. Visible left/right chevron buttons appear when `items.length > 1`, disabled at the ends. ArrowLeft/ArrowRight key handlers live on the lightbox overlay in `base.html`. Backwards-compatible: legacy `$store.lightbox.show(sourceId)` still works as a single-item open. Converted call sites: `face_grid`, `singleton_grid`, `subject_finding_grid`, `together_results`, `cluster_detail`, `singletons`, `subject_detail` (Faces tab).

### FEAT-19: Replace browser confirm() with app dialog
**Reported:** 2026-04-11 | **Closed:** 2026-04-13
**Shipped:** Alpine `$store.dialog` + reusable `partials/dialog.html` (backdrop-blurred, scale-in transition, auto-focuses the confirm button, Escape/backdrop click cancels). Promise-based API:
```js
const ok = await confirmDialog({ title, message, confirmLabel, cancelLabel, danger });
```
`window.confirmDialog(opts)` is the global helper; falls back to native `window.confirm` if invoked before Alpine boots. All 11 in-app `confirm()` call sites converted (`face_grid.html`, `subject_finding_grid.html`, `singletons.html`, `subject_detail.html` ×3, `cluster_detail.html` ×4, `photo.html`). Cross-species needs_confirm flow (BUG-19) now uses the dialog cleanly with a neutral Assign button; destructive actions (delete, dismiss, exclude, unassign) get the red `danger` variant.

### FEAT-14: Video findings browsing — frame viewer and subject video section
**Reported:** 2026-04-11 | **Closed:** 2026-04-13
**Shipped:**
- **`GET /api/findings/{id}/frame?max_size=1600`** serves the right image regardless of source type — extracted frame JPEG for video findings, resized source image for photo findings. Reuses `db.resolve_finding_image()` so the dispatch logic lives in one place.
- **`GET /api/findings/{id}/info`** returns `{source_id, file_path, latitude, longitude, type}`. `file_path` is always the **source's** path (e.g., `wedding.mp4`), never the internal frame jpeg.
- **Lightbox is finding-aware**: callers opening from a finding grid no longer pass `source_id`; the store hits the new `/api/findings/{id}/frame` and `/info` endpoints. Source-mode is preserved for grids that iterate sources (Together).
- **Videos tab** on the subject detail page (third tab, conditional on `videos|length`). Each card shows a representative frame (first finding's thumbnail), filename, and finding count ("3 moments"). Click → opens the new full-screen `$store.videoPlayer` overlay with a native `<video controls autoplay>` streaming from `/api/sources/{id}/original`. Escape / backdrop click / X button closes.
- New DB query: `get_subject_sources_with_findings(subject_id, source_type)` pairs each source with the subject's findings on it.
- `subject_detail` route now passes `videos` and `video_groups` (grouped by month) to the template.

### FEAT-15: Together "alone" filter — exclude sources with other subjects
**Reported:** 2026-04-11 | **Closed:** 2026-04-11
**Shipped:** "Just them" checkbox on Together page. When checked, only shows sources where exactly the selected subjects appear and no other named subjects. Uses `HAVING COUNT(DISTINCT person_id) = N` on all named findings per source.

### FEAT-5: Global single-step UNDO for all write operations
**Reported:** 2026-04-10 | **Closed:** 2026-04-16
**Shipped:**
- In-memory single-slot undo store with 60s TTL. `UndoPayload` ABC — each payload subclass carries prior state and implements `undo(db)`. Five concrete payloads cover all operations: `DismissPayload`, `RestoreClusterPayload`, `RestorePersonIdsPayload`, `DeleteSubjectPayload`, `ResurrectSubjectPayload`.
- Undoable endpoints: cluster dismiss, cluster merge, cluster assign, cluster name, subject delete, subject merge, findings dismiss, findings exclude, claim-faces, swap, single-finding unassign.
- Toast with Undo button (15s), `z` keyboard shortcut. Toast dismiss clears the shortcut so `z` can't fire after the toast is gone. `POST /api/undo/{token}` + `GET /api/undo/peek` (recovers toast after redirect).
- Intentionally not undoable (result is visible and easily reversed by the user): rename subject, create subject, single-finding assign.

### FEAT-13: Inline search/filter on subjects list page
**Reported:** 2026-04-11 | **Closed:** 2026-04-11
**Shipped:** Filter input on `/{kind}` page, client-side Alpine filtering over the subjects grid. Count updates live. "No matches" message when filter yields nothing. `/search` route kept for now as fallback.

### FEAT-7: Generic search/filter across all metadata
**Reported:** 2026-04-10 | **Priority:** Medium
Unified search across names, paths, dates, tags. Filter by date range, person, location. Currently name-only.

### FEAT-8: Photo scene descriptions and tags via computer vision — **deferred (retired from default)**
**Reported:** 2026-04-10 | **Retired from default:** 2026-04-18 | **Priority:** Low
Vision-language model captioning and tagging is now opt-in via `ritrova analyse --caption`. The default pipeline is subject detection only. See ADR-011 for the rationale: the Qwen2.5-VL-7B + MarianMT en→it pipeline introduced a ~10% face-recall regression on real photos (clear close-ups of babies and aquarium scenes), produced Italian translations with systematic grammar errors (singular-they → plural "loro"), and cost ~2 s/source in throughput. Revisit when a better Italian-native VLM or integrated VLM-with-translation model exists for Apple Silicon.

### FEAT-21: Caption in face-hover tooltip — **withdrawn**
**Reported:** 2026-04-17 | **Withdrawn:** 2026-04-18
Moot after ADR-011 retired captions from the default pipeline. The `descriptions.caption` data it was going to surface is no longer populated by default.

### FEAT-22: Prune rare tags — **withdrawn**
**Reported:** 2026-04-17 | **Withdrawn:** 2026-04-18
Moot after ADR-011: the default pipeline no longer produces tags. Revisit if/when captions get re-enabled.

### FEAT-24: LLM-proposed tag normalization — **withdrawn**
**Reported:** 2026-04-17 | **Withdrawn:** 2026-04-18
Moot after ADR-011 (no default-pipeline tags to normalize). Revisit if captions get re-enabled.

### FEAT-25: Kill-list for uninformative tags — **withdrawn**
**Reported:** 2026-04-17 | **Withdrawn:** 2026-04-18
Moot after ADR-011 (no default-pipeline tags). Keep as a reference design if captions are re-enabled.

### FEAT-26: LLM caption polish sweep — **withdrawn**
**Reported:** 2026-04-17 | **Withdrawn:** 2026-04-18
Moot after ADR-011 (no captions to polish). Keep as reference design for when captions are revisited.

<details><summary>Original proposal (for archaeology)</summary>

MarianMT (`opus-mt-tc-big-en-it`) was trained on pre-2020 OPUS/EuroParl data, so it mishandles modern English constructs the VLM emits — most visibly the singular *they* / *their*, which gets rendered as plural Italian *loro* instead of the correct *sue* / *sua* / definite article. Observed in the wild: *"Un bambino su un seggiolone mangia il cibo dalle **loro** mani"* — every Italian speaker parses it instantly but it reads as broken. Class of errors is: pronoun-agreement, gender-agreement on fixed-gender nouns, literal-translation idioms, occasional wrong verb form.
Fix the corpus in a one-shot post-processing sweep rather than swapping the translator mid-archive. New CLI `ritrova captions polish [--backend mlx|anthropic] [--model <id>] [--dry-run --sample N]`. Reads every row from `descriptions`, sends `(caption, tags)` through a grammar-only LLM prompt ("fix Italian grammar/agreement only; preserve every fact; do not add or remove content"), UPDATEs the caption, keeps the original in a new `caption_original` column for auditing/undo.

**Backend comparison** (32K captions, ~100 tokens each round-trip):

| Backend | Model | Cost | Wall time | Italian quality |
|---|---|---|---|---|
| Local MLX (recommended default) | `Qwen2.5-7B-Instruct-4bit` via **mlx-lm** (not mlx-vlm — cleaner batching) | $0 | ~2.5 h batched @ 8 | catches ~80–85% of errors |
| Anthropic API | `claude-haiku-4-5` | ~$3–5 | 30–60 min | ~95% |
| Anthropic API | `claude-sonnet-4-6` | ~$15–25 | 30–60 min | 99%+ |

**Batching:** sort captions by length, batch-8 on mlx-lm, `temperature=0`. Correctness at temperature=0 is unchanged by batching (causal+padding masks isolate sequences; only sub-0.1% tied-token flips from FP non-associativity).

**Safety rails:**
- `--dry-run --sample 100` is the only way to invoke the first time — surface the rewrites in a diff-style report. Don't wire the no-dry-run path until the dry-run output has been eyeballed.
- Strict system prompt + 2–3-shot examples to stop the polisher from adding content or preamble ("Ecco la frase corretta:").
- Length-ratio sanity check per caption: if the polished caption is <0.7× or >1.4× the original word count, skip and flag (indicates hallucination or truncation).
- Every UPDATE is FEAT-5 undoable — restore from `caption_original`.
- **Idempotent**: re-running on already-polished captions should be a near-noop at `temperature=0`.

**When:** only after the full re-index finishes — otherwise we polish rows that will be overwritten.

**Relationship to FEAT-24 / FEAT-22 / FEAT-25:** orthogonal — captions here, tags there. The tag-cleanup trio touches `descriptions.tags`; this one touches `descriptions.caption`. Can run in either order.

**Upstream fix (not this feature):** swapping MarianMT for NLLB-200 or an LLM-based translator is the right long-term move, tracked separately under ADR-010 §2f. FEAT-26 is the one-shot patch for the current 32K corpus.

</details>

### FEAT-23: Periodic rare-tag trim during analyse (mid-loop) — **withdrawn**
**Reported:** 2026-04-17 | **Withdrawn:** 2026-04-18
Moot after ADR-011 (no default-pipeline tags). Revisit together with FEAT-22 if captions are re-enabled.

### FEAT-9: Trigger background tasks from web UI
**Reported:** 2026-04-10 | **Priority:** Medium
Dashboard task panel: launch scan/cluster/cleanup from UI. Background thread, SSE progress, log tail. One task at a time.

### FEAT-10: Person/pet avatar in directory — done
**Reported:** 2026-04-10 | **Closed:** 2026-04-11
Avatar + contextual label shipped. Random face selection is intentional (variety).

### FEAT-12: Person avatar in typeahead picker — done
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
Shipped inline with typeahead picker. `/api/persons/all` returns `face_id`.

## Closed

### FEAT-1: Global species toggle → kind-based routing
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
`/people/...` and `/pets/...` paths, `kind` variable, all nav links species-aware.

### FEAT-2: Sort by time, group by month
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
Path-based date extraction, `_group_by_month()` helper, person detail tabs.

### FEAT-3: GPS in lightbox
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
EXIF GPS extraction, `backfill-gps` CLI, map pin in lightbox.

### FEAT-4: "Not this person" unassign button
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
Per-face "x" button on person detail face samples.

### FEAT-6: Find photos with multiple people/pets
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
`/together` page, multi-select picker, cross-kind queries, infinite scroll.

### FEAT-11: Browser-side image rotation
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
Rotate button in lightbox, 90° steps, CSS transform.
