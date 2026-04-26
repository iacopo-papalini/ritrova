# Feature Backlog

## Open

### FEAT-30: Split selected cluster faces into a new cluster
**Reported:** 2026-04-25 | **Closed:** 2026-04-25
**Shipped:**
- Cluster detail selection bar now includes **Split into new cluster**.
- Selected findings are moved to a fresh `cluster_id`; stale selections outside the current cluster are ignored.
- The action is undoable via the existing FEAT-5 toast path, restoring moved findings to the original cluster.
- After splitting, the original cluster page removes the selected tiles in place and shows an "Open" toast for the new cluster.

### FEAT-31: Cluster detail queue header
**Reported:** 2026-04-25 | **Closed:** 2026-04-25
**Shipped:**
- Cluster detail now uses the shared `partials/entity_header.html` shape.
- Header shows unnamed-queue position (`N of M unnamed`) when the cluster is in the review queue.
- Previous / Skip / Next controls are available in the header; `s` skips to the next cluster when focus is not in an input.
- "Dismiss entire cluster" moved out of the primary assignment row into the header overflow menu to reduce destructive-action adjacency.
- Cluster selected-face Exclude / Not-a-face actions now patch the grid in place and show undo toasts instead of reloading the page.

### FEAT-32: Fold standalone search into scoped names filter
**Reported:** 2026-04-25 | **Closed:** 2026-04-25
**Shipped:**
- Top-nav search now submits to the current scoped names page (`/{kind}?filter=...`) instead of `/search`.
- `/{kind}` seeds its live client-side filter from the `filter` query param.
- Legacy `/search?q=...` redirects to `/people?filter=...`.
- Removed the redundant `search.html` standalone page.

### FEAT-33: Browse sources by people, path tags, and archive dates
**Reported:** 2026-04-25 | **Status:** In progress
**Shipped (foundation, 2026-04-25):**
- Added `source_path_metadata` as the canonical cache for path-derived `date_text`, date precision/source, conflict flags, and normalized path tags.
- Added a pure parser for archive paths. Directory dates win filename conflicts and bogus filename placeholders; EXIF is fallback only.
- New sources populate path metadata automatically; existing archives can run `ritrova backfill-path-metadata`.
- Added shared `SourceSearchMixin` for composable source browsing filters: subject IDs, path tags, path date range, source type, and "just selected people".
- Added `/browse` and `/api/browse-html`, using the same reusable source grid partial as Together.

**Still open:**
- Autocomplete / suggestion UI for high-value path tags.
- Date-confidence indicators in the browse grid for conflict / EXIF-fallback cases.
- Optional folding of Together into Browse once filter ergonomics cover that workflow cleanly.

### FEAT-34: Persistent print selection and zip export
**Reported:** 2026-04-26 | **Closed:** 2026-04-26
**Shipped:**
- Added a persistent ordered `print_selection` worklist for photo sources only.
- Added Print toggles on source grids, the lightbox toolbar, and photo detail pages.
- Added `/print` review page with remove / clear actions.
- Added `/api/print-selection/export`, which returns an ordered zip of original photo files with stable `0001_...` filenames.
- Videos are rejected at selection time; printing-service API integration remains out of scope.

### FEAT-29: Manual finding — Shift+drag on the photo to create a bbox, with nearest-subject suggestion
**Reported:** 2026-04-21 | **Priority:** Medium
Detectors miss faces in hard cases (profile, occlusion, low light, distant subjects). Today the only recourse is accepting the miss. Let the user draw a bounding box manually on the photo viewer to create a new finding, have the server compute its embedding AND suggest the nearest named subject based on centroid similarity, then confirm or override in one keystroke.

**Shipped (MVP, 2026-04-20):** `POST /api/sources/{id}/findings` with body `{bbox, species}`; `FindingsService.create_manual` wires embedding + nearest-named-subject suggestion + `DeleteManualFindingPayload` undo; photo.html gains a Shift+drag Alpine component (`manualFinding`) with a species popover and suggestion-prefilled picker on page reload. Videos are rejected at 422. Phase 2 (video frame scrubber, lightbox gesture, cluster-scope bulk) remains open.

**UI surface (photo page):**
- On the photo viewer's `.relative` wrapper: hold **Shift** and press-drag to draw a live rectangle (SVG/DOM via an Alpine component). Plain clicks stay reserved for the existing bbox-overlay jump-to-tile affordance.
- On release: small popover near the drawn box — species radio (default from URL `kind`, but user can override for a human-in-a-pets page or vice versa), confirm button, escape cancels.
- POST `/api/sources/{id}/findings {bbox, species}` with bbox in source-pixel coordinates.
- On success: a new tile is appended to the "Faces in this photo" grid in **Pending** state (FEAT-28 vocabulary), and a new bbox overlay appears on the main image (red, like any pending finding).

**Two-keystroke common case:**
- Shift-drag → popover → Enter (accept default species).
- The server returns `suggestion: {subject_id, name, similarity_pct}` based on nearest named-subject centroid (reuse `rank_subjects_for_cluster` semantics, top-1 above ~55% cosine sim). The new tile's subject picker is **prefilled** with the suggested name.
- User presses **Enter** in the picker → assigns. Total: Shift-drag, species Enter, name Enter. Three keystrokes for a full manual find + assign. Backspace/type to override.

**Server work:**
- New endpoint `POST /api/sources/{id}/findings` — body `{bbox: [x, y, w, h], species: "human"|"dog"|"cat"}`. Validates bbox ⊆ source dimensions and above a small area threshold. 401/403 not applicable (single-user desktop).
- New service method `FindingsService.create_manual(source_id, bbox, species) → CreateManualResult` (or extend `SubjectService` — pick whichever is cleanest). Returns `{finding_id, suggestion, receipt}` where `receipt` is an `UndoReceipt` whose payload deletes the created finding on undo.
- Inside the service: load the image once, crop to bbox, embed using the correct model (ArcFace for human, SigLIP for pet), insert a `findings` row tied to the source's most-recent `subjects` scan, `detected_at=now`, `confidence=1.0` (user-asserted), `frame_path=None` for photo sources. All inside `db.transaction()` so the insert+undo-snapshot are atomic.
- New helper `nearest_named_subject(db, embedding, species, min_similarity=0.55) -> (subject_id, name, similarity_pct) | None` in `cluster.py`. Small generalization of the existing `rank_subjects_for_cluster` path — takes a raw normalized embedding instead of a cluster-centroid.
- Extract `embed_crop(image: PIL.Image, bbox) → np.ndarray` on `FaceDetector` / `PetDetector` if not already exposed (detection is internally `crop → embed`; expose just the embed step).
- New undo payload `DeleteManualFindingPayload(finding_id)` in `undo.py` — on undo, delete the created finding row (CASCADE takes care of finding_assignment / cluster_findings).

**API shape (response):**
```json
{
  "ok": true,
  "finding_id": 12345,
  "suggestion": {"subject_id": 7, "name": "Caterina", "similarity_pct": 78.4},  // or null
  "undo_token": "...",
  "message": "Added manual face"
}
```

**Edge cases:**
- **Video sources:** disabled at the endpoint (422) — a single-frame JPG isn't what the user's looking at; proper support needs a frame scrubber (Phase 2). The photo page for a video source today renders a single extracted frame (see BUG-22); don't compound that.
- **Overlap with an existing finding:** allowed. Dedup happens at clustering time, not at the per-photo layer.
- **Suggestion species mismatch with URL kind:** none — suggestions are filtered to subjects whose kind matches the manually-picked species.
- **No subjects of the matching kind exist yet:** `suggestion: null`. User picks from scratch, same as any Pending tile.

**MVP scope:**
1. Server endpoint + `FindingsService.create_manual` with nearest-subject suggestion + undo token.
2. Photo page: Shift+drag component, species popover, appended tile + overlay with prefilled picker.
3. Videos explicitly disabled at the endpoint with a clear error message.
4. Tests: round-trip create → suggestion correctness → undo deletes the row.

**Phase 2:**
- Video support via frame scrubber.
- Lightbox support (same gesture on the lightbox image).
- Cluster-scope bulk: "for every manual finding I just added, find the nearest subject in one pass".

### FEAT-28: Photo-page face tiles — inline re-assign + "Not a face" action
**Reported:** 2026-04-21 | **Priority:** Medium
Two related UX gaps on the per-face tile in `/photos/{id}` ("Faces in this photo"):

**(a) Clicking the name currently navigates away instead of re-assigning.**
An assigned face renders the subject name as a plain `<a>` to `/{kind}/{subject_id}` (`templates/photo.html:93-97`). That navigation is redundant with the global nav links and gets in the way of the frequent workflow "this face is mis-assigned; re-assign it". Today the only way to correct a wrong name is to click the red X to unassign, then open the picker — two steps for a single mental operation.

Desired: clicking the name chip flips the tile into picker mode — the same `subjectPicker` component already used for unassigned faces drops in, pre-filtered, and selecting a name re-assigns the finding via the existing `/api/findings/{id}/assign` flow (which handles 409 cross-species confirm out of the box). To navigate to the subject page, use the global typeahead or the person's avatar elsewhere.

**(b) No "Not a face" action on the tile.**
When the user is looking at a photo and sees that one of the detected "faces" is actually a doorknob / reflection / background pattern, the only way to dismiss it is to go to `/singletons` (or the cluster it's in) and select it from a bulk action. On the photo itself, where the error is visible, there's no per-tile action for it. Add a small "Not a face" button (trash/forbidden icon, muted colour) on every tile regardless of assignment state — sometimes a mis-assigned "face" is actually not a face at all. Clicking calls `POST /api/findings/dismiss` with a confirm dialog (same copy as the selection bar), with undo.

**Scope:**
- Photo page only. Other places that render subject chips (subject detail, together page, cluster detail) keep their navigation affordance.
- Alpine: `x-data="{ editing: false }"` on each face tile. Click on the name toggles `editing = true` and swaps the chip for the `subjectPicker`; `onSelect` POSTs to `/api/findings/{id}/assign` (server-side overwrite of prior assignment). Escape closes without POSTing.
- "Not a face" button visible on every tile (assigned + unassigned). Confirm dialog, then `POST /api/findings/dismiss` with `{face_ids: [id]}`. On success, the tile animates out and the bbox overlay on the main photo disappears (DOM update, no reload). Undo token surfaces via the existing toast.
- Small visual cue the name is clickable for editing (e.g. pencil icon on hover) — the loss of navigation shouldn't be a silent surprise.

**Relationship to BUG-19 / claim-faces:** the server-side 409 needs_confirm flow is already in place; (a) is a client-side reshuffle only. `assign_finding_to_subject` already overwrites any prior assignment.

**Effort:** S. One template change in `photo.html`, ~30 lines of Alpine state + a bit of icon markup.

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
**Shipped:** Filter input on `/{kind}` page, client-side Alpine filtering over the subjects grid. Count updates live. "No matches" message when filter yields nothing. `/{kind}?filter=...` seeds the filter from navigation search; legacy `/search` redirects there.

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
