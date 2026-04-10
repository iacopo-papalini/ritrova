# Bug Tracker

## Open

### BUG-5: Thumbnails in data/ dir get indexed by pet scanner
**Page:** N/A (scanner)
**Reported:** 2026-04-10
**Status:** Fix applied, pending verification
**Fix:** Implemented `.fr_exclude` marker file. Scanner skips any directory containing this file. Created marker in `data/`. Cleaned 14,237 bogus DB entries.
**To close:** Run a new pet scan and verify directories with `.fr_exclude` are excluded.

### BUG-6: After assigning a pet cluster, page doesn't navigate
**Page:** /clusters/{id} (pet clusters only — works for humans)
**Reported:** 2026-04-10
**Status:** Fix applied, pending verification
**Description:** Click Assign on a pet cluster detail page. The assignment succeeds (DB updated) but the browser stays on the same page instead of navigating to the next similar cluster. Works correctly for human clusters.
**Root cause:** `find_similar_cluster()` was not species-aware — compared 768-dim pet embeddings against 512-dim human clusters.
**Fix:** Added `species` parameter to `find_similar_cluster()`, passed from `_next_similar_cluster()`.

### BUG-12: Video frame photos return 404
**Page:** /persons/{id} (photos tab), any page showing video-sourced photos
**Reported:** 2026-04-10
**Status:** Fixed
**Description:** Photos extracted from videos have paths like `tmp/frames/vid_xxx.jpg` in the DB. After moving data to `data/`, `resolve_path` tries `PHOTOS_DIR/tmp/frames/...` which doesn't exist. Actual files are at `data/tmp/frames/...`.
**Fix:** `resolve_path()` now falls back to `db_path.parent / path` when `base_dir / path` doesn't exist, covering locally-generated files (video frames, etc.).

### FEAT-6: Find photos containing multiple specific people/pets
**Reported:** 2026-04-10
**Status:** Fixed
**Description:** /together page — multi-select typeahead picker, cross-kind (human + pet). DB query intersects photo_ids across all selected persons. Results grouped by month. Nav link added.
**Fix:** `db.get_photos_with_all_persons()`, `/together` page, `/api/together` + `/api/together-html` endpoints.

### FEAT-7: Generic search/filter across all metadata
**Reported:** 2026-04-10
**Status:** Open
**Description:** Unified search that queries across person/pet names, photo paths, dates, and (when available) tags/descriptions. Filter results by date range, person, location. Currently search only matches person names via `db.search_persons()`.

### FEAT-9: Trigger background tasks from web UI
**Reported:** 2026-04-10
**Status:** Open
**Description:** Run scan, scan-pets, cluster, auto-assign, cleanup, backfill-gps, and describe as background tasks from the web UI instead of requiring CLI access. Dashboard should show a "Tasks" panel with buttons to launch jobs, a running indicator, and a log tail. Tasks run in a background thread/process so the UI stays responsive. Progress reported via SSE (Server-Sent Events) or polling. Only one task at a time (queue or reject if busy).

### FEAT-8: Photo scene descriptions and tags via computer vision
**Reported:** 2026-04-10
**Status:** Open
**Description:** Use a vision-language model (e.g. SigLIP, BLIP-2, or LLaVA) to generate natural language descriptions and tags for photos ("people at a table", "christmas tree", "beach sunset", "birthday cake"). Store as searchable text in DB. Enables FEAT-7 to search by scene content, not just faces. Should run on Apple Silicon GPU (MPS/CoreML). Could reuse the SigLIP model already loaded for pet embeddings, or use a dedicated captioning model.
**Design notes:** New DB table `photo_tags(photo_id, tag, source)` + `photo_descriptions(photo_id, description)`. New CLI command `ritrova describe` to batch-process. Tags surfaced in photo view and searchable via FEAT-7.

### FEAT-12: Person avatar in typeahead picker
**Reported:** 2026-04-10
**Status:** Open
**Description:** The /api/persons/all endpoint should include a representative face_id per person so the typeahead picker can show a small face thumbnail next to each name. Makes it much easier to pick the right person when names are similar.

### BUG-16: Person card inconsistent between directory and search results
**Page:** /search
**Reported:** 2026-04-10
**Status:** Fixed
**Description:** Person cards in the directory show face avatar + name + count, but search results show only name + count (no avatar). Same entity, different rendering — violates UI consistency.
**Fix:** Extract person card into a shared partial, use it in both persons.html and search.html.

### FEAT-11: Browser-side image rotation in lightbox
**Reported:** 2026-04-10
**Status:** Fixed
**Description:** Add a rotate button in the lightbox for manually rotating images (90° steps). Useful when EXIF orientation is missing or wrong. CSS `image-orientation: from-image` handles most cases, but some old photos need manual rotation.

### BUG-15: Broken/truncated images cause 500 stacktraces
**Page:** Any page serving images (thumbnails, photo view, lightbox)
**Reported:** 2026-04-10
**Status:** Fixed
**Description:** Corrupted or truncated JPEG files crash PIL with `OSError: broken data stream`. 
**Fix:** Catch `OSError` in `face_thumbnail` and `photo_image` endpoints, return 404 instead of 500.

### FEAT-10: Person/pet avatar thumbnail + contextual label in directory
**Reported:** 2026-04-10
**Status:** Open
**Description:** The Names directory (/{kind}) should show a representative face thumbnail for each person/pet, not just name + count. Pick a random (or best-quality) face sample as avatar. Also rename the nav label from "Names" to match the context: "Persons" when in people mode, "Pets" when in pets mode.

### BUG-14: Pet cluster assign dropdown shows 0 face count for all persons
**Page:** /clusters/{id} (pet clusters)
**Reported:** 2026-04-10
**Status:** Fixed
**Description:** The person dropdown in cluster detail showed "Figaro (0)" for all pet persons.
**Root cause:** `rank_persons_for_cluster()` hardcoded face_count to 0 instead of fetching actual counts.
**Fix:** Fetch counts from `get_persons_by_species()` and populate the tuple.

### FEAT-4: "Not this person" button on face samples
**Page:** /{kind}/{id}
**Reported:** 2026-04-10
**Status:** Fixed
**Description:** Per-face "x" button on hover in face samples tab, calls `/api/faces/{id}/unassign` and removes the face from the grid. Works on both initial and htmx-loaded faces.

### FEAT-5: Global single-step UNDO for all write operations
**Reported:** 2026-04-10
**Status:** Open
**Description:** Every write operation (merge, assign, unassign, dismiss, exclude, rename, delete person, create person) should be undoable with a single UNDO action. No stack needed — just the last operation. After any write, show a toast/snackbar with "Undo" button. Clicking it reverses the operation. The undo state is replaced by the next write operation.
**Design notes:** Requires storing the inverse operation (e.g. assign face 42 to person 5 → undo = unassign face 42, or assign back to previous person). Could use a simple `last_action` record in the DB or an in-memory store on the server. The undo button should be visible globally (e.g. fixed bottom-right toast) and disappear after ~15 seconds or on the next write.

### BUG-13: Month grouping uses unreliable EXIF dates
**Page:** /persons/{id}
**Reported:** 2026-04-10
**Status:** Fixed
**Description:** Face/photo month grouping used `taken_at` from EXIF, which can be wrong (e.g. misconfigured camera clock showing 2014 for a 2006 photo). The directory path contains the real date (`2006/2006-03-12.Event/...`).
**Fix:** Extract YYYY-MM from directory names in the file path instead of EXIF. Sort by file_path (which is date-ordered). Extracted `_group_by_month()` helper to avoid duplication.

## Closed

### BUG-1: Cluster hint doesn't disappear on mouse leave
**Page:** /clusters
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** Added Alpine `x-data` with `showHint` toggle on mouseenter/mouseleave; htmx loads hint content once, Alpine controls visibility.

### BUG-2: Lightbox must show photo full path
**Page:** All pages with lightbox
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** Added `photoPath` to Alpine lightbox store, fetches `/api/photos/{id}/info` on open. Path displayed below image.

### BUG-3: "Create new pet" doesn't work
**Page:** /clusters/{id} (pet clusters)
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** Fixed `createNew` init (was `false` even when only `__new__` option existed). Simplified form to build FormData manually.

### BUG-4: No way to remove a single face from a cluster
**Page:** /clusters/{id}
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** Added per-face "x" button (visible on hover) that calls `/api/faces/exclude` and removes the element from DOM.

### BUG-7: Nav shows "Humans" on pet cluster detail pages
**Page:** /clusters/{id} (pet clusters)
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** Added `species` to cluster_detail template context. Made nav pills server-rendered (no Alpine needed).

### BUG-8: "Merge clusters" button returns 422
**Page:** /merge-suggestions
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** Changed endpoint from `Body(...)` to `Form(...)`. Changed template from `hx-vals` to `<form>` with hidden inputs.

### BUG-9: No merge suggestions for pets
**Page:** /merge-suggestions
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** Nav links now carry `?species=pet` when in pet mode (FEAT-1). Merge suggestions page already accepted species param.

### BUG-10: No compare for pets
**Page:** /compare
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** Compare page now accepts `species` query param. Dropdown filtered by `get_persons_by_species(species)` instead of `get_persons()`.

### BUG-11: Infinite scroll on person detail pushes Photos section off-screen
**Page:** /persons/{id}
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** Replaced stacked sections with Alpine tabs (Face Samples / Photos). Each tab has independent content.

### FEAT-1: Humans/Pets toggle should be a consistent global toggle
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** All 5 missing routes now pass `species` to template context. Nav links in base.html all carry `?species=pet` when in pet mode. Toggle pills are server-rendered. Compare and merge-suggestions are now species-aware.

### FEAT-2: Sort faces and photos by time, group by month
**Page:** /persons/{id}
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** Added `get_person_faces_with_dates()` DB method (JOIN with photos, ORDER BY taken_at). Person detail groups faces and photos by month with section headers.

### FEAT-3: Show GPS location in lightbox, link to Google Maps
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** Added latitude/longitude columns to photos table (migration). GPS extracted from EXIF during scan. Added `backfill-gps` CLI command. Lightbox shows map pin icon linking to Google Maps when GPS available.
