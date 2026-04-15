# Feature Backlog

## Open

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
**Reported:** 2026-04-10 | **Priority:** High
Every write action undoable. Toast with Undo button, auto-dismiss ~15s. In-memory inverse action store on server. Affects every write endpoint.

### FEAT-13: Inline search/filter on subjects list page
**Reported:** 2026-04-11 | **Closed:** 2026-04-11
**Shipped:** Filter input on `/{kind}` page, client-side Alpine filtering over the subjects grid. Count updates live. "No matches" message when filter yields nothing. `/search` route kept for now as fallback.

### FEAT-7: Generic search/filter across all metadata
**Reported:** 2026-04-10 | **Priority:** Medium
Unified search across names, paths, dates, tags. Filter by date range, person, location. Currently name-only.

### FEAT-8: Photo scene descriptions and tags via computer vision
**Reported:** 2026-04-10 | **Priority:** Medium
Vision-language model (BLIP-2, LLaVA) for captions/tags ("people at a table", "christmas tree"). Apple Silicon GPU. New DB tables, `ritrova describe` CLI. Enables FEAT-7 content search.

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
