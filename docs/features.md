# Feature Backlog

## Open

### FEAT-14: Video findings browsing — frame viewer and subject video section
**Reported:** 2026-04-11 | **Priority:** Medium
Video findings have a `frame_path` (extracted JPEG in `tmp/frames/`) but no way to view them in the UI. The lightbox calls `/api/sources/{id}/image` which returns 404 for video sources. Needs:
- A `/api/findings/{id}/frame` endpoint that serves the full frame JPEG for video findings (falls back to source image for photo findings)
- Lightbox triggered by finding ID, not source ID, so it can show the right image
- A "Videos" section on the subject detail page showing video sources that contain the subject, with frame thumbnails
- Possibly inline video playback or a link to the source video file

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
