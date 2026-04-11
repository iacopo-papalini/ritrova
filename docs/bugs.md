# Bugs

## Open

### BUG-5: Thumbnails in data/ dir get indexed by pet scanner
**Reported:** 2026-04-10 | **Status:** Fix applied, pending verification
**Fix:** `.fr_exclude` marker file. Scanner skips dirs with this file. Created in `data/`. Cleaned 14K bogus entries.
**To close:** Run a new pet scan and verify exclusion.

### BUG-6: After assigning a pet cluster, page doesn't navigate
**Reported:** 2026-04-10 | **Status:** Fix applied, pending verification
**Root cause:** `find_similar_cluster()` was not species-aware.
**Fix:** Added `species` parameter, passed from `_next_similar_cluster()`.

### BUG-17: Person picker needs keyboard navigation
**Reported:** 2026-04-10 | **Status:** Open
Arrow keys to navigate dropdown, Enter to select. Essential for high-volume curation.

## Closed

### BUG-1: Cluster hint doesn't disappear on mouse leave
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** Alpine `showHint` toggle on mouseenter/mouseleave.

### BUG-2: Lightbox must show photo full path
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** `photoPath` in lightbox store, fetched from `/api/photos/{id}/info`.

### BUG-3: "Create new pet" doesn't work
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** Fixed `createNew` init, simplified FormData.

### BUG-4: No way to remove a single face from a cluster
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** Per-face "x" button on hover.

### BUG-7: Nav shows "Humans" on pet pages
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** `species` in context, server-rendered nav pills.

### BUG-8: "Merge clusters" button returns 422
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** `Body(...)` → `Form(...)`, `hx-vals` → hidden form inputs.

### BUG-9: No merge suggestions for pets
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** Nav links carry species. Page already accepted species param.

### BUG-10: No compare for pets
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** Compare accepts species, filters by `get_persons_by_species()`.

### BUG-11: Infinite scroll pushes Photos off-screen
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** Alpine tabs (Face Samples / Photos).

### BUG-12: Video frame photos return 404
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** `resolve_path()` routes `tmp/` paths to `db_path.parent`.

### BUG-13: Month grouping uses unreliable EXIF dates
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** Extract YYYY-MM from directory path, not EXIF.

### BUG-14: Pet cluster assign shows 0 face count
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** Fetch actual counts from `get_persons_by_species()`.

### BUG-15: Broken images cause 500 stacktraces
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** `LOAD_TRUNCATED_IMAGES`, always resize through PIL.

### BUG-16: Person card inconsistent between directory and search
**Reported:** 2026-04-10 | **Closed:** 2026-04-10
**Fix:** Shared `person_card.html` partial with avatar.
