# Bug Tracker

## Open

### BUG-1: Cluster hint doesn't disappear on mouse leave
**Page:** /clusters
**Reported:** 2026-04-10
**Status:** Fixed
**Fix:** Added Alpine `x-data` with `showHint` toggle on mouseenter/mouseleave; htmx loads hint content once, Alpine controls visibility.

### BUG-2: Lightbox must show photo full path
**Page:** All pages with lightbox
**Reported:** 2026-04-10
**Status:** Fixed
**Fix:** Added `photoPath` to Alpine lightbox store, fetches `/api/photos/{id}/info` on open. Path displayed below image.

### BUG-3: "Create new pet" doesn't work
**Page:** /clusters/{id} (pet clusters)
**Reported:** 2026-04-10
**Status:** Fixed
**Fix:** Fixed `createNew` init (was `false` even when only `__new__` option existed). Simplified form to build FormData manually.

### BUG-4: No way to remove a single face from a cluster
**Page:** /clusters/{id}
**Reported:** 2026-04-10
**Status:** Fixed
**Fix:** Added per-face "x" button (visible on hover) that calls `/api/faces/exclude` and removes the element from DOM.

### BUG-5: Thumbnails in data/ dir get indexed by pet scanner
**Page:** N/A (scanner)
**Reported:** 2026-04-10
**Status:** Fixed
**Fix:** Implemented `.fr_exclude` marker file. Scanner skips any directory containing this file. Created marker in `data/`. Cleaned 14,237 bogus DB entries.

### BUG-6: After assigning a cluster, page doesn't navigate
**Page:** /clusters/{id}
**Reported:** 2026-04-10
**Status:** Open
**Description:** Click Assign on a cluster detail page. The assignment succeeds (DB updated) but the browser stays on the same page instead of navigating to the next similar cluster.
**Root cause:** `fetch()` follows the 303 redirect silently; `r.redirected` + `location.href = r.url` not triggering navigation.

### BUG-7: Nav shows "Humans" on pet cluster detail pages
**Page:** /clusters/{id} (pet clusters)
**Reported:** 2026-04-10
**Status:** Open
**Description:** The nav pill toggle highlights "Humans" even when viewing a pet cluster. The `species` variable is not passed to the cluster_detail template context.

## Closed

(none yet -- bugs move here once verified fixed in production)
