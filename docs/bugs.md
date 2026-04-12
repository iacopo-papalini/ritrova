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

### BUG-18: Video frame paths broken for pet scanner; video management needs rethink
**Reported:** 2026-04-11 | **Status:** Open
**Symptom:** Pet-scanned video frames have wrong stored paths (e.g. `face_recog/data/tmp/frames/vid_xxx.jpg__pets` instead of `tmp/frames/vid_xxx.jpg__pets`). The human scanner stores `tmp/frames/...` correctly, but the pet scanner prefixes with the project-relative path, so `resolve_path()` can't find the actual file.
**Scope:** This is part of a broader issue with video management:
- Video frames live in `data/tmp/frames/` as extracted JPEGs — ephemeral but treated as permanent photos
- The `__pets` pseudo-photo suffix is a hack to track pet-scan state per photo
- Human and pet scans of the same video frame create separate photo rows with different path conventions
- No cleanup of orphaned frames when videos are removed
**Next step:** Design a cleaner video/frame model — possibly a dedicated `video_frames` table or a scan-state table, rather than overloading the `photos` table with suffixed paths. Fix the pet scanner's path normalization as an immediate patch.

### BUG-19: Cross-species assignment should confirm and correct, not block
**Reported:** 2026-04-11 | **Status:** Open
**Symptom:** Assigning a cluster to a person subject raises a 500 ValueError when the cluster's faces were detected as "dog" by the pet detector, even though the faces are actually human (misdetection).
**Expected:** The UI should show a confirmation dialog ("These faces were detected as dog — assign to person and correct species?"). On confirm, the assignment proceeds and `faces.species` is corrected to match the subject's kind.
**Scope:** Affects `assign_face_to_subject` and `assign_cluster_to_subject` in db.py. The hard `ValueError` needs to become a check that the API layer can handle gracefully, returning a 409 with details instead of a 500. The UI re-submits with a `force` flag to correct species and assign.

### BUG-20: UI silently swallows server errors
**Reported:** 2026-04-11 | **Status:** Open
**Symptom:** When a server endpoint returns a 500 (or other error), fetch-based actions (face assignment, cluster actions) fail silently — no toast, no message, no visual feedback. The user has no idea the action failed.
**Expected:** All mutation endpoints should surface errors to the user. At minimum: catch non-2xx responses from `fetch()` and show an error toast/banner. HTMX responses should use `htmx:responseError` to display a message.

### BUG-21: Jinja template variables in JS contexts need |tojson, not autoescape
**Reported:** 2026-04-11 | **Closed:** 2026-04-11
**Root cause:** Jinja2's autoescape (HTML escaping) does not protect JavaScript evaluation contexts — Alpine `x-data`, `@click`, inline `<script>`. After HTML-decoding, `'` and `"` break JS string literals. Subject names with apostrophes, emoji, or angle brackets caused Alpine parse failures and are potential XSS vectors.
**Rule:** `|tojson` is safe inside `<script>` tags but NOT inside HTML attributes (`x-data="..."`, `@click="..."`) because its `"` output breaks attribute delimiters. For server-side values needed in Alpine expressions within attributes, use `data-*` attributes (autoescape protects `"` via `&quot;`) and read via `$el.dataset.*` in the Alpine expression.
**Pattern:** `<div data-name="{{ name }}" @click="alert($el.dataset.name)">` — safe. `<div @click="alert({{ name|tojson }})">` — broken.
**Fixed in:** `subjects.html` (names array via `<script>` tag), `cluster_detail.html` (subject name via `data-subject-name`), `subject_detail.html` (merge confirm + unassign confirm via `data-subject-name`), `compare.html` (URL via `data-compare-url`). E2E test added with gnarly name: `Al'ice 🧑<test>`.

### BUG-17: Subject picker needs keyboard navigation
**Reported:** 2026-04-10 | **Closed:** 2026-04-11
**Fix:** Arrow Down/Up navigate, Enter selects, highlighted item scrolls into view. `hi` index in Alpine `subjectPicker` component.

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
