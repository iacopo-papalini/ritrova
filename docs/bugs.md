# Bugs

## Open

### BUG-23: /together infinite scroll doesn't load more pages
**Reported:** 2026-04-21 | **Closed:** 2026-04-21
**Repro:** Open `/together`, select 1-2 subjects, scroll past the first page worth of results.
**Observed:** Sentinel spinner stays on screen; no network request to `/api/together-html` is made; no console / server error. Second page never arrives.
**Root cause:** `templates/together.html:27` was swapping the search results into `#results` with `el.innerHTML = html;`. htmx only wires up `hx-*` attributes on nodes it **processes itself** (via `hx-get`, `hx-swap`, or `htmx.process()`). Plain `innerHTML` bypasses htmx, so the infinite-scroll sentinel's `hx-trigger` was never registered — neither `revealed` nor `intersect once` could fire because htmx wasn't watching. Initial diagnosis blamed `revealed` misfiring; the real cause is upstream.
**Fix:** after `el.innerHTML = html;`, call `window.htmx.process(el)` so htmx scans the swapped-in subtree and binds the sentinel's trigger. Also kept the earlier `revealed → intersect once` tweak on `partials/together_results.html:45` as a best-practice hardening (more reliable when the sentinel is already in the viewport at render time), though the primary fix is `htmx.process`.
**Not at risk:** other pages using `hx-trigger="revealed"` (`subject_finding_grid`, `cluster_detail`, `face_grid`, `singleton_grid`, `singletons`) go through htmx-driven routes, so their sentinels are htmx-processed at swap time. No fix needed there.

### BUG-22: "Go to photo" from video-frame lightbox renders the frame JPG with bbox overlays for every finding
**Reported:** 2026-04-20 | **Closed:** 2026-04-21 (short-term fix)
**Fix:** Lightbox store's `detailUrl` getter returns empty when the current item's source `type === 'video'`, which hides the "Open details" button (`x-show="$store.lightbox.detailUrl"` already gated on the URL). Users can no longer land on a misrendered `/photo/{id}` for video sources. Bumped `app.js?v=16` to bust the cached module.
**Proper fix (deferred):** a dedicated video-source page that plays the video with a frame scrubber and draws only the current frame's finding bbox. Track as a new feature if the need arises.

### BUG-5: Thumbnails in data/ dir get indexed by pet scanner
**Reported:** 2026-04-10 | **Closed:** 2026-04-15
**Fix:** `.fr_exclude` marker file. Scanner skips dirs with this file. Created in `data/`.
**Verified:** 2026-04-13 benchmark (`scan-pets` ran cleanly on fresh content, zero bogus entries). Residual orphan rows from the original incident were purged on 2026-04-15 as part of BUG-18 cleanup.

### BUG-18: Orphaned video frame source rows with bad paths
**Reported:** 2026-04-11 | **Closed:** 2026-04-15
**Fix:** Nuke-and-pave cleanup on 2026-04-15. The "rescan pets on videos" path in the original plan was invalid (pet scanner walks photos only), so the residual was purged directly:
- 14,769 orphan source rows deleted (14,687 empty + 78 video-frame rows with legacy findings + 4 library-asset rows from `.venv/`).
- ON DELETE CASCADE removed 14,787 scan rows and 110 findings.
- 95 manual subject assignments on the orphan findings were lost (all dog/cat-species, distributed across "Cani a caso", "Mordicchia", "Figaro", "Strega", "penny", "Gatti a caso", "Caterina - baby" — each affected subject retained the vast majority of its real findings).
- VACUUM reclaimed ~280 MB (562 MB → 281 MB).
- Pre-cleanup DB backup kept at `data/faces.db.pre-nuke-20260415-151033` (and the two pre-benchmark backups from 2026-04-13 as deeper fallbacks).

### BUG-19: Cross-species assignment should confirm and correct, not block
**Reported:** 2026-04-11 | **Closed:** 2026-04-11
**Fix:** API returns 409 with `{error, needs_confirm}` on species/kind mismatch. UI shows `confirm()` dialog. On confirm, re-sends with `force=true`. DB methods accept `correct_species=True` to update the finding's species to match the subject's kind. Applies to cluster assign, single finding assign, and claim-faces endpoints.

### BUG-20: UI silently swallows server errors
**Reported:** 2026-04-11 | **Closed:** 2026-04-13
**Fix:** Introduced a global Alpine `$store.toast` (`app.js`) + reusable `partials/toast.html` (fixed bottom-right, level-coloured, dismissable, supports optional action button for future FEAT-5 undo).
- `window.fetch` is wrapped to auto-toast any non-2xx response and network errors. 409 is excluded — it's reserved for `needs_confirm` flows (BUG-19).
- `htmx:responseError` and `htmx:sendError` handlers replace the prior silent `console.error` and surface via the same store.
- Opt-out via `{ skipErrorToast: true }` on the fetch init for call sites that intentionally handle their own errors.
- Exposed `window.showToast(opts)` for inline handlers and legacy helpers.

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
