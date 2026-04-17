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

### FEAT-8: Photo scene descriptions and tags via computer vision
**Reported:** 2026-04-10 | **Priority:** Medium
Vision-language model (BLIP-2, LLaVA) for captions/tags ("people at a table", "christmas tree"). Apple Silicon GPU. New DB tables, `ritrova describe` CLI. Enables FEAT-7 content search.

### FEAT-21: Caption in face-hover tooltip
**Reported:** 2026-04-17 | **Priority:** Medium
On hover of a face thumbnail, show the source photo's VLM caption next to the file path — turns the existing tooltip from a path-only anchor into a quick scene preview ("family dinner in the garden, 2018"). Makes the caption data (already stored in `descriptions.caption`) actually useful in the UI without adding navigation. Scope: add `db.get_descriptions_batch(source_ids)` helper (avoid N+1), extend the cluster_detail endpoint to pass a `face_tooltips` dict of `"path\ncaption"`, rename the template binding. Tooltip falls back to path-only when a source has no caption yet. Not shown: tags (kept for the search UI, not the hover). Audit all face-card render sites for consistency.

### FEAT-22: Prune rare tags from the descriptions table
**Reported:** 2026-04-17 | **Priority:** Medium
VLM-generated tags are free-form, so a full-archive re-index produces a long tail of singletons and near-singletons (spelling variants, one-off objects, model hallucinations). A tag that appears <N times cannot contribute to set-based retrieval and just clogs tag autocomplete + UI. Ship as a **separate, manually-invoked command** (`ritrova tags prune --min-count N [--dry-run]`), *not* folded into the duplicate-findings `prune` — running this mid-re-index would delete tags whose count would have crossed the threshold later, and it's destructive without a backup. The separation forces explicit intent: "I know the archive is stable, cleaning up now."
Default **N=10** (deliberately high — truly useful tags like "spiaggia" or "gatto" appear in hundreds of sources on a 32K-source archive, so a cut at 10 removes the long tail without losing meaningful retrieval).
Implementation: (a) count tag frequencies across `descriptions.tags` JSON, (b) strip below-threshold tags from every description row, (c) report tags removed + sources touched. Idempotent after convergence; re-runnable after each full re-index. Warns if any scans are still in progress.

### FEAT-24: LLM-proposed tag normalization with per-rule review
**Reported:** 2026-04-17 | **Priority:** Medium
Simple pruning throws information away — `micio(1)` just vanishes even though it points at the same concept as `gatto(243)`. Richer approach: feed the tag vocabulary + counts to a text LLM, get a mapping of synonyms / diminutives / plural variants to their canonical form (`micio, gattino → gatto`; `mare` and `spiaggia` stay distinct), and apply the merges so the photos keep their retrieval value under the canonical name.

**Workflow mirrors subject-cluster merge (FEAT-5 / subject-merge):**
1. `ritrova tags plan` generates proposals via the LLM and stores them in a new `tag_merge_plans` + `tag_merge_rules` table. Rules carry `(id, plan_id, from_tag, to_tag, status, applied_at, undone_at)`. No writes to `descriptions.tags` yet.
2. Web UI surfaces pending plans: list of rules, counts affected, a per-rule **apply** / **reject** action and a per-rule **undo** once applied. Same pattern the subject-cluster UI uses today.
3. Apply rewrites `descriptions.tags` atomically for that rule (replacing `from_tag` with `to_tag` everywhere, deduping the per-source tag set if both were present). Undo restores `from_tag` on every source touched — sources are recorded on the rule row.
4. CLI peers for headless use: `ritrova tags apply <rule-id>` / `ritrova tags undo <rule-id>` / `ritrova tags plan-show <plan-id>`.

**Safety rails:**
- LLM prompt rules: only merge when the tags refer to the **same concept**; never merge into a hypernym (labrador ≠ cane); canonical = most frequent member.
- Dry plan by default — nothing mutates the description table until the user explicitly applies a rule.
- Plan stays in the DB so sessions can resume; user doesn't have to accept everything in one sitting.
- Every rule undoable individually (FEAT-5 infrastructure reused).

**LLM choice deferred** — dedicated session to pick between a small text model on MLX (Qwen2.5-3B / Llama 3.2), embedding-based clustering, or hybrid. Record that decision in ADR-011 when it happens.

**Relationship to FEAT-22:** FEAT-22 (simple threshold prune) stays as a composable mop-up: normalise first (keep signal), prune the residual long tail second.

### FEAT-25: Kill-list for uninformative tags
**Reported:** 2026-04-17 | **Priority:** Medium
Common-but-useless tags the VLM keeps producing — "maglione", "pantaloni", "oggetto", "superficie", "sfondo". They're frequent enough to survive FEAT-22's threshold prune, and not synonyms of anything so FEAT-24's merge planner won't touch them. Only the user can decide "this tag adds no retrieval value in *my* archive." Ship a **persistent blocklist** so the call is made once and stays made.

**Schema:** new `tag_blocklist(tag TEXT PRIMARY KEY, killed_at TEXT, killed_by TEXT)`. Trivial table; FK nothing.

**Write path:** `_parse_vlm_response` (or CaptionStep.analyse right after) strips any tag present in the blocklist before the result is saved. The next re-index therefore self-heals — killed tags never come back without an explicit un-kill.

**User surface:**
- Web UI — anywhere a tag is rendered (tag chips, search autocomplete, photo detail), a compact ⨉ button on the tag fires `POST /api/tags/{tag}/kill`. Toast with Undo (FEAT-5 integration) holds a `KillTagPayload` that restores the tag to every source that had it and removes the blocklist row.
- CLI peers: `ritrova tags kill <tag>` / `ritrova tags unkill <tag>` / `ritrova tags blocklist` (list). Non-interactive; dry-run supported.

**What "kill" does:**
1. Insert into `tag_blocklist`.
2. Strip `tag` from every `descriptions.tags` JSON set it appears in (batch update).
3. Record the source IDs touched on the undo payload so restore is exact.

**Safety:** killing is single-tag and reversible, so no plan/apply dance needed (unlike FEAT-24). The user's decision is local to one tag at a time.

**Composability:** the three tag-cleanup tools stack naturally — normalize first (FEAT-24: keep signal), kill second (FEAT-25: remove explicitly-useless concepts), prune last (FEAT-22: drop the residual long tail).

### FEAT-23: Periodic rare-tag trim during analyse (mid-loop)
**Reported:** 2026-04-17 | **Priority:** Low
Companion to FEAT-22 for the case where we re-enable VLM vocab hints (see `describer.DEFAULT_USER_PROMPT_WITH_VOCAB` / `CaptionStep(vocab_hint=...)`, currently unused in the new `analyse` pipeline). With vocab hints on, every captioned source contributes its tags to the prompt fed to later sources — and singleton junk from the first 10% of the archive pollutes the prompt for the remaining 90%. Fix: every N sources, recount frequencies and strip tags whose running count is clearly noise, using a **scaled threshold** (e.g. `min_count = max(2, 0.002 * processed_count)`) so the filter is lax early and stricter later. Keep a separate `--no-vocab-hint` escape hatch. Out of scope until FEAT-7 (search UI) shows that vocab-hint-driven consistency actually improves retrieval quality enough to re-enable the feature.

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
