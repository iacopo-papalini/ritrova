# Subject-detail page redesign

Author: UI Designer pass · 2026-04-21 · target templates `subject_detail.html`, `partials/subject_face_tile.html`, `partials/subject_finding_grid.html`.

## TL;DR

Collapse the **two-and-a-half-row control strip** (rename form · Show-similar · Merge-into · Delete · Circles · tabs) into a **single header**: inline-editable title with a hover pencil (reuse the `circles.html` pattern shipped in `158fac0`) · a tight action cluster on the right (`Show similar` primary, `Merge…` in overflow, `Delete` in overflow, destructive), and a discreet second line of circle chips directly under the title. The tab pill moves into the same row as the subject count. That reclaims ~180 px of vertical real estate above the fold, removes three competing input patterns, and pushes the face grid — the actual content — onto the first screen. The **same shape** (`partials/entity_header.html`) then serves cluster-detail, circle-detail and subject-detail, executing recommendation #4 of `docs/ui-assessment.md`.

## Evidence

Screenshots captured at 1440×900 against a copy of the live DB; saved under `/Users/iacopo.papalini/nextCloud/Photos/face_recog/docs/subject-detail-redesign/`.

1. **Three input grammars in one row** — `current-agnese-headercrop.png` and `current-agnese-viewport.png` show the action row: a bare label-less rename input with a sibling `Rename` button (`subject_detail.html:28-39`), an amber CTA link (`:42-48`), a label-above typeahead picker `Merge into…` with an explicit caption (`:51-69`), a flex spacer, and a red-outline `Delete person` button (`:74-86`). Four different visual weights, two different label-placement conventions, no clear primary/destructive hierarchy. "Delete person" has the same optical weight as a benign tab pill two rows down.
2. **Circles row is a second, half-duplicated action bar** (`subject_detail.html:91-122`) — a grey uppercase `CIRCLES:` caption, chips with inline ×, and yet another typeahead (`circle_picker`). The only differentiator from the merge picker visually is width and the orange chip fill. The user reads two rows of inputs before seeing a face.
3. **Tabs buried below two rows of controls** — `current-agnese-viewport.png`: the Face Samples / Photos / Videos pill group sits at ~y=300 on a 900-tall viewport. On the sparse case `current-nazareno-viewport.png` it's still at y=260. The face grid never starts above the fold.
4. **Selection bar is tall because of the picker input** — `current-agnese-selection-bar.png` shows `2/3 selected` · wide gap · `Type to search…` · red `Remove from Agnese` · `Cancel`. The bar is `py-3` (`subject_detail.html:175-177`) with a full-height input, while `singletons.html:42-43` uses the same `py-3` but the picker is the same widget. Net: both bars are tall, but subject-detail adds a destructive danger button *and* a bare Cancel on the same line — 5 elements of varying heights create the ragged look. Compared to the cluster selection bar (`cluster_detail.html:98-137`) which has 4 buttons of matching height, the subject bar is the outlier because it mixes a typeahead with actions.
5. **Sparse months waste full rows** — `current-nazareno-fullpage.png` and `current-agnese-viewport.png` show `2026-04` (1 face) taking a grid row to itself, same as `2025-06` (4 faces), same as `2025-04` (10+ faces). Each group starts with `<h3 class="mt-4 mb-2 …">` (`subject_detail.html:221`, `partials/subject_finding_grid.html:3`). On Nazareno (11 faces) the page is ~40% empty whitespace *because* of month breaks.
6. **"Unknown" bucket is technically "no YYYY-MM in path"** — `app/helpers.py:53-59` returns the string `"Unknown"` when the regex `\d{4}-\d{2}` matches nothing in any path segment. In `current-agnese-selection-bar.png` the top group labelled `Unknown` (4 faces) are real findings from sources whose directory tree contains no date token. Labelling that as "Unknown" reads like a bug.
7. **Three different entity-detail headers** — `current-cluster-detail-header.png` (Cluster #392334, 160 faces: one row, typeahead + two buttons), `current-agnese-headercrop.png` (Subject: three rows, six control types), `current-circle-detail-header.png` (Circle Famiglia, 14 members: title only, no actions visible). All three are the same conceptual shape ("entity + list + operations"); nothing about their current visual grammar says that.

## Proposed layout

Top-to-bottom sections at 1440w, gallery-dark theme unchanged:

### 1. **Header block** (one row, plus a one-line meta row)

**Row A** — inline-edit title · count · action cluster:

- `<` back arrow → `/{kind}` (unchanged).
- **Subject name** as an `<h1>` with the `circles.html` hover-pencil pattern: a pencil icon appears on hover, click swaps the span for an input pre-selected; Enter saves via `/api/subjects/{id}/rename` returning JSON, Esc cancels. **Delete** the separate rename form + button. This is the canonical rename pattern across the app; aligns with `circles.html`, `circle_detail.html` (double-click) and resolves the pattern drift called out in `docs/ui-assessment.md` §"Revisions" #4.
- `(609 faces in 507 photos)` muted meta, unchanged.
- **Flex spacer.**
- **Action cluster**, right-aligned, left-to-right priority:
  - `Show similar unclustered` — **primary filled-amber** button (unchanged, but now the only primary). Shown only when `has_unclustered` is true.
  - `Merge into…` — **secondary ghost** button labelled `Merge…`, opens a popover containing the `subject_picker` (same typeahead, same behaviour as today but no longer occupying a wide inline slot). The label moves from above the input to the button face.
  - **Overflow `⋯`** menu button (secondary ghost). Two items:
    - `Find similar across unclustered` (duplicate of the primary when needed)
    - **Delete person/pet** — flagged red, opens confirm dialog. Destructive adjacency risk (`ui-assessment` H11) is solved by moving Delete out of the visible action row entirely.

Primary / secondary / destructive are now **one** / **outline** / **inside an overflow with a red label** — not three competing input shapes in one strip.

**Row B** — meta strip, single line, left-aligned, `text-xs text-gallery-muted`:

- `CIRCLES:` caption (kept for scanning) + circle chips (unchanged component) + a single `+ Add to circle…` pill that expands into the typeahead on click (`circle_picker` as popover, not inline input). This keeps all circle affordances but at chip-weight, not input-weight. When there are zero circles, show only `+ Add to circle`.

### 2. **Tabs + count sub-header** (one row, same line as tab)

- Tabs pill group unchanged (`Face Samples (n)` · `Photos (n)` · `Videos (n)` gated as today).
- Right-aligned on the same line: a small **density toggle** for the Face Samples tab — `Dense / Group by month` (default: Dense). Default off fixes issue 5 without taking the feature away from the user who prefers chronology. Empty for the Photos/Videos tabs.

### 3. **Selection bar** — compact, amber, when faces selected

Match `singletons.html:41-92` density exactly: `py-2` (not `py-3`), same horizontal spacing, same button sizing.

- `N selected` (amber).
- Flex spacer.
- **Move to…** — **primary filled-amber** button that opens a popover with `subject_picker` (same swap endpoint already wired). Popover, not inline input, is what kills the tall-bar problem — the input stops competing vertically with the buttons.
- **Unassign from {subject}** — secondary ghost (renamed from `Remove from…` to match the verb used elsewhere).
- **Dismiss as non-face** — red ghost (adds the third action of the canonical per-face menu called out in `ui-assessment.md` §"Missing affordance… Recommended pattern"; today subject-detail only has unassign, which forces 4-click round-trips for "oh this is a mis-detected shape not a face").
- `Cancel` — muted link button.

All four actions are button-sized (`py-1.5`); the input lives inside a popover that opens below. Bar height drops by ~12 px and stops looking ragged.

### 4. **Face grid** — dense-first, month-grouped-optional

Default view (Dense): one continuous grid, no month breaks. The grid stays at `grid-cols-14` on `lg` as today.

When the user flips the density toggle to **Group by month**, render month labels **inline** — a small `sticky left` pill (`2025-06`) that occupies the **first grid cell** of its row rather than a whole `<h3>` row. Implementation hint for the developer: insert a `<span class="col-span-1 aspect-square flex items-center justify-center text-xs text-gallery-muted">{{ group.month }}</span>` as the first child of each group's contiguous run, then the face tiles flow around it. A 1-face month now costs one tile-width, not one full row.

Relabel the "Unknown" bucket to `No date in path` (or, better, suppress it entirely when it is the *only* bucket and the user is in Dense mode). Both copies are more honest than the bare word "Unknown" which reads as a data quality bug.

Interaction pattern rationale: the user's primary workflow on this page is **verifying a named subject's face samples** (scrub, remove outliers, move mis-assigned faces to the right person). Chronology is secondary — useful when something looks off and you want to anchor it to a year — so it becomes an opt-in not a default.

### 5. **Per-face action menu** (not in scope for this redesign but noted)

`ui-assessment.md` §"Missing affordance" promotes the canonical `Move / Unassign / Dismiss` menu. Subject-detail is the main beneficiary. Delivering that is a separate follow-up (it affects `photo.html` and the partial), but the selection-bar changes above anticipate it so the two land cleanly.

## Component spec for handoff

New:
- `partials/entity_header.html` — the unified block-A + block-B above, taking slots: `title`, `count`, `actions_primary`, `actions_overflow` (list), `meta_chips` (optional, list). Used by cluster / subject / circle.
- `partials/inline_rename_title.html` — the hover-pencil + inline-edit `<h1>` Alpine component, extracted from `circles.html`. Drops into any entity header.
- `partials/popover_subject_picker.html` — the `subject_picker` already exists; wrap it in a button-opens-popover Alpine shell so `Merge into…` and selection-bar `Move to…` can share one component.

Modified:
- `subject_detail.html` — rewritten against the spec above; removes the inline rename form, the flat action row, the separate circles row; adopts `entity_header.html`. Adds the density toggle on the Face Samples tab.
- `partials/subject_finding_grid.html` — supports a `dense` mode (no month headers) and an `inline-month-label` mode (month occupies one cell, not a row). Both paths used by the initial render and the htmx infinite-scroll fragment, so the toggle state has to be forwarded as a query param.
- `cluster_detail.html`, `circle_detail.html` — migrated to `entity_header.html`. Cluster keeps its typeahead as the primary action inside block-A; circle-detail inherits the hover-pencil rename (replaces the current double-click-only affordance, keeping double-click as a keyboard-power-user shortcut).

Deleted:
- The label-above-input pattern used for `Merge into…` and `Add to circle…` — gone everywhere, replaced by popover buttons.

No new CSS. Everything uses existing Tailwind + `gallery-*` tokens.

## Unification claim

**Cluster detail** — header becomes: `<` back · `Cluster #183275` (no hover-pencil; cluster IDs are not renamable) · `(160 faces)` count · primary `Assign to…` popover (today's inline typeahead, relocated) · secondary `I don't know this person` · overflow with `Dismiss entire cluster` (demoted from the current nuclear-button-in-the-primary-row placement, resolving `ui-assessment.md` H11). Suggestions line moves into block-B meta as muted pills. The current typeahead + two buttons already fit this shape; the visual change is entirely "shrink and regroup".

**Circle detail** — header becomes: `<` back · inline-rename `Famiglia` title · `(14 members)` · primary `Add member…` popover (today there is no such affordance on circle-detail; it has to be added to fulfil the promise that a circle detail is a working surface, not a read-only list — `ui-assessment.md` H8). Overflow holds `Delete circle` and a future `Rename description`. The page body underneath remains the member list, with the H8 follow-up (a `/together?subjects=…` preview block) slotting beneath when that work lands.

All three pages now answer the same first question in the same place: *"what is this thing, how many findings does it contain, and what are my top three actions on it?"*

## Open decisions

1. **Rename pattern on subject-detail**: hover-pencil (matches `circles.html`) or double-click-title (matches `circle_detail.html`)? Proposal above picks hover-pencil because it's the newer shipped pattern (commit `158fac0`), is more discoverable, and reduces the chance of accidental double-click-selects. Flag: do we want to retrofit `circle_detail.html` to match, dropping its double-click? Probably yes, but confirm.
2. **Default density** of the Face Samples grid: Dense vs Group-by-month. Proposal picks Dense; the user may prefer chronology by default for a subject-curation workflow where "is this the right year?" is part of verification. If so, flip the default and keep the toggle.
3. **"Unknown" copy**: `No date in path` is more honest. If you prefer shorter, `Undated`. Confirm which wording.
4. **Merge / Delete as overflow vs visible buttons**: hiding Delete behind `⋯` solves adjacency but adds a click. For a once-per-session destructive operation that feels correct. Confirm you're OK with the extra tap on delete.
5. **Tab placement**: spec puts tabs + density toggle on the same row. If the Videos tab is present for most subjects, the row gets tight at 1440w — we may need to push density toggle to the right edge of the grid header instead. Confirm after first mock.
6. **Per-face action menu** (Move / Unassign / Dismiss) — out of scope here, but the selection-bar proposal assumes it will land. If the per-face menu gets deprioritised, the selection bar is still an improvement on its own.

---

### Screenshots referenced

- `/Users/iacopo.papalini/nextCloud/Photos/face_recog/docs/subject-detail-redesign/current-agnese-viewport.png`
- `/Users/iacopo.papalini/nextCloud/Photos/face_recog/docs/subject-detail-redesign/current-agnese-fullpage.png`
- `/Users/iacopo.papalini/nextCloud/Photos/face_recog/docs/subject-detail-redesign/current-agnese-headercrop.png`
- `/Users/iacopo.papalini/nextCloud/Photos/face_recog/docs/subject-detail-redesign/current-agnese-selection-bar.png`
- `/Users/iacopo.papalini/nextCloud/Photos/face_recog/docs/subject-detail-redesign/current-agnese-scrollbottom.png`
- `/Users/iacopo.papalini/nextCloud/Photos/face_recog/docs/subject-detail-redesign/current-caterina-viewport.png`
- `/Users/iacopo.papalini/nextCloud/Photos/face_recog/docs/subject-detail-redesign/current-caterina-scrollbottom.png`
- `/Users/iacopo.papalini/nextCloud/Photos/face_recog/docs/subject-detail-redesign/current-nazareno-viewport.png`
- `/Users/iacopo.papalini/nextCloud/Photos/face_recog/docs/subject-detail-redesign/current-nazareno-fullpage.png`
- `/Users/iacopo.papalini/nextCloud/Photos/face_recog/docs/subject-detail-redesign/current-cluster-detail-header.png`
- `/Users/iacopo.papalini/nextCloud/Photos/face_recog/docs/subject-detail-redesign/current-circle-detail-header.png`
