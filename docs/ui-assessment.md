# Ritrova UI Assessment

## TL;DR

**Targeted fixes, not a rehaul.** The visual language, the component grammar (cards, pills, face grids, selection bar, subject picker, lightbox), and the interaction rules in `docs/design-guide.md` are all coherent and well-executed. What is broken is **orientation and flow**: the dashboard is a stats dump, the nav is 10 elements wide for a workflow that uses two of them, merge is mis-cast as a destination, and the cluster-detail screen (where the user actually lives) has no queue context. Four to six focused changes would make this app feel like it was designed as one thing instead of accreted.

The single biggest win: **turn the Dashboard into a "next cluster to review" landing page and let the top nav shrink to Clusters / Names / Circles / Search.** That alone fixes H1, H3, and H5 at once.

## What I observed

### Consistent, good
- **Component grammar is solid.** Face grid (`cluster_detail.html:143`, `singletons.html:80`, `find_similar.html:90`, `compare.html:93`) uses the same classes, same selection-bar pattern, same hover-x delete affordance, same ring-offset selected state. `subject_picker` is reused everywhere assignment happens.
- **Toasts + undo + dialog** are unified (`base.html:126`, `partials/dialog.html`, FEAT-5). The rule "every write is undoable in 15s" is respected end-to-end.
- **Typography and chrome are one thing.** DM Sans + `gallery-*` tokens + sticky translucent header = all screens feel like the same app visually.
- **Empty states** on `clusters.html:18`, `subjects.html:59`, `circle_detail.html:39` actually guide — they're written, not stock.

### Anti-patterns

1. **Dashboard is a stat sheet, not a review tool.** `index.html` renders seven animated counters then two buttons. For the stated primary workflow (crunch through 5,295 unnamed clusters), this is a dead-end page — violates design principle #1 "no dead-end stats pages". The 64K → 31K → 5.3K → 19K pipeline should be a **funnel**, not four disconnected tiles.
2. **Top nav has 10 elements for a 2-element workflow.** `base.html:29-99` renders: logo, Humans pill, Pets pill, Clusters, Persons, Merge, Compare, Together, Circles, search, stats badge. The user spends 95% of their time on Clusters + a chosen named subject. Humans/Pets gets dominant visual weight (orange fill) though the user "works almost exclusively with humans" and the dashboard already hardcodes `species="human"` (`app.py:127`) — confirming H2.
3. **Merge is nav-level, but it's a review mode.** `/people/merge-suggestions` duplicates the cluster-review pattern with a different layout. Two named / one named / neither-named branches in `partials/merge_row.html:33-64` duplicate cluster-detail's assignment logic. Confirms H5.
4. **No queue context on cluster detail.** `cluster_detail.html` header is "Cluster #12345 (14 faces)" with a back arrow. There is no "3 of 5,295", no Skip/Next, no Previous. After assigning, the user gets redirected to `/{kind}/clusters` (the 4,450-card firehose). Design guide §"Progress: show the backlog" is not honored on the single most-used screen.
5. **Clusters list dumps 4,450 cards in one HTML response** (`clusters.html:21` has no pagination; verified by curling — 165K lines of HTML, 4,450 `hx-get` hover-hint requests wired up). Every card registers a `mouseenter` htmx trigger. On a laptop this is fine; it is also a poor primary-task surface (no sort, no "start here").
6. **Three "entity + face grid + actions" screens, three layouts.** `cluster_detail.html` has back-arrow-title + assign row + selection bar. `subject_detail.html` has back-arrow-title + rename/find-similar/merge/delete row + circles chips row + tabs + face grid. `circle_detail.html` has back-arrow-title + double-click-rename + member list (no faces). Same conceptual entity, three different headers, different action placements. Confirms H4.
7. **`location.reload()` is still the feedback for 12 actions** across `photo.html`, `circles.html`, `compare.html`, `cluster_detail.html`, `singletons.html`, `subject_detail.html` — a direct design-guide violation. `photo.html:100,118,132` is the worst offender (reload after every single face assign, even though the DOM patch is trivial).
8. **`/search` is a dead page.** The nav bar search submits to it and the page renders a form + the same grid as `/people`. `subjects.html` already has live client-side filter (`subjects.html:48`), making `/search` redundant and worse (name-only, not scoped to kind). FEAT-7 backlog acknowledges this.
9. **"Stranger" is a side concept in the primary flow.** `cluster_detail.html:68` has a "I don't know this person" button but it creates an auto-numbered Stranger subject. `subjects.html:29` has a "hide strangers" pill. This is the user's classification primitive ("named / stranger / unsure") but it's expressed through three different affordances in three different corners.
10. **Inline assign on `photo.html` is the only place dropdown-style assignment still feels off** — the card-level picker is fine, but the full-page reload after each assign makes bulk photo-level correction painful.

## Recommendation

**Targeted fixes.** Do not rehaul. The design system is sound, FEAT-5 undo is universal, the design guide is authoritative and close-to-obeyed. What's needed is (a) reframing the landing page around the actual workflow, (b) demoting secondary nav items, (c) giving the cluster-review screen the queue context the guide promised, and (d) unifying the "entity detail" layout across Cluster / Subject / Circle.

## Top 6 changes (prioritized by impact / effort)

### 1. Rebuild `index.html` as a review-focused landing page — **S / huge impact**
Replace the seven stat tiles with a single vertical funnel and one big primary action.
- **Primary**: "Continue where you left off — next unnamed cluster" → deep-link straight to a cluster detail with queue context (`?queue=unnamed`).
- Secondary cards for the three work streams: "Name clusters (5,295 left)", "Review merges (40 suggestions above 50%)", "Tidy singletons (19K)". Each card = one click into a seeded queue.
- Keep stats as a collapsed footer ("Archive: 31,681 photos · 64,331 faces · 152 named").
- Files: `src/ritrova/templates/index.html`, `src/ritrova/app.py:126` (pass next-cluster-id and counts).

### 2. Add queue context + Skip/Next to `cluster_detail.html` — **S / huge impact**
On the single most-used screen the user has no sense of progress.
- Header: "Cluster 3 of 5,295 unnamed · 14 faces". Add **Previous / Skip / Next** buttons and `?queue=unnamed&pos=3` URL state.
- After `mark-stranger` / `dismiss` / `assign`, redirect to the next cluster in the queue, not `/people/clusters`.
- Bind `s` → skip, `a` → assign top suggestion, per the guide's keyboard section (already spec'd, not implemented).
- Files: `src/ritrova/templates/cluster_detail.html`, `src/ritrova/app.py:134-160`, add a small `db.get_unnamed_cluster_queue(species, after_id)` helper.

### 3. Shrink the top nav; demote Humans/Pets — **S / medium impact**
Current: logo + Humans pill + Pets pill + 6 links + search + stats = 10 items. Target: logo + Clusters + Names + Circles + Search. Move Compare, Together, Merge-suggestions, Singletons into a "More" overflow menu (or expose them contextually on subject_detail / clusters pages). Demote Humans/Pets from a coloured pill to a **one-line tab inside the Clusters screen and the Names screen** only — it's a scope filter, not a top-level section. Confirms H2, H5.
- Files: `src/ritrova/templates/base.html:29-99`, `src/ritrova/templates/clusters.html` (+ tab), `src/ritrova/templates/subjects.html` (+ tab).

### 4. Unify the "entity detail" layout across Cluster / Subject / Circle — **M / high impact**
Introduce a shared `partials/entity_header.html` that renders: back-breadcrumb, entity title, count badge, action cluster (primary · secondary · danger · overflow). All three detail pages call it with different action sets. Same for the face-grid: a single `partials/entity_face_grid.html` with a `mode` prop (`cluster` vs `subject`) driving the hover-x action. This kills the drift H4 identified and halves the surface area for future changes.
- Files: `src/ritrova/templates/partials/entity_header.html` (new), `cluster_detail.html`, `subject_detail.html`, `circle_detail.html`.

### 5. Fix the `location.reload()` violations — **S / medium impact**
Twelve call sites, all mechanical. The guide already prescribes the pattern (DOM remove + toast). `photo.html` assigns should patch the face tile in place (replace the typeahead with a pill showing the assigned name). Circles / compare / cluster exclude all already have the structure for a surgical update.
- Files: `photo.html:100,118,132`, `circles.html:81`, `compare.html:85,173`, `cluster_detail.html:112,125`, `singletons.html:13,66`, `subject_detail.html:64,117`.

### 6. Fold `/search` into the nav bar; delete the standalone page — **S / low impact**
The nav input already searches; the `/search` page is redundant and worse than the live filter on `/people`. Make the nav input a typeahead (reuse `subject_picker`) that drops to either `/people/{id}` on select or `/people?filter=<q>` on Enter. Remove `search.html` and the `/search` route (or leave a legacy redirect). Frees the URL for FEAT-7's richer syntax later.
- Files: `src/ritrova/templates/base.html:86-92`, `src/ritrova/app.py:316-331` (remove), `src/ritrova/templates/search.html` (delete).

## Open questions for the user

1. **Queue definition.** For "next unnamed cluster", should we order by face count desc (biggest first), by most recent cluster id, or by density of named-subject similarity hints? (My default: face count desc — biggest wins first, best feeling of progress.)
2. **Stranger as first-class concept.** Would you want "I don't know this person" to become one of the primary actions on cluster detail (same visual weight as Assign), alongside a "Skip for later" that doesn't create a Stranger subject? Today "skip" implicitly = "leave unnamed, navigate away".
3. **Humans/Pets demotion.** OK to remove the top-level pill and use a small `?kind=pets` tab inside the cluster / names screens? Or keep a minimal pill for discoverability in case pets occasionally show up in a batch?
4. **Compare + Together + Singletons** — promote the two you actually use in steady state, hide the others in a More menu? (My guess: Together is high-value for browsing, Compare is rare, Singletons is high-volume during setup but sporadic afterwards.)
5. **Photo detail reload** — is it OK to change the post-assign behavior to a DOM patch (face tile gets the person pill, no scroll loss), matching the rest of the app?

---

## Visual pass (screenshots)

Rendered at 1440×900 from a copy of the live DB (`faces_ui_review.db`) via Playwright + headless Chromium. All PNGs are under `./ui-assessment/`.

### 01 · Dashboard
![](./ui-assessment/01-dashboard.png)

- **Prior claim**: "stat sheet, 7 tiles, no queue entry point".
- **Verdict**: **Confirmed and worse than described.** 7 tiles arranged 4+3 with a visibly empty 4th slot on the second row — the grid looks unfinished, not deliberate. The two CTAs at the bottom ("Review unnamed clusters", "Browse names") are underweight for the amount of pipeline above them.
- **New visual observations**:
  - Cognitive-load problem: **the numbers are semantically related in a pipeline (31,681 photos → 64,331 faces → 31,428 identified · 18,996 unclustered · 5,295 unnamed clusters · 368 dismissed)** but the tiles are laid out as a flat grid with no arrows, no percentages, no relationship. The user has to do mental arithmetic to see the funnel.
  - "Persons named: 152" is the only small-cardinality number and it gets the same visual weight as "Faces detected: 64,331". Should be right-sized.
  - Massive bottom whitespace — on a 900-tall viewport, 60% of the page is empty. The page doesn't even try to occupy the fold.
  - Corrects prior-report detail: nav shows **8 elements**, not 10 (Ritrova / Humans / Pets / Clusters / Persons / Merge / Compare / Together / Circles + search). No "Singletons", no "stats badge" in nav — the report was reading template conditionals, not the rendered output. The conclusion (nav is too wide for the workflow) still stands, just recount the badges.

### 02 · Clusters list (viewport + full-page)
![](./ui-assessment/02-clusters-list.png)
![](./ui-assessment/02b-clusters-list-full.png) — full-page, renders to **~109,000 pixels tall**.

- **Prior claim**: "4,450 cards in one response, no pagination".
- **Verdict**: **Massively confirmed**. The full-page screenshot is a visceral demonstration — it's a single HTML response that renders as a 100K+ pixel strip. No header, no sort, no filter visible once you scroll past the first viewport. Header just reads "Unnamed Clusters (4450)" with a tiny "View unmatched singles" link top-right.
- **New visual observations**:
  - **Each card is a 2×2 mosaic of face thumbnails**, which is actually a really nice affordance — at a glance you see whether a cluster has face-variety (likely a bad cluster) or is a single person (worth naming). Don't lose this in any redesign.
  - No "start here" handle on the page. The user is dropped into a wall of 4,450 equally-weighted options and has to pick one.
  - **Grid visually works** at 1440w — 8 columns, clean gutters, consistent card height. The problem is strictly pagination / prioritization, not grid design.
  - Cluster IDs are rendered as `#183275 · 14 faces` in tiny low-contrast grey under each mosaic — technically legible, visually irrelevant to the task.

### 03 · Cluster detail (big unnamed cluster — 105 faces)
![](./ui-assessment/03-cluster-detail.png)
Header-only crop: ![](./ui-assessment/17-cluster-header-crop.png)

- **Prior claim**: "header is 'Cluster #12345 (14 faces)', no X-of-Y progress, no Skip/Next".
- **Verdict**: **Confirmed exactly.** Header reads `< Cluster #183275 (105 faces)`. There is **no queue context, no Previous/Skip/Next, no similar-cluster sidebar, no "assign + go to next" action**. Actions are: "Type to search" (typeahead), "I don't know this person" (outline grey), "Dismiss entire cluster" (red).
- **New visual observations** (only visible rendered):
  - **"Suggested:" inline row** shows three low-confidence hints: `Amica di Teresa #2 20.1%, Vincenzo Carlucci 12.4%, Sconosciuto #1 11.1%`. These numbers are all under 25% — if the best guess is 20% the row is essentially decorative noise. This wasn't obvious from template reading.
  - With 105 face tiles the grid runs to ~13 rows tall — no virtualization, no "show first 30 / see all". User has to scroll 2.5 viewports just to see the whole cluster they're naming.
  - The typeahead input is **compact and centred**, but "I don't know this person" and "Dismiss entire cluster" sit on the *same row* — two very different consequences (create a Stranger subject vs. kill the cluster) styled as sibling buttons with no visual separation beyond colour. Dangerous adjacency.
  - Hint line under the action row: "Click faces to select them for exclusion. Click thumbnails to view full photo." — This is the **only** place the selection mechanic is documented, and it lives inside a 14px low-contrast caption. See screen 15 below for the interaction discovery bug.

### 04 · Merge suggestions
![](./ui-assessment/04-merge-suggestions.png)

- **Prior claim**: "duplicates the cluster-review pattern with a different layout; merge is mis-cast as a destination".
- **Verdict**: **Refines / softens prior claim.** Rendered, this screen is **actually the most polished review surface in the app**. Clean pair layout, 4 representative faces per side, a centred similarity score (65.0% highlighted in green, 40%/50%/60%/70% threshold filter top-right), a single "Merge clusters" primary action. This is honestly better-designed than the single-cluster review.
- **New visual observations**:
  - Prior-report claim "nav-level, but it's really a review mode" is *correct but understates it*: this screen is the **template for what `/clusters/{id}` should look like** — two mosaics, a similarity score, threshold filter, one primary action. It's ironic that the secondary workflow has the affordances the primary workflow lacks.
  - Entries like "Cluster #201923 (4 faces) · 65.0% · Cluster #201925 (4 faces)" show **cluster IDs are exposed as user-facing identifiers** — fine for the user, but if these pairs were named ("Stranger #12 vs Stranger #14") or collaged-in-grid it would read even better.
  - The 40/50/60/70 threshold pills pattern is reused on find-similar (screen 07) — good consistency.

### 05 · Persons list (named subjects)
![](./ui-assessment/05-subjects-list.png)

- **Prior claim**: "live client-side filter already makes /search redundant".
- **Verdict**: **Confirmed** — top-right has a "Show strangers" toggle and evident live filtering. 6-column grid of face-portrait cards with name beside. Clean, scannable.
- **New visual observations**:
  - **Overflowed name text** — several cards show truncated names like "Ale Di Sipio", "Alessandro B...", "Amica di Tec...". At 1440w there's no hover-reveal. This is the one screen where more horizontal space per card would help.
  - Sort is alphabetic — no "by face count desc" or "recently edited" option visible, which seems like it would be valuable once you have 135 people.
  - The face-portrait cards get a rounded square avatar + name. Compare with **circles list** (screen 08) where entries have no avatar — inconsistent treatment of entity rows across the app.

### 06 · Subject detail (Caterina — 6,296 faces)
![](./ui-assessment/06-subject-detail.png)
Header-only crop: ![](./ui-assessment/18-subject-header-crop.png)

- **Prior claim**: "three different layouts for the same face-grid + actions pattern".
- **Verdict**: **Strongly confirmed and now visibly damning when cropped side-by-side with the cluster header (17) and circle header (9).**
- Direct comparison:
  | | Cluster | Subject | Circle |
  |-|-|-|-|
  | Title | `Cluster #183275 (105 faces)` | `Caterina (6296 faces in 5526 photos)` | `Famiglia (3 members)` |
  | Primary action | typeahead + "Dismiss entire cluster" | Rename input + Rename + "Show similar unclustered" + "Merge into" + "Delete person" | *(none)* |
  | Filter/Segment | — | Tabs: Face Samples / Photos / Videos | — |
  | Meta row | Suggestions | Circles chips + "Add to circle" | — |
- Three completely different headers, different action groupings, different vertical density. The subject-detail header alone is 3 rows of controls; the circle-detail header is just the title. All three wrap the same underlying concept (entity + list). The prior recommendation (#4 — unify via `partials/entity_header.html`) is **even more justified than the text suggested**.
- **New visual observations**:
  - Subject-detail action row has three different button styles side-by-side (outline black "Rename", filled orange "Show similar unclustered", outline grey "Merge into" typeahead). Looks like three designers.
  - "Delete person" is in the far right of the action row — red text on grey, but visually equivalent weight to a harmless tab change below. Destructive action needs either a guard or visual weight (boxed outline) to stand out.

### 07 · Find similar (empty state)
![](./ui-assessment/07-find-similar.png)

- **Prior claim**: none specific.
- **Verdict**: Rendered beautifully. "No similar unclustered faces found above 55.0%." with a centred magnifying-glass glyph and a back link. Good empty state copy. Threshold pills 40/50/55/60/70 top-right, 55 selected (orange). Minor: "← Back to Caterina" bottom-left is a bit orphaned — could live in the header's back-arrow position.

### 08 · Circles list
![](./ui-assessment/08-circles-list.png)

- **Prior claim**: none specific in prior report.
- **Verdict**: Clean, functional. Top-of-page "create circle" form is well-integrated (name + optional description + "Create circle" button). Six rows, each with name + member count.
- **New visual observations**:
  - "Strangers" row gets a description "Auto-created faces the user explicitly marked as unknown." and the description is rendered in the *list row*, while the other circles with no description just show the count. Inconsistent row height visually.
  - No avatar/stack for circles (which would be doable — 3 overlapping face avatars per row). Makes the list feel like a settings screen more than a working surface.

### 09 · Circle detail (Famiglia)
![](./ui-assessment/09-circle-detail.png)

- **Prior claim**: "no faces, just a member list".
- **Verdict**: **Confirmed — and the visual emptiness is dramatic at 1440×900**. The entire rendered page is `< Famiglia (3 members) / person Agnese / person Caterina / person Cesare` in a narrow strip. 85% of the screen is black. No add-member affordance visible on the page (have to go somewhere else to add). No member portraits. No "photos of these people together" link (even though `/together` exists!).
- **New visual observations**:
  - The kerned "person" caption before each name in tiny lowercase letters is a cute but unusual type pattern — looks like a placeholder.
  - Opportunity: a Circle detail should be the **landing page for browsing photos where these people appear together**. Today it's a member-picker at best. Elevate `/together?subjects=2,3,5` as the primary content.

### 10 · Together (empty form)
![](./ui-assessment/10-together.png)

- **Prior claim**: none.
- **Verdict**: Simple form: Type-to-search · "Find sources together" button · Either/Photos/Videos pill group · "Just them (exclude group photos)" checkbox. Clean; well-scoped.
- **New visual observations**:
  - Vast empty bottom — a "recent searches" or "popular combinations" panel would make this page not feel empty on first arrival.
  - The pill group `Either / Photos / Videos` is visually identical to the Humans/Pets pill on the top nav — good consistency with a known component.

### 11 · Compare (empty form)
![](./ui-assessment/11-compare.png)

- **Prior claim**: "rare use case; hide in More".
- **Verdict**: Refines prior recommendation — the form is a visibly trivial "two inputs vs each other + Compare button" that fits in 120px of height. Having it as a top-nav item makes the nav look busier than it needs to be for a tool that costs nothing to tuck into subject-detail's overflow.

### 12 · Singletons (Unmatched Singles)
![](./ui-assessment/12-singletons.png)

- **Prior claim**: "19K singletons, high volume during setup".
- **Verdict**: Confirmed. Rendered as a tight 10-column face grid. "Unmatched Singles (18,996 faces, showing 200) / Faces that didn't match any cluster. Select to dismiss or assign. Show first 200." Clean.
- **New visual observations**:
  - "Show first 200" is both the subtitle *and* the only affordance to load more. Should be a paginator / "Show next 200" button not a subtitle string.
  - No threshold-filter pills here like the merge/find-similar screens have. Similarity isn't the axis of this screen, but "quality" or "date" or "source" might be.

### 13 · Search
Empty: ![](./ui-assessment/13a-search-empty.png)
With query "C": ![](./ui-assessment/13b-search-c.png)

- **Prior claim**: "dead page, redundant vs nav typeahead and /people filter".
- **Verdict**: **Confirmed — and visually, `/search?q=C` is almost a perfect clone of `/people`** (6-col portrait-card grid). The only delta: the search page lacks the "Show strangers" toggle and has a dedicated search input above the grid. Strong case for deletion + redirect to `/people?filter=<q>`.

### 14 · Photo detail
![](./ui-assessment/14-photo-detail.png)

- **Prior claim**: "reload after every single face assign".
- **Verdict**: Confirmed structurally (this view does have 3 face tiles below the main image, each with name + delete-×). Clean layout: large photo above, "Faces in this photo" + 3 thumbs + names below, "Download original" top-right.
- **New visual observations**:
  - All three faces are already assigned (Teresa / Agnese / Caterina). This happens to be a "clean" photo — if we want to see the assignment flow we'd need an un-named-face photo. Note for a follow-up pass.
  - Path metadata at the bottom (`/Users/iacopo.papalini/.../2013_01_01_02_24_17.jpg`) is shown in tiny monospace — useful to you specifically, feels like debug UI in a normal app.

### 15 · Interaction: cluster with 3 faces selected
Initial attempts (opened lightbox instead — see below): ![](./ui-assessment/15-cluster-selection.png) ![](./ui-assessment/15c-cluster-cornerclick.png)

Final, triggered via Alpine `toggle()` directly: ![](./ui-assessment/15-cluster-selection-final.png)

- **Discovery / bug**: **On first render, clicking a face tile in cluster-detail opens the lightbox rather than selecting.** The tile has `@click="toggle(id)"` on the wrapper div but the child `<img>` inside a lightbox-registered subtree intercepts pointer events, so Playwright's click (and likely a real user's click) lands on the image and opens the lightbox. The hint text under the assign row says "Click faces to select them for exclusion. Click thumbnails to view full photo." — which implies two targets, but they're nested and the inner one wins. Real-world reproducibility worth a human test.
- Once selection is forced via Alpine: the sticky action bar works well — "3 selected" on the left, "Exclude from cluster" and "Not a face" (primary-orange + red) — clean, well-positioned. Selected tiles get the orange ring-offset style the prior audit praised. That part is fine.
- **Refines prior report**: cluster-detail selection UI is well-designed on paper, but **the click target for selection vs lightbox appears broken**. This should be added to the bug list.

### 16 · Interaction: subject-picker typeahead open
Accidental (top-nav input): ![](./ui-assessment/16-subject-picker-open.png)
Correct (cluster assign typeahead): ![](./ui-assessment/16-subject-picker-final.png)

- **Prior claim**: "subject_picker reused everywhere; component grammar solid".
- **Verdict**: **Confirmed and very nicely rendered.** Typing "Cat" into the assign input drops a compact 5-row menu: "Amica di Caterina ???" / "Caterina" / "Caterina - baby" / "Emma amica di Cater..." / "Marina amica di Cater...". Fuzzy match reaches substrings, not just prefixes. Doesn't crowd the page, positioned below the input, Escapes cleanly.
- **New visual observations**:
  - No avatar thumbnails in the dropdown — at 135+ named subjects, having the 48px face portrait beside each candidate would massively speed disambiguation ("which Caterina?").
  - The "???" placeholder on "Amica di Caterina ???" looks like a TODO — is that real data? Looks like a naming artifact that should be cleaned up.
  - The dropdown has no empty-state / "create new" affordance. Design guide says "no dropdowns for assignment — use search-based typeahead with inline 'create new'" — this **almost** respects that but lacks the inline "+ Create new person" row at the bottom. Might be present on no-match; worth verifying.

### 19 & 20 · Responsive at 900px width
Dashboard: ![](./ui-assessment/19-dashboard-900w.png)
Clusters: ![](./ui-assessment/20-clusters-900w.png)

- **New finding (not in prior report)**: at 900px-wide (tablet / split-screen desktop), **the top-right "Search faces…" input gets clipped** — the design guide says "tablet-friendly" but the nav clearly doesn't flex or collapse at this width. The input literally falls off the right edge.
- Dashboard stats reflow to 2-column which actually looks *better* than the 4-column grid on wide — again suggesting the wide-screen layout is accidental not deliberate.

---

## Revisions to the prior recommendation

**Targeted-fixes-not-rehaul verdict holds.** Visual inspection made that stronger: the component grammar (mosaic cards, selection bar, typeahead, lightbox, toasts) is genuinely solid when rendered. The architecture / flow issues the prior audit named are real — they're not camouflaging design-system problems.

Ranked deltas:

- **#1 Rebuild Dashboard** — **Stronger**. The empty 4th slot + disconnected numbers + 60% bottom whitespace + underweight CTAs make the dashboard visibly worse than the template read implied.
- **#2 Queue context + Skip/Next on cluster detail** — **Stronger**. The rendered cluster-detail has one-line-of-low-confidence-suggestions-then-105-face-grid, with a dangerous "Dismiss entire cluster" red button adjacent to "I don't know this person". No progress, no next, no similar-cluster sidebar. Fixing this is the biggest single-screen win.
- **#3 Shrink nav; demote Humans/Pets** — **Same strength, minor correction**: nav is 8 elements rendered (not 10), but the critique still applies. Also add: responsive clipping at 900w is a nav-level bug — a narrower nav would also fix that.
- **#4 Unify entity-detail layout** — **Much stronger**. The three header crops (cluster 17, subject 18, circle 09) side-by-side are the clearest evidence in the whole report. This is no longer "drift", it's three separate designs for one component.
- **#5 Fix `location.reload()` violations** — **Same**. Visual pass can't easily confirm without triggering writes; structural claim stands.
- **#6 Fold `/search` into nav** — **Stronger**. The rendered `/search?q=C` is nearly pixel-identical to `/people` with a filter — trivial to delete.

### New items to add

- **(H7) Cluster-detail selection click target is broken** — clicking a face opens the lightbox instead of selecting. Hint text promises selection; reality delivers lightbox. Needs a real-user repro and a fix to stopPropagation / z-order.
- **(H8) Circles is the most under-built entity** — circle-detail is a 3-line list on a black page. It should be the landing for "photos where these people appear together" (i.e. a pre-seeded `/together` query).
- **(H9) Responsive break at 900w** — top-nav search clips; a real tablet user would hit this.
- **(H10) Subject-picker dropdown lacks face avatars** — at 135+ named subjects, adding a 32-48px face thumb per row would materially reduce mistakes.
- **(H11) Destructive-action adjacency** — "Dismiss entire cluster" (nuclear) sits next to "I don't know this person" (benign) on cluster-detail; "Delete person" sits in the same action row as "Merge into" / "Show similar" on subject-detail. Needs visual weight or separation.

Net: **keep the 6 prior recommendations, promote #4 to second priority** (it's the most obviously fixable with the biggest "looks like one app" payoff), add H7 as a bug, and H8/H10 as small polish wins.

---

## Missing affordance: move a finding between known subjects

User-stated use case: pick a face on Alice's page (or in a cluster, or on a photo) and move it **directly to another known person** — not unassign-then-reassign. Two clicks, not four, no intermediate "unnamed" state, one undo token.

### Backend: already supports it
- `POST /api/findings/swap` (`src/ritrova/app.py:295`) — takes `face_ids[]` + `target_person_id`, reassigns atomically, returns undo token. Snapshot preserves prior `person_id` so Undo restores.
- `POST /api/subjects/{id}/claim-faces` (`src/ritrova/app.py:334`) — target-side equivalent with a 409 `needs_confirm` when the face already belongs elsewhere (species mismatch, etc.).
- No DB or API work needed.

### UI coverage today
| Surface | Per-face action exposed | Move-to-subject? |
|---|---|---|
| `photo.html` (face grid under the photo) | Click name pill → `subject_picker` → swap | **Yes, but undiscoverable** — pill has no affordance hint that it's clickable; `location.reload()` after swap (`photo.html:122`) violates design guide. |
| `subject_detail.html` (face grid tabs) | Hover `×` → confirm → **unassign only** (`subject_detail.html:159-164`) | **No.** User must unassign, navigate, search, reassign — 4 clicks + context loss. |
| `cluster_detail.html` (face grid) | Hover `×` → **exclude from cluster** | **No.** Semantics are different (this face doesn't belong in this cluster) but a "move to a named person" shortcut would also make sense for the common "I spotted Bob inside Alice's cluster" case. |
| Selection bar (multi-select) | "Dismiss", "Exclude" | **No** "Move to…" button, even though swap API already accepts `face_ids: list[int]`. |

### Recommended pattern (one canonical per-face menu)

Replace the current single-purpose `×` button with a small **action menu** (icon-button opens a popover):

- **Move to…** → opens `subject_picker` typeahead (with face avatars per H10) → on select, calls `/api/findings/swap` → DOM patch the face tile (update pill) + 15s undo toast. No reload.
- **Unassign** (secondary) — current behaviour, keep it for the "I don't know yet" case.
- **Dismiss as non-face** (destructive, muted) — current singleton/cleanup behaviour, promoted here.

On the **selection bar**, add a "Move selected to…" primary action that uses the same picker with the batch variant of `/api/findings/swap` (already supports `face_ids[]`). Multi-select 12 faces on Alice's page → "Move to…" → type "Bob" → done.

On `photo.html`, keep the click-the-pill interaction but:
1. Add a visual affordance (chevron/edit icon on hover, or underline-on-hover) so the pill reads as editable.
2. Replace `location.reload()` with a DOM update of the face tile (swap the pill text + color).
3. Make the face bbox overlay labels reflect the new name without reload.

### Where this lands in the priority list

This sharpens and extends **recommendation #4 (unify entity-detail layout)**. The unified per-face action menu is the concrete deliverable; without it, "unify" is an aesthetic claim. Promote this sub-item to an explicit line in the top 6:

> **#4b Canonical per-face action menu** (Move / Unassign / Dismiss) — **S/M, high impact**. Touches `cluster_detail.html`, `subject_detail.html`, `photo.html`, `singletons.html` (and the shared face-grid partial this would naturally produce). Backed entirely by existing endpoints.

### Decision: clusters and subjects stay semantically separate

Clusters group similar findings; the only per-face cluster action is **"remove the intruder"** (exclude). They are not a place for reassignment — that's a subject-level operation. Two menus, not one.

**Per-face affordances by surface:**

| Surface | Per-face menu |
|---|---|
| `cluster_detail.html` | **Exclude from cluster** (only). Single icon-button, no menu needed. |
| `subject_detail.html` | **Move to…** / **Unassign** / **Dismiss as non-face** |
| `photo.html` | **Move to…** (via pill) / **Unassign** / **Dismiss** |
| `singletons.html` | **Assign to…** (primary) / **Dismiss** |

**"Nuke this whole cluster"** already exists (`cluster_detail.html:78-84` — "Dismiss entire cluster", red button, confirm dialog). Keep the feature but **demote its visual weight** — it's destructive and low-frequency, so it should not sit in the primary header action row next to benign operations like "I don't know this person" (H11 adjacency issue). Move it to a small icon-button in an overflow menu or to the page footer, muted, with the same confirm dialog.

So **#4b simplifies**: the canonical menu is the three-item one (Move / Unassign / Dismiss), and it applies to subject/photo surfaces only. Cluster-detail keeps its single-purpose `×` (exclude) and a demoted "dismiss entire cluster" at the edge of the page.

