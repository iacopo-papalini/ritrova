# Ritrova Design Guide

Living document. Every UI pattern, interaction rule, and development convention goes here. When in doubt, follow this guide. When the guide is wrong, update it.

---

## Product Principles

1. **Review tool, not admin panel.** Every screen should help the user make a decision or find a memory. No dead-end stats pages.
2. **Safe to experiment.** Every write action must be undoable. Users should never fear clicking.
3. **People and pets are one family.** Retrieval crosses the kind boundary. Curation can be scoped, but search and discovery combine everything.
4. **Show the next step.** After any action, the UI should suggest what to do next — not dump the user back to a list.
5. **Respect the archive.** Photo paths are the source of truth for dates. EXIF is supplementary. Never modify source files.

---

## UI Patterns

### Assignment: search, not dropdowns

Never use `<select>` dropdowns for person/pet assignment. They break at scale and hide options.

**Use instead:** a text input with typeahead search. Show top matches as the user types. If the typed name doesn't match any existing entry, show "+ Create [typed name]" as the last option. Selecting it creates the person/pet inline and assigns immediately.

Applies to: cluster detail assign, singleton assign, photo face assign, any future assignment surface.

### Feedback: toasts, not reloads

After every write action, show a toast notification (fixed bottom-right, auto-dismiss ~8s) with:
- What happened: "Assigned 14 faces to Teresa"
- Undo button (see FEAT-5)

Never use `location.reload()` as the only feedback. Prefer local DOM updates (remove element, update count, swap content) with a toast confirmation.

Applies to: assign, unassign, dismiss, exclude, merge, rename, delete.

### Confirmation: app dialogs, not browser confirms

Never use `confirm()`. Use an Alpine-driven modal or inline confirmation:
- Show what will happen in plain language
- Two clear buttons: cancel (muted) and confirm (danger or accent)
- Escape key cancels

### Navigation: preserve context

After actions, keep users in their current review flow:
- Cluster assign → next cluster in queue (with skip/back), not clusters list
- Singleton assign → remove face from grid, stay on page
- Person detail unassign → remove face, stay on tab
- Never lose scroll position unnecessarily

### Progress: show the backlog

On review screens (clusters, singletons, merge suggestions), show:
- Queue position: "Cluster 12 of 87"
- Skip / Previous / Next controls
- A sense of momentum and completion

### Empty states: guide, don't just say "nothing here"

When a list is empty, explain what would populate it and what the user should do:
- "No unnamed clusters. All faces have been assigned! You can review singletons for stragglers."
- "No merge suggestions above 40%. Try lowering the threshold or review manually."

### Lightbox: context hub

The lightbox is not just a zoom view. It's a context panel. Show:
- Full photo path
- GPS link (when available)
- Date from path
- Who else is in this photo (when FEAT-6 lands)

### Responsive: desktop-first, tablet-friendly

- Design for desktop curation sessions (mouse, keyboard, large screen)
- Ensure grids and tabs work on tablet (touch targets, no hover-only actions without fallback)
- Phone is not a target — don't compromise desktop density for it

---

## Interaction Patterns

### Keyboard shortcuts (when implemented)

High-volume review screens (cluster detail, singletons) should support:
- `a` — assign top candidate
- `s` — skip / next item
- `d` — dismiss
- `l` — open lightbox on focused face
- `z` — undo last action
- `Esc` — close lightbox / cancel selection

### Selection model

- Click face thumbnail → toggle selection (amber ring + checkmark)
- Click face image → open lightbox (use `@click.stop` to separate from selection)
- Selection bar appears when any faces are selected (sticky, amber-themed)
- "Select all" / "Clear" always available in selection bar

### Per-face actions

- Hover reveals a small action button (top-right corner)
- Cluster detail: "x" removes from cluster (exclude)
- Person detail: "x" unassigns (not this person)
- Both use confirm dialog before acting

---

## Development Conventions

### Templates

- All styling via Tailwind utility classes. No custom CSS classes except in `input.css` for framework integration (x-cloak, htmx-indicator).
- No inline `<script>` tags. All JS in `app.js` via Alpine components/stores.
- Alpine for client state (selection, tabs, lightbox, toasts). htmx for server communication (infinite scroll, fragment loading, form submission).
- Partials in `templates/partials/` for htmx fragments. Each partial is self-contained.

### Routes

- Page routes: `/{kind}/...` where kind is `people` or `pets` (Literal type prevents catch-all conflicts)
- API routes: `/api/...` (registered before `/{kind}/...` routes)
- htmx fragment routes: `/api/.../...-html`
- Catch-all `/{kind}` routes registered last
- Legacy redirects at the very end for old URLs

### Backend

- `kind` in templates, `species` in DB queries. Map via `_species_for_kind()` / `_kind_for_species()`.
- No `is_pet` boolean — use `kind == 'pets'` in templates.
- `resolve_path()` handles relative paths: `tmp/` → db_path.parent, everything else → base_dir.
- `.fr_exclude` marker file to exclude directories from scanning.

### Testing

- pytest with TestCase classes
- Tests use real SQLite (no mocking the DB)
- Test routes via Starlette TestClient
- 131+ tests, all must pass before commit

### Git

- Pre-commit hooks: ruff lint, ruff format, mypy strict
- Commit after each coherent unit of work
- Tag before major refactors
