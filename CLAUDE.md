# Ritrova

Personal photo archive: face + pet recognition, 61K+ faces, 100K+ photos, Apple Silicon.

## Must read before any work
- `docs/design-guide.md` — UI patterns, interaction rules, development conventions
- `docs/bugs.md` — open bugs and closed bug history
- `docs/features.md` — feature backlog and closed features

## Key rules
- UI variable: `kind` ("people"/"pets"). DB field: `species` ("human"/"dog"/"cat").
- No dropdowns for assignment — use search-based typeahead with inline "create new".
- No `confirm()` — use app dialogs. No `location.reload()` — use DOM updates + toasts.
- Every write action must be undoable (FEAT-5).
- Photo paths are the source of truth for dates, not EXIF.
- `/{kind}/...` page routes registered AFTER all `/api/...` routes.

<!-- kipi-pointer:do-not-remove -->

### Project knowledge base

This project has a Kipi-managed knowledge base under `core-knowledge/`. When you need to understand any part of the codebase, **start with the relevant chapter under `core-knowledge/`** — it carries curated, deeply-detailed documentation that explains structure, intent, and decisions. Don't re-derive the project's shape from raw source if the KB already covers it.

- Index: [`core-knowledge/index.md`](core-knowledge/index.md)
- Grow / refresh the KB while working: `/kipi scan <path>`, `/kipi update`, `/kipi process`
- Coverage report: `/kipi status`
- Side-channel findings (bugs, gaps, smells) accumulate in `KIPI_FINDINGS.md`. It is a live backlog, not a log: when you fix what a bullet describes, **delete the bullet**. Never mark it resolved, struck through, or done — git holds the history.

<!-- /kipi-pointer -->
