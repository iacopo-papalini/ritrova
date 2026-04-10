# Ritrova

Personal photo archive: face + pet recognition, 61K+ faces, 100K+ photos, Apple Silicon.

## Must read before any work
- `docs/design-guide.md` — UI patterns, interaction rules, development conventions
- `docs/bugs.md` — open bugs and feature backlog

## Key rules
- UI variable: `kind` ("people"/"pets"). DB field: `species` ("human"/"dog"/"cat").
- No dropdowns for assignment — use search-based typeahead with inline "create new".
- No `confirm()` — use app dialogs. No `location.reload()` — use DOM updates + toasts.
- Every write action must be undoable (FEAT-5).
- Photo paths are the source of truth for dates, not EXIF.
- `/{kind}/...` page routes registered AFTER all `/api/...` routes.
