# Project Knowledge Index

> {{PROJECT_DESCRIPTION}}

This is the core-knowledge tree, maintained by Kipi. Every chapter is incrementally built from code scans and curated user-data. Strategic decisions live in `decisions/` and are written by `/grillme` or imported via `/kipi process`.

## Chapters

{{CHAPTER_LIST}}

## Modules with their own knowledge

{{INCLUDED_MODULES}}

## Decisions

See [`decisions/`](./decisions/) for ADRs (architecture decision records).

## How this is maintained

- `/kipi scan <path>` — deep-document a subtree
- `/kipi process` — process files dropped in `user-data/`
- `/kipi update` — reconcile recent code changes
- `/kipi status` — coverage report
- `/kipi-verify light|deep` — read-only audit: citations, ADRs, staleness, install
- `/grillme` — drive out tradeoffs adversarially, files an ADR into `decisions/`

_Last updated: {{DATE}}_
