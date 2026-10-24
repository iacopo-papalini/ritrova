<!-- kipi-pointer:do-not-remove -->

### Project knowledge base

This project has a Kipi-managed knowledge base under `core-knowledge/`. When you need to understand any part of the codebase, **start with the relevant chapter under `core-knowledge/`** — it carries curated, deeply-detailed documentation that explains structure, intent, and decisions. Don't re-derive the project's shape from raw source if the KB already covers it.

- Index: [`core-knowledge/index.md`](core-knowledge/index.md)
- Grow / refresh the KB while working: `/kipi scan <path>`, `/kipi update`, `/kipi process`
- Coverage report: `/kipi status`
- Audit the KB (read-only): `/kipi-verify light` or `/kipi-verify deep`
- Side-channel findings (bugs, gaps, smells) accumulate in `KIPI_FINDINGS.md`. It is a live backlog, not a log: when you fix what a bullet describes, **delete the bullet**. Never mark it resolved, struck through, or done — git holds the history.

<!-- /kipi-pointer -->
