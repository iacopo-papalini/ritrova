---
name: kipi-setup
description: Use when the user wants to set up Kipi (a project knowledge keeper) in a project, says "set up kipi", "kipi setup", or runs /kipi-setup. Bootstraps a /kipi skill that builds and maintains core-knowledge/ documentation trees. Supports nested per-module KBs in monorepos via includes.
---

# Kipi Setup — Bootstrap the Core-Knowledge Keeper

## Overview

Kipi is a workflow that builds and maintains structured, deeply-detailed documentation trees (`core-knowledge/`) so anyone — even someone who has never seen the codebase — can understand, fix, or rebuild it.

> **⚠️ LANGUAGE — NON-NEGOTIABLE: write in Simplified English.** Short sentences. One idea per sentence. Common words over rare ones. Active voice. No idioms, no metaphors, no filler. The ONLY exception: domain-specific jargon, product/component names, and acronyms may be used as-is. This applies to everything this wizard writes, and it is carried into the generated `/kipi` skill so every chapter, index, and finding follows it too.

**Each invocation sets up exactly one KB.** Run from the repo root for a top-level KB; run from inside a module directory to set up a per-module KB that gets registered as a child of its nearest enclosing parent KB. Recurse arbitrarily deep — a "particularly nasty submodule" can have its own KB without polluting the parent's everyday context.

## When to Use

- User asks to "set up Kipi" / "install Kipi" / "bootstrap Kipi" in a project
- User runs `/kipi-setup`
- User wants to add a per-module KB inside an existing Kipi-managed repo

## Wizard Steps

### 1. Greet and orient

Greet warmly, in your own words but covering:

> "Hi! I'll set up Kipi — a Core-Knowledge Keeper — at this location. Kipi builds and maintains deep, structured docs incrementally and never writes without your approval. If this is a sub-module of a Kipi-managed repo, I'll register this KB as a child of the parent."

### 2. Resolve the KB location and detect parent

Use cwd as the KB root unless the user specifies a different path.

Walk up from cwd looking for an existing `<dir>/core-knowledge/`. Stop at the filesystem root or the git repo root.

- **Parent KB found** at `<parent>/core-knowledge/`: this invocation creates a **child KB** at `<cwd>/core-knowledge/`. Tell the user: "I see a parent KB at `<parent>`. I'll scaffold this as a child KB and register it in the parent's includes." Hooks/`/kipi` skill are already installed at the project root — skip those steps.
- **No parent KB**: this is a **top-level KB**. Hooks and the `/kipi` skill will be installed at the project root.
- **`<cwd>/core-knowledge/` already exists**: tell the user Kipi is already installed at this location and ask whether to re-run (overwriting templates) or abort.

### 3. Detect existing data

Look for a `user-data/` directory at the KB root.

- **Found**: "I see a `user-data/` directory — Kipi will process those files in its `process` workflow."
- **Not found**: "I'll create `user-data/`. Drop notes, specs, diagrams, transcripts there later and Kipi will classify them into structured knowledge."

### 4. Check git ignore status for `.claude/` (top-level only)

Skip this step for child KBs.

If the project is a git repo (`git rev-parse --is-inside-work-tree` succeeds):

1. Run `git check-ignore -q .claude/skills/kipi/SKILL.md` (the path doesn't need to exist — `git check-ignore` evaluates rules).
2. If exit code 0, `.claude/` is being ignored. Read `.gitignore` to find which rule does it.
3. Tell the user:

   > "Your `.gitignore` excludes `.claude/`. If Kipi should be shared with collaborators, the project skill at `.claude/skills/kipi/` and the handshake hook need to be tracked. Suggestion: replace the broad rule with narrower entries:
   >
   > ```gitignore
   > # before
   > .claude/
   >
   > # after
   > .claude/worktrees/
   > .claude/state/kipi-dirty
   > .claude/settings.local.json
   > ```"

4. Ask via AskUserQuestion: **[A]** Apply the narrower rules (recommended), **[B]** Keep `.claude/` fully ignored — Kipi stays local-only, **[C]** I'll edit `.gitignore` myself, continue setup.
5. On A: edit `.gitignore` (show the diff first, get approval).

If not a git repo, skip this step.

### 5. Two questions (REQUIRED — wait for answers)

Ask both via AskUserQuestion in a single call. The answers scope **this** KB only — for a child KB they describe the module, not the whole repo.

1. **What does this {project / module} do?** — Brief description of purpose, domain, and users.
2. **Which areas of {this codebase / this module} matter most to document first?** — Used to prioritize chapter creation.

### 6. Detect and migrate existing ADRs (top-level only)

Skip this step for child KBs (cross-cutting ADRs belong at the top level by default).

Scan candidate paths from project root: `docs/adrs/`, `docs/adr/`, `doc/adrs/`, `doc/adr/`, `adrs/`, `architecture/decisions/`, `architecture/adrs/`. Collect any `.md` files.

If found:

1. List the count and first few filenames per location.
2. Tell the user: "Kipi keeps decisions under `core-knowledge/decisions/`. I can migrate these at scaffold time. If this is a git repo, I'll use `git mv` so history is preserved. Filenames stay as-is — Kipi's NNN-detection accepts any leading-digits prefix."
3. If multiple locations exist, ask (multiSelect AskUserQuestion) which to migrate.
4. Per chosen location, ask: **[A]** Move (recommended), **[B]** Copy, **[C]** Skip. Defer the actual move/copy to step 8.

### 7. Propose layout

Based on the step 5 answers, propose a `core-knowledge/` chapter structure. Examples:

- Python data pipeline, priority "ingestion + scheduling": `01-overview/`, `02-architecture/`, `03-ingestion/`, `04-scheduling/`, `05-data-models/`, `06-deployment/`, `99-reference/`.
- React module, priority "auth + API": `01-overview/`, `02-architecture/`, `03-auth/`, `04-api/`, `05-state-management/`, `06-build/`, `99-reference/`.

Always include `01-overview/`, `99-reference/`, and `decisions/` (unnumbered, holds ADRs). Get explicit approval before writing.

### 8. Scaffold (after approval)

Create at the KB root (cwd):

```text
core-knowledge/
├── index.md                    # rendered from templates/index.md
├── 01-overview/README.md       # placeholder: "(pending — run /kipi scan)"
├── ... (each approved chapter)
├── 99-reference/README.md
├── decisions/                  # ADRs (top-level KBs); cross-cutting ADRs only for child KBs
├── ready-for-classify/         # /kipi process non-ADR output
└── .kipi/
    ├── manifest.json           # {"name": "<from cwd>", "scope": "<answer 5.1>", "includes": []}
    └── state.json              # {"version": 1, "processed_files": [], "last_workflow": null}
user-data/                       # if not already present
user-data/.archive/              # for processed originals
```

For a **top-level KB only**, also create:

```text
.claude/
├── skills/kipi/SKILL.md         # copied verbatim from templates/kipi-skill.md
├── skills/kipi/.version         # current source version (see "Version handling" below)
├── skills/kipi/upgrade-source   # OPTIONAL — where /kipi upgrade fetches from (see step 8.5)
├── hooks/kipi-handshake.sh      # copied from templates/kipi-handshake.sh, chmod +x
└── settings.json                # merged with hooks (see step 9)
```

**Version handling:** Read the source's `<source>/CHANGELOG.md` (where `<source>` is `${CLAUDE_CONFIG_DIR:-~/.claude}/skills/kipi-setup/`, following symlinks). Find the topmost `## N — date` heading; N is the current source version. Write that integer (and only that integer, e.g. `1`) to `.claude/skills/kipi/.version`. This file is what `/kipi upgrade` reads to decide whether the project needs syncing.

Skip the `.version` file for child KBs — version is project-wide, not per-KB.

### 8.5. Configure upgrade source (top-level only)

Skip for child KBs.

`/kipi upgrade` syncs the project's installed skill against the source. The default source is `${CLAUDE_CONFIG_DIR:-~/.claude}/skills/kipi-setup/` on the current machine (`CLAUDE_CONFIG_DIR` is the folder Claude Code keeps its config in; `~/.claude` when unset) — fine for solo work but breaks for collaborators who haven't installed the source skill locally.

Ask the user (AskUserQuestion):

- **[A] Local default (recommended for solo).** Don't write `upgrade-source`. `/kipi upgrade` uses `${CLAUDE_CONFIG_DIR:-~/.claude}/skills/kipi-setup/`. Each developer must clone/install the source skill on their own machine to use upgrade.
- **[B] Remote URL (recommended for shared projects).** Provide a base URL where the source's `CHANGELOG.md` and `templates/*` are HTTP-fetchable (e.g. `https://raw.githubusercontent.com/<owner>/<repo>/<branch>/kipi-setup`). Write that URL to `.claude/skills/kipi/upgrade-source`. `/kipi upgrade` will `curl <base>/CHANGELOG.md` and `<base>/templates/<file>` instead of reading from the local skill.
- **[C] Custom local path.** Provide an absolute path. Useful when the source skill is checked out somewhere non-standard.

If A, write nothing. If B or C, write the value as a single line to `.claude/skills/kipi/upgrade-source`.

Token replacements in `index.md`:

- `{{PROJECT_DESCRIPTION}}` ← answer to question 5.1
- `{{CHAPTER_LIST}}` ← markdown list of approved chapters
- `{{DATE}}` ← today's date in `YYYY-MM-DD`
- `{{INCLUDED_MODULES}}` ← empty initially; updated when child KBs register themselves (see step 10)

For migrated ADRs (from step 6): perform the move/copy now. Use `git mv` if in a git repo. Preserve filenames. If a source folder is empty after move, remove it. Then `grep -r '<old-path>' --include='*.md'` for inbound references and offer (with diff approval) to update them to the new path.

### 9. Wire the handshake hooks (top-level only)

Skip for child KBs.

Merge into `.claude/settings.json` (creating it if missing). If the file already has a `hooks` section, **merge** — never overwrite.

```json
{
  "hooks": {
    "PostToolUse": [
      {"matcher": "Edit|Write|MultiEdit", "hooks": [{"type": "command", "command": "${CLAUDE_PROJECT_DIR}/.claude/hooks/kipi-handshake.sh mark"}]}
    ],
    "Stop": [
      {"hooks": [{"type": "command", "command": "${CLAUDE_PROJECT_DIR}/.claude/hooks/kipi-handshake.sh check"}]}
    ]
  }
}
```

`${CLAUDE_PROJECT_DIR}` is injected by Claude Code as the absolute path to the project root. Relative paths fail when the hook's CWD isn't the project root (which happens whenever the prior tool ran with a `cd` elsewhere).

Never touch `.claude/settings.local.json`.

### 9.5. Add Kipi pointer to project `CLAUDE.md` (top-level only)

Skip for child KBs — the project-root `CLAUDE.md` is shared.

Without this pointer, future agents in the project re-derive structure from raw source on every conversation and never reach for the KB. A short block in `<project-root>/CLAUDE.md` fixes that.

1. Read `templates/claude-md-pointer.md` from this skill. The block is wrapped in `<!-- kipi-pointer:do-not-remove -->` … `<!-- /kipi-pointer -->` markers for idempotency.
2. Check `<project-root>/CLAUDE.md`:
   - **Marker `<!-- kipi-pointer:do-not-remove -->` already present**: pointer is installed; skip.
   - **File exists, no marker**: prepare to append the block (one blank line of separation) at the end of the file.
   - **File does not exist**: prepare to create `CLAUDE.md` whose sole contents are the pointer block.
3. Show the diff and get explicit approval via AskUserQuestion: **[A]** Apply (recommended), **[B]** Skip — I'll add it myself later.
4. On A, write the change.

Keep the block as-is — anything beyond the pointer belongs in the user's own `CLAUDE.md` sections or in the KB itself.

### 10. Register with parent KB (child KBs only)

If a parent KB was detected in step 2:

1. Read `<parent>/core-knowledge/.kipi/manifest.json`.
2. Compute the relative path from the parent KB root to this KB's root (e.g., `apps/frontend`).
3. Add that path to the parent's `includes` array if not already present.
4. Show the diff and get approval before writing.
5. Also update the parent's `index.md`: re-render the `{{INCLUDED_MODULES}}` section (or its rendered output) to list this child with a link.

### 11. Confirm and explain next steps

Tell the user:

> "Kipi is installed at `<kb-path>`. From now on:
>
> - `/kipi help` — explain what Kipi is and how the workflow fits together
> - `/kipi init` — draft the index from a fresh scan
> - `/kipi scan <path>` — deep-document a subtree (auto-routes to the nearest KB)
> - `/kipi process` — classify and extract knowledge from `user-data/`
> - `/kipi update` — reconcile recent code changes
> - `/kipi status` — coverage report (use `--recursive` to walk includes)
> - `/kipi-verify light` — read-only audit: citations, ADRs, staleness, install (`deep` adds prose checks)
>
> Side-channel findings (bugs, smells, gaps, open questions spotted during scans) land in `<kb-root>/KIPI_FINDINGS.md` — sibling of `core-knowledge/`, lazy-created on first finding. Process and clear them out-of-band; Kipi never reads them back.
>
> When the source skill gets updated (new templates / behavior), run `/kipi upgrade` to sync this project. The source is configured at `.claude/skills/kipi/upgrade-source` (or defaults to `${CLAUDE_CONFIG_DIR:-~/.claude}/skills/kipi-setup/`); its `CHANGELOG.md` lists what changed between versions.
>
> To add a sub-module KB later, `cd` into the module and run `/kipi-setup` again."

## Key Rules

- **One KB per wizard run.** Don't try to set up multiple modules in a single invocation; have the user re-run from each module's directory.
- **Hooks, the `/kipi` skill, and the `CLAUDE.md` pointer are top-level only.** Child KB setups skip steps 4, 6, 9, and 9.5.
- **Never write without approval.** Propose the layout, wait for "yes", then scaffold.
- **Never overwrite existing `core-knowledge/` content.** If found, list what's there and ask before any change.
- **Never edit `.claude/settings.local.json`.** That's user-managed; use `.claude/settings.json`.
- **Don't commit anything.** Per the user's global rule: never commit unless asked.

## Files in this skill

- `SKILL.md` — this file (the wizard).
- `templates/kipi-skill.md` — the per-project `/kipi` skill body, copied verbatim into `.claude/skills/kipi/SKILL.md`.
- `templates/index.md` — template for `core-knowledge/index.md`. Tokens filled in at scaffold time.
- `templates/kipi-handshake.sh` — `mark | check` hook script.
- `templates/claude-md-pointer.md` — the block injected into the project's `CLAUDE.md` so future agents start from the KB rather than re-parsing source.
