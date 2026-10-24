<!-- markdownlint-disable-file MD024 -->
<!-- Keep-a-Changelog format intentionally repeats "Features", "Migration", etc. under each version. -->

# Kipi Changelog

The topmost `## N` heading is the current source version. Bump it whenever any template or wizard behavior changes, or to announce a new shared skill (so `/kipi upgrade` surfaces it to every project). Each entry should describe what changed and any migration steps for already-installed KBs.

`/kipi upgrade` parses this file: it reads the topmost version as the source target, compares against the installed `<.claude/skills/kipi/.version>`, and prints all intermediate entries before applying file diffs.

Versions are monotonically-increasing integers. No semver — just `1`, `2`, `3`, ...

---

## 14 — 2026-10-14

The default upgrade source was hard-coded to `~/.claude/skills/kipi-setup/`. Claude Code keeps its config in `$CLAUDE_CONFIG_DIR` when that is set, so a machine with a moved config folder found no source.

### Changes

- **The default source is `${CLAUDE_CONFIG_DIR:-~/.claude}/skills/kipi-setup/`.** `/kipi upgrade`, `/kipi-setup` and `/kipi-verify` all resolve it this way. Nothing changes when `CLAUDE_CONFIG_DIR` is unset.
- An `upgrade-source` path may start with `$`; environment variables are expanded before the path is read.

### Migration

- Run `/kipi upgrade` to pull the new skill template. No KB content changes.

## 13 — 2026-09-10

Chapter citations carried a line number. Line numbers move whenever code is added above them, so the reference broke on edits that changed nothing about what the chapter said. One repo re-pointed 96 citations across 27 commits in six weeks.

### Changes

- **Chapters cite `path` — `Symbol`, never `path:line`.** Name the class, function or constant the footnote points at; use a short description only where no single symbol fits. The symbol is what a reader searches for, and the only part a checker can verify — a line number verifies nothing and rots on every edit above it.
- The chapter template's `## References` example drops its `:42`.
- `/kipi-verify` now checks that a References symbol appears in the file, and flags any References entry still carrying a `:line` suffix.
- **Findings are unchanged.** `KIPI_FINDINGS.md` bullets keep inline `file:line`: they are short-lived, and the exact location is the content of the observation.

### Migration

- Run `/kipi upgrade` to pull the new template and verify rules.
- Existing chapters keep working — nothing breaks on the old form, and `/kipi update` and `/kipi trim` convert footnotes as they touch them. To convert a KB in one pass, strip the `:NNN` from each `## References` entry and add the symbol where the entry names none.
- A project with its own anchor-checking hook (for example a `check_kb_anchors.py` in `scripts/`) must be updated to match, or it will fail on the new form.

---

## 12 — 2026-08-20

All Kipi prose must be written in Simplified English.

### Changes

- **Simplified English is mandatory.** `kipi-skill.md` now opens its Writing style section with a non-negotiable language rule: short sentences, one idea per sentence, common words, active voice, no idioms. Domain jargon, names, and acronyms are the only exception.

### Migration

- Run `/kipi upgrade` to pull the new template. Existing chapters are not rewritten; `/kipi trim`, `/kipi update`, and future `/kipi scan` passes apply the rule as they touch text.

---

## 11 — 2026-08-14

`KIPI_FINDINGS.md` kept turning into a log — agents annotated addressed findings instead of removing them, so the backlog read as full of open issues that were long fixed.

### Changes

- **Addressed findings are deleted, never marked.** The Findings section of `kipi-skill.md` now forbids `✅`, `~~strikethrough~~`, `(resolved)`, `(fixed)`, `DONE`, `[x]`, and Resolved/History/Archive sections outright — remove the bullet, and the `## YYYY-MM-DD` heading if it empties. Git carries the audit trail.
- The `CLAUDE.md` pointer block says the same, so any agent fixing a listed bug clears the line without reading the Kipi skill.
- `/kipi-verify light` flags annotated-instead-of-deleted findings as 🔴 broken.

### Migration

Re-run `/kipi upgrade` to pick up the skill and pointer text. Existing `✅ resolved` lines: delete them.

---

## 10 — 2026-08-03

Announce **`/kipi-verify`**, a new shared skill that audits a KB. It ships next to `kipi-setup` rather than being installed into the project, so it works against any installed version and can check the install itself.

### Features

- **`/kipi-verify light`** — mechanical audit, no chapter prose read: citation integrity (superscript ↔ `## References`, dangling targets, inline `path:line`), link resolution, staleness vs `last_documented_commit`, ledger text and drift-prone pins, ADR numbering / `Proposed`-or-`Accepted` status / historical records, KB wiring (`includes`, unregistered child KBs, stale `processed_files`), install integrity (version behind, hand-edited install, missing hook, absent `CLAUDE.md` pointer, dead `upgrade-source`), backlog, competing doc trees, and `KIPI_FINDINGS.md` hygiene.
- **`/kipi-verify deep`** — the above plus judgment checks: map-principle violations, cross-chapter duplication, template conformance, ADR relevance and terseness, drift spot-checks.
- Read-only by construction: writes nothing, not even `state.json`. Every finding names the subcommand that fixes it. A check that can't run reports `not checked` and is never turned into a finding.
- Diagram checks, following v9's Diagram style rules: node count and drift-prone node labels in `light`; function/class nodes, prose that restates the diagram, a needless `sequenceDiagram`, a missing required `## Data flow`, and depicted-flow drift in `deep`.
- `/bulk-review` runs `/kipi-verify light` on Kipi-managed repos — once on the default branch for standing debt, and again at the head of any PR that touches `core-knowledge/` so a broken citation or duplicate ADR number lands in that PR's findings. Obtains a tree via a detached worktree or a blobless temp clone. It also flags a PR that changes code inside a KB's scope without touching `core-knowledge/`.

### Changes

- `/kipi`'s router, `help` lifecycle, the `CLAUDE.md` pointer block, and `index.md` now point at `/kipi-verify`. `verify` is deliberately **not** a `/kipi` subcommand.

### Migration

None. Run `make update` in the shared-skills checkout (or `make install`) to link the new skill; `/kipi upgrade` only refreshes the pointers.

---

## 9 — 2026-08-03

Mermaid diagrams become first-class KB content: required for pipeline/flow chapters, planted at the top level by `init`, and kept in sync by `update`'s drift protocol.

### Features

- **Diagram style** section added to Writing style: diagrams follow the map principle — mermaid in the markdown, nodes are modules/services/queues (never functions or classes), ~15 nodes max, `flowchart`/`sequenceDiagram` preferred, no drift-prone labels.
- **Chapter Template `## Data flow`** upgraded from optional to **required** whenever the chapter covers a pipeline, request/event flow, background-job chain, or multi-service interaction.
- **`init`** now includes a top-level mermaid architecture diagram in `index.md` — module boundaries and included child KBs, with data direction left to the first `scan` where init's shallow read can't establish it.
- **`update`** treats diagrams as drift-checkable: changed code that alters a depicted flow triggers a corrected-diagram proposal through the standard Drift protocol, and a missing-but-required diagram counts as drift too. `index.md` joins the set of files `update` reconciles.

### Migration

`/kipi upgrade` syncs `kipi-skill.md` automatically; no hook or `settings.json` changes. Existing KBs gain diagrams lazily: re-run `/kipi init` to add the architecture diagram to `index.md`, and flow chapters pick up diagrams the next time `/kipi scan` or `/kipi update` touches them.

---

## 8 — 2026-06-19

Sharpen the writing-style guidance around the **map** metaphor, and add a `trim` subcommand to enforce it on chapters that already drifted.

### Features

- **`/kipi trim [path]`** — corrective counterpart to `scan`: re-reads existing chapters and cuts them back to a map (mechanical step-by-step, file-by-file enumeration, re-explained standard patterns, cross-chapter duplication). Approve per chapter; preserves `## References`; never touches `decisions/` or `KIPI_FINDINGS.md`.

### Changes

- Writing style now **leads** with the map principle and reframes the test as **derivation cost**, not "derivability" (everything is derivable from the code given enough reading — that's not the bar). Replaces the misleading "cut anything derivable from the code" rule.
- "A map, not the territory" is now the first **Principle**.

### Migration

Run `/kipi upgrade` to sync `.claude/skills/kipi/SKILL.md`. No state, manifest, or chapter changes. Optionally run `/kipi trim` afterward to tighten existing chapters.

---

## 7 — 2026-06-04

New skill available in the repo: `/react-review` (VAL-205). No Kipi template or wizard changes — this entry exists so `/kipi upgrade` surfaces the new skill to every project.

### Features

- **`/react-review`** — React-specific code review: memoization restraint, state placement (local/URL/cache), responsibility-based component SRP, effect/timer/async lifecycle hygiene, keys and identity, abstraction judgment. Complements `/clean-code`, which stays paradigm-agnostic.

### Migration

Not installed by Kipi. Copy `react-review/SKILL.md` from this repo into `~/.claude/skills/react-review/SKILL.md` (user-level, alongside the other shared skills).

---

## 6 — 2026-05-29

Fixes the handshake reminder, which never reached anyone (VAL-204). The `check` branch printed to stderr and exited 0 — but a `Stop` hook at exit 0 discards stderr and only debug-logs stdout, so the "code changed, consider /kipi update" nudge was visible nowhere except the Ctrl-R debug log. Claude never saw it; neither did the user.

### Fixes

- **`templates/kipi-handshake.sh` `check`** now emits `Stop`-hook JSON on stdout — `{"systemMessage": "…", "suppressOutput": true}` — instead of an stderr `echo`. `systemMessage` is the user-facing channel that actually surfaces. It renders emoji but strips ANSI colour, so the reminder uses emoji-and-unicode styling (the `/kipi update`, `/kipi scan`, `/kipi status` quick-reference) rather than escape codes.

### Migration

`/kipi upgrade` syncs `kipi-handshake.sh` automatically — no `settings.json` change is needed (the hook wiring is unchanged; only the script body changed).

If you can't upgrade, re-copy `templates/kipi-handshake.sh` from the source skill into `<project>/.claude/hooks/kipi-handshake.sh` (keep it `chmod +x`).

---

## 5 — 2026-05-28

A Kipi pointer in the project's `CLAUDE.md` makes future agents start from the KB instead of re-deriving structure from raw source on every conversation. Planted by the wizard on fresh installs and refreshed by `/kipi upgrade` on existing ones.

### Features

- **New template** `templates/claude-md-pointer.md` — the markdown block to inject, wrapped in `<!-- kipi-pointer:do-not-remove -->` … `<!-- /kipi-pointer -->` markers for idempotency. Tells future agents to start from `core-knowledge/`, lists the `/kipi` subcommands, and points to `KIPI_FINDINGS.md`.
- **Wizard step 9.5** (top-level KBs only): after wiring hooks, the wizard updates `<project-root>/CLAUDE.md` with the pointer block — or creates the file if it doesn't exist. Gated on user approval (diff + AskUserQuestion).
- **Upgrade step 8.5**: `/kipi upgrade` now refreshes the pointer as a *managed block*. It plants the block on pre-v5 installs (marker absent → append), updates the block in place when the source diverges (marker present → diff between markers only), and creates `CLAUDE.md` if missing. Content outside the marker pair is never touched.

### Migration

`/kipi upgrade` covers it: any pre-v5 install gets the pointer block planted automatically (with approval) once you upgrade. No manual copy required.

If you can't or don't want to upgrade, copy `templates/claude-md-pointer.md` from the source skill into your project's `CLAUDE.md` manually — the block is fully self-contained.

---

## 4 — 2026-05-20

Introduces the Drift protocol: KB/code inconsistencies are treated as stale documentation, not findings.

### Features

- **Drift section** added to `kipi-skill.md`: defines "drift" (KB says X, code does Y), explains why it is not a finding, and specifies the 3-step protocol — explain what drifted → ask for confirmation → update the KB on approval.
- **`scan` workflow** gains an explicit drift-detection step (batch step 3): when scanned files contradict existing chapter content, follow the Drift protocol rather than routing to `KIPI_FINDINGS.md`.
- **`update` workflow** (step 4) now states that KB/code mismatches are the expected, routine outcome of `update` runs — not a failure condition — and directs Kipi to apply the Drift protocol.
- **Findings section** adds an explicit exclusion: KB ↔ code inconsistencies are not findings and must not be logged there.
- **Principles** add: "KB ↔ code inconsistency is drift, not a finding."

### Migration

No template-file shape changes besides text. `/kipi upgrade` syncs `kipi-skill.md` automatically. No hook or `settings.json` changes required.

Existing installed KBs may have KIPI_FINDINGS.md entries that are really documentation drift (e.g. "chapter says X but code does Y"). These can be removed from KIPI_FINDINGS.md and addressed via `/kipi update` at your convenience.

---

## 3 — 2026-05-12

Tightens writing discipline in chapters, ADRs, and findings. Six new rules from field experience writing a real KB; nothing behavioral in the workflow code.

### Features

- **Writing-style additions** in `kipi-skill.md`:
  - "Describe present state, not history" promoted from ADR-only to a global KB rule. Explicit anti-patterns: `(introduced in RUP-NNN)` suffixes, "closes the historical gap", "the pre-RUP-NNN code did X". For new modules / workers: state what they currently do, not what they retroactively repair.
  - "Don't pin drift-prone identifiers" — no specific alembic revisions, no current package versions, no SHA pins. Refer to the convention or the always-fresh source.
  - "Skip temporary code" — one-shot backfills, migration helpers, gap-closer commands don't belong in chapters or runbooks. Their rationale stays in the ADR; the code itself disappears.
- **Data-model chapter scope** is now spelled out: conventions + entities-with-non-obvious-rules, NOT an exhaustive table/column catalog. The schema is in code; chapters add only what can't be inferred from `entities/*.py` or migration files.
- **ADR section** gains explicit anti-patterns: full JSON example payloads, per-flag tables, specific revision identifiers, package-version pins. ADRs describe what the shape *carries* and what role each piece plays.
- **Findings are a backlog**, not a ledger. The previous "append-only" rule is refined: append new entries, but **delete resolved ones** rather than annotating them `✅ resolved`. Git carries the audit trail.
- **Principles** add "Present state, not ledger" as a unifying line.

### Migration

No template-file shape changes besides text. `/kipi upgrade` syncs `kipi-skill.md` automatically. No hook changes; no `settings.json` edits required.

Existing KBs may carry artifacts that violate the new rules:

- Historical references in chapters (e.g. "closes the gap left by …") — rewrite when next `/kipi update` touches the chapter.
- `✅ resolved` lines in `KIPI_FINDINGS.md` — delete them at any convenient moment.
- DTO examples / flag tables / alembic revision references in ADRs — trim when the ADR next comes up for review or via a manual sweep.

---

## 2 — 2026-05-12

Citation style changed to footnote-and-bibliography. Hook commands and the handshake script now anchor to the project root via `$CLAUDE_PROJECT_DIR` instead of relying on the shell's CWD.

### Features

- **Superscript footnote citations** — chapter body text cites code via inline superscripts (`¹ ² ³ ... ¹⁰ ¹¹ ¹² ...`). The bibliography lives in a new `## References` section at the bottom of each chapter, mapping each footnote to `path/to/file.py:line`. Previously, citations were inline `file:line` strings, which broke visual flow.
- **Chapter Template** now includes the `## References` section.
- **Findings remain inline** — in `KIPI_FINDINGS.md`, the `file:line` *is* the content of each bullet, not a citation, so no change there.

### Fixes

- **Stop hook no longer fails with "No such file or directory"** when the shell's CWD isn't the project root. The wizard's generated `settings.json` example now uses `${CLAUDE_PROJECT_DIR}/.claude/hooks/kipi-handshake.sh`, and `templates/kipi-handshake.sh` anchors `STATE_DIR` to `${CLAUDE_PROJECT_DIR}/.claude/state` (with a guard that fails fast if the env var isn't set).
- **Findings are now explicitly observational, not prescriptive.** Added a "Describe, don't prescribe" rule under the Findings section of `kipi-skill.md` so future scans can't drift into code-review territory. Kipi states facts and consequences; the reader decides on solutions. Pure-prescription "findings" with no observed defect get dropped instead of recorded.

### Migration

**Chapter rewrites**: existing chapters keep working — old inline `file:line` cites still read fine; they're just inconsistent with the new style. When you next `/kipi update` a chapter, Kipi will rewrite the body in the new style and add a `## References` section.

**Hook fix**: `/kipi upgrade` syncs `kipi-handshake.sh` automatically, but `settings.json` is **never** touched by upgrade. Manually edit `.claude/settings.json` and replace `.claude/hooks/kipi-handshake.sh` with `${CLAUDE_PROJECT_DIR}/.claude/hooks/kipi-handshake.sh` in both the `PostToolUse` and `Stop` hook commands. Without this edit, the hook continues to fail with "No such file or directory" whenever the shell's CWD isn't the project root — the script update alone doesn't fix the symptom, it just makes the script CWD-robust *once invoked*.

---

## 1 — 2026-05-08

Initial versioned release. (Earlier development happened pre-versioning; treat any installed KB without a `.version` file as v0 and bring it to v1.)

### Features

- **Wizard** scaffolds a top-level KB or registers a child KB under the nearest enclosing parent. Supports arbitrary nesting via `manifest.json` includes.
- **Project skill** subcommands: `help`, `init`, `scan`, `process`, `update`, `status`, `upgrade`.
- **Findings sidecar** — bugs, smells, gaps, and open questions land in `<kb-root>/KIPI_FINDINGS.md` (sibling of `core-knowledge/`), never in chapter files.
- **ADR migration** — wizard detects ADRs at common locations (`docs/adrs/`, etc.) on first setup and offers to move them to `core-knowledge/decisions/` with `git mv`.
- **`/grillme` ↔ Kipi** integration — grillme reads `decisions/` to ground adversarial questions in past choices, and offers to file new ADRs back into the active KB.
- **Stop-hook handshake** — after code edits, prints a reminder to run `/kipi update`.
- **Writing-style discipline** — chapters group by module/responsibility (not file enumeration), name patterns instead of explaining them, omit empty sections.
- **Upgrade source is configurable** — `.claude/skills/kipi/upgrade-source` (one line, optional) overrides the default `~/.claude/skills/kipi-setup/`. Accepts a local path or an HTTP(S) base URL, so collaborators without the source skill installed locally can still run `/kipi upgrade`.

### Migration

None — first versioned release. Pre-versioned installs (no `.version` file present) get treated as v0; running `/kipi upgrade` brings them to v1 by syncing the templates and writing the version file.
