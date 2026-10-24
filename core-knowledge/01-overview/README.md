# Overview

> Ritrova finds people and pets again in a family photo archive: it detects every face and animal once, groups them by identity, and gives one household a web UI to name and curate the results.

## What it is

A single-household tool, not a product. One SQLite database, one person curating, a home server on the local network. That premise drives most of the design: no multi-tenancy, no authentication, no job queue, no horizontal scale. The hard numbers are 61K+ faces over 100K+ photos, which is large enough that every full-archive operation needs a plan and small enough to fit one SQLite file.

The work splits in two. **Analysis** is batch, expensive and offline: walk the photo tree, run detection models, write findings. **Curation** is interactive and cheap: look at groups of faces, give them names, fix mistakes. Everything in the architecture follows from keeping those apart — see [`../02-architecture/`](../02-architecture/).

## Who uses it

One or two family members. They are not administrators and will not read logs. Two consequences the code takes seriously:

- **Every write is undoable.** A single-slot undo store with a short TTL backs every mutation, and the UI always offers the undo before it expires. The rule is absolute in CLAUDE.md; where the code misses it, that is a defect, not a design choice.
- **Screens are for deciding, not for reporting.** Review surfaces show queue position, suggest the next item, and explain empty states. `docs/design-guide.md` is the authority on all of it.

## The vocabulary problem

One domain concept carries three spellings, and mixing them is the most common way to break this codebase.

| Layer | Name | Values |
|---|---|---|
| URL and template | `kind` | `people`, `pets` |
| Subject row | subject kind | `person`, `pet` |
| Finding row | `species` | `human`, `dog`, `cat`, `other_pet` |

`pet` also doubles as an umbrella query value that the database expands into the concrete pet species. The HTTP layer is allowed to know only the plural `kind` and the finding `species`; the singular subject kind belongs to the database and domain layers. Mapping helpers exist on both sides of that line and are deliberately not shared¹ ².

## Core nouns

- **Source** — one photo or video file. Immutable. The archive is never written to.
- **Scan** — one analysis pass over a source, under a named `scan_type`. Recording a scan with zero findings is meaningful: it is what stops the next run re-reading the file.
- **Finding** — one detected face or animal: bounding box, embedding, species, confidence. Detector output, never user-editable.
- **Cluster** — a machine-proposed group of findings believed to be the same individual.
- **Subject** — a named person or pet. The thing the user actually cares about.
- **Circle** — a user-defined grouping of subjects, such as a side of the family.

Findings are the fixed substrate; clusters and names are a sparse overlay on top of them, and absence of a row is the default state. [`../04-data-model/`](../04-data-model/) is where that stops being a slogan.

## Platform constraints

Apple Silicon is the first-class target and the only one where everything works. Face and pet detection run cross-platform, with CUDA on Windows and CPU elsewhere. VLM captioning runs on Apple Silicon only, through MLX, and is opt-in per run.

Model runtimes are an optional install. The base package serves, clusters and browses with no `torch` and no CUDA wheels; a separate extra adds the detection stack, and a third adds the VLM. A machine can therefore serve the archive without being able to analyse it — which is the intended split, with one exception for hand-drawn findings in the web UI. [`../07-operations/`](../07-operations/) has the boundary and its failure modes.

## Dates

Photo paths are the source of truth for dates. Directory and filename patterns are parsed first; EXIF is a last-resort fallback, and videos never carry EXIF dates at all. An archive organised into dated folders is the assumption the whole browse experience rests on.

## Related decisions

Every ADR in [`../decisions/`](../decisions/) constrains some part of this. The two that shape the overall shape most are [`../decisions/ADR-009-unified-analysis-pipeline.md`](../decisions/ADR-009-unified-analysis-pipeline.md) (one pass, N composable steps, persistence separate) and [`../decisions/ADR-012-architecture-refactor.md`](../decisions/ADR-012-architecture-refactor.md) (the package and layer boundaries).

## References

¹ `src/ritrova/app/helpers.py` — `species_for_kind`, `kind_for_species`
² `src/ritrova/db/paths.py` — `PathMixin`, `PET_SPECIES`, `KIND_TO_SPECIES`

---

_Last updated: 2026-10-24 by Kipi_
