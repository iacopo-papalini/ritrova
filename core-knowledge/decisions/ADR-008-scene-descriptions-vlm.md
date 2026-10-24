# ADR-008: Scene descriptions and tags via vision-language model

## Status

Accepted

## Context

Ritrova has 100K+ photos with face/pet recognition but no understanding of *what's happening* in a photo. Searching means browsing by person or scrolling. The user wants to find "photos at the beach" or "Natale in montagna" without manually tagging anything.

Requirements:

- Generate a short Italian caption and a set of Italian tags per photo/video
- Tags must be in Italian — the user searches in Italian ("mare", not "sea")
- The model must be swappable — start with something fast, upgrade later
- Must integrate with a new advanced search that combines people/pets + tags + free text
- Must run on Apple Silicon (M-series Mac) at a pace that makes 100K photos feasible

## Decision

### Model: Qwen2.5-VL-3B via MLX (default, swappable)

**Why not Florence-2?** Florence-2-large was the original candidate (0.7B params, ~0.5s/img, already in `transformers`). It's English-only. Italian is mandatory — not "nice to have" — because tag vocabulary must match how the user thinks and searches. "Natale" is not interchangeable with "christmas" in a personal photo archive. Translation is fragile for short tags.

**Why MLX and not torch+MPS?** The project already has `torch` + `transformers` for SigLIP. MLX is a new dependency. We chose it because:

- MLX is purpose-built for Apple Silicon — quantized VLM inference is 2-4x faster than torch+MPS for the same model
- `mlx-vlm` provides a clean inference API for vision-language models with minimal boilerplate
- `uv` manages the dependency cleanly — no environment conflicts
- The speed difference matters at 100K scale: ~1s/img (MLX) vs ~3s/img (torch+MPS) = 28h vs 83h

**Why Qwen2.5-VL-3B?** Strong multilingual training (Italian is well-represented), 3B fits comfortably in memory on 16GB+ Macs, and it's available in MLX-quantized format. The `--model` CLI flag makes it trivial to switch to 7B or another model family later.

**Why not a larger model?** 7B would produce better captions but doubles inference time. For tag generation, 3B is sufficient — tags are short and factual ("mare", "montagna", "cena"), not creative writing. The user can re-run with 7B on a subset if 3B quality disappoints.

### Output: caption + colon-delimited tags

Each photo gets:

- **caption** (TEXT): one Italian sentence, e.g. "Un gruppo di persone a tavola in giardino durante l'estate"
- **tags** (TEXT): colon-delimited, e.g. `:giardino:estate:cena:gruppo:tavola:`

**Why colon-delimited and not a junction table?** Simplicity. Searching is `tags LIKE '%:mare:%'` — exact tag match with no false positives ("mare" won't match "tramonto"). A normalized `description_tags` table would be cleaner relationally but adds JOINs to every search query and a whole CRUD surface for tag management. At 100K rows with an index on `tags`, LIKE is fast enough. Accepted as a v1 tradeoff — migrating to a junction table later is a data transform, not a schema redesign.

**Why not FTS5?** Considered. FTS5 adds sync triggers and operational complexity. LIKE on the caption column handles free-text keyword search for v1. If search latency becomes a problem at scale, FTS5 is an incremental upgrade that doesn't change the schema.

**People/pets are just tags.** The VLM sees "three men and a dog at a table" — those are tags, not subject assignments. The existing `person_id` FK on findings is the authoritative identity link. The two systems coexist: structured recognition (findings → subjects) for identity, VLM tags for scene understanding. A search for "photos with Alessandro at the beach" combines both: `person_id = 57 AND tags LIKE '%:mare:%'`.

### DB schema: `descriptions` table with FK to `scans`

```sql
CREATE TABLE descriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    scan_id INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    caption TEXT NOT NULL,
    tags TEXT NOT NULL,
    generated_at TEXT NOT NULL
);

CREATE INDEX idx_descriptions_source ON descriptions(source_id);
CREATE INDEX idx_descriptions_scan ON descriptions(scan_id);
CREATE INDEX idx_descriptions_tags ON descriptions(tags);
```

**Why FK to scans?** A description run is a scan — it processes a source with a specific model and records what it found. The `scans` table already tracks `(source_id, scan_type, detection_strategy)` with a UNIQUE constraint on `(source_id, scan_type)`. A description scan uses `scan_type = "describe"` and `detection_strategy` records the model ID (e.g. `"qwen2.5-vl-3b-mlx"`).

This means:

- `ritrova scans list` shows description scans alongside face/pet scans
- `ritrova scans prune --source-pattern '2024/*'` cleans up descriptions too
- `ritrova rescan` can re-run descriptions
- Idempotency via `is_scanned(file_path, "describe")` — same pattern as face/pet scans
- A scan produces either findings (face/pet detection) or descriptions (VLM) or both (future "super scan")

**Why not columns on `sources`?** Adding `caption`, `tags` to sources conflates source identity with derived metadata. The separate table supports:

- Multiple descriptions per source (different models, A/B comparison)
- Clean deletion when re-running with a new model (`DELETE FROM scans WHERE scan_type = 'describe'` cascades)
- The `scans` infrastructure (list, prune, rescan) works without modification

### Tag vocabulary guidance via prompt

To prevent tag drift (within a run and across model switches), the prompt includes the existing tag vocabulary:

> "Descrivi questa foto in italiano. Usa questi tag quando appropriato: :mare:, :montagna:, :pranzo:, :natale:, :giardino:, ... Puoi aggiungere nuovi tag se necessario."

The vocabulary is loaded from the DB at batch start and refreshed every ~500 images. New photos reuse established tags; genuinely new concepts get new tags. When switching models, the existing vocabulary guides the new model toward consistent terminology.

### CLI: `ritrova describe`

```
ritrova describe [--model qwen2.5-vl-3b] [--batch-size 16] [--force] [--sample N]
```

- Follows the `scan` / `scan_pets` pattern: find unprocessed sources, load model, batch process, store results
- `--force`: re-run on sources that already have descriptions (deletes old scan + description, creates new)
- `--sample N`: process only N random unprocessed sources (for quality validation before committing to 100K)
- Idempotent: skips sources with an existing "describe" scan unless `--force`

### UI: Advanced Search (replaces Together)

The Together page becomes an advanced search with three inputs:

1. **People/pets picker** (existing typeahead, multi-select) + "just them" toggle
2. **Tags input** (autocomplete from known tags)
3. **Keyword input** (free-text LIKE search on captions)

Results are source-level: photo/video cards grouped by month, same layout as current Together results. The query combines:

```sql
-- People filter (existing Together logic)
SELECT source_id FROM findings
WHERE person_id IN (?) GROUP BY source_id HAVING COUNT(DISTINCT person_id) = ?

-- Tag filter
AND source_id IN (
    SELECT source_id FROM descriptions WHERE tags LIKE '%:mare:%' AND tags LIKE '%:cena:%'
)

-- Keyword filter
AND source_id IN (
    SELECT source_id FROM descriptions WHERE caption LIKE '%keyword%'
)
```

With no inputs filled, the page shows nothing (not "all photos") — the user must provide at least one filter. People-only with no tags = the old Together behavior.

## Consequences

### Positive

- Photos become searchable by content for the first time — "mare", "Natale", "montagna" find relevant photos instantly
- Italian tags match the user's mental model — no translation friction
- The `scans` + `descriptions` schema is clean: description runs are first-class scans, all existing tooling (list, prune, rescan) works
- Tag vocabulary guidance prevents drift across batches and model switches
- The `--sample` flag enables quality validation before committing to a 14-28 hour full run
- Model swappability is real: change `--model`, re-run, compare via `scans list`

### Negative

- MLX is a new dependency — adds ~500MB to the environment and only works on Apple Silicon (not portable to Linux servers, but this is a desktop app)
- Colon-delimited tags are not normalized — no tag frequency counts, no tag hierarchy, no "rename tag X to Y" without a text replace across 100K rows
- LIKE search on caption has no relevance ranking — results are "matches" or "doesn't match", not "best match first"
- The VLM adds ~3-4 GB of model weights to disk (quantized Qwen2.5-VL-3B)
- 100K photos at ~1s/img = ~28 hours for the initial run (acceptable as a one-time batch)

### Risks

**Caption quality on personal photos.** *Mitigated:* `--sample 100` validates quality before committing to the full archive. If 3B quality disappoints, switch to 7B and re-run.

**Tag vocabulary drift across models.** *Mitigated:* prompt-based vocabulary guidance. The existing tag set is fed to the new model, which reuses established terms. Not perfect — a fundamentally different model may still introduce synonyms — but good enough in practice.

**Colon-delimited format is a one-way door at scale.** *Accepted for v1.* Migrating to a junction table later is a data transform (`split on ':'`, insert rows), not a redesign. The format supports exact-match search, which is the primary use case.

**Together page identity change.** *Mitigated:* with no tags or keywords filled, the search behaves exactly like the old Together page. Existing muscle memory is preserved; the new fields are additive.

**MLX dependency locks the project to Apple Silicon.** *Accepted.* This is a personal desktop app running on a Mac. If portability becomes a requirement, the `Describer` interface can be backed by torch+transformers instead — the DB schema and CLI don't change.
