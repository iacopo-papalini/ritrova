# ADR-007: Auto-assign workflow

## Status

Accepted

## Context

After clustering and manually naming some clusters in the web UI, users often have many remaining unnamed clusters that belong to already-named persons. Re-clustering with different parameters can help, but it requires clearing all cluster assignments and starting over.

We needed a way to bulk-assign unnamed clusters to existing named persons without re-running the full clustering pipeline. The naive approach -- adding all unnamed faces to the embedding matrix and re-clustering -- has two problems:

1. It is O(n^2) in the total number of faces (see ADR-002), which becomes expensive as the collection grows
2. It destroys existing cluster assignments, requiring the user to re-review everything

## Decision

We implemented an **auto-assign** workflow that uses centroid-based matching:

1. **Compute person centroids**: for each named person, compute the mean of all their face embeddings and L2-normalize it. This produces a single representative vector per person.

2. **Compute cluster centroids**: for each unnamed cluster, compute the mean of all its face embeddings and L2-normalize it.

3. **Match by cosine similarity**: for each unnamed cluster, compute cosine similarity against all person centroids. If the best match exceeds the `min_similarity` threshold (default 50%), assign all faces in that cluster to that person.

The complexity is **O(clusters x persons)** for the matching step, plus O(faces) to compute centroids. This is dramatically cheaper than full re-clustering which is O(faces^2).

The auto-assign command operates per-species (default `human`), so human clusters are only matched against persons who have human faces, and dog clusters against persons who have dog faces.

### CLI interface

```bash
uv run face-recog auto-assign --min-similarity 50.0 --species human
```

The `--min-similarity` parameter is specified as a percentage (0-100) on the CLI and converted to a 0-1 cosine similarity internally.

## Consequences

**Positive:**

- Fast: matching 1000 unnamed clusters against 100 persons takes seconds, not minutes
- Non-destructive: existing person assignments and cluster assignments are preserved. Only unnamed clusters (where `person_id IS NULL`) are affected.
- Incremental: can be run repeatedly as more persons are named in the UI. Each run picks up newly-named persons and tries to match remaining unnamed clusters.
- The similarity threshold gives users control over precision vs recall. A higher threshold (e.g., 70%) produces fewer but more confident assignments.

**Negative:**

- Centroid matching is less accurate than embedding-level comparison. A cluster containing a few outlier faces will shift its centroid away from the true identity, potentially causing a mismatch.
- The approach assumes each unnamed cluster corresponds to a single identity. If clustering produced a mixed cluster (two people merged), auto-assign will assign all faces in that cluster to one person, which is wrong.
- The 50% default threshold is relatively permissive. For collections with many similar-looking people (e.g., large families), a higher threshold may be needed to avoid false assignments.
- Auto-assign cannot create new persons -- it only matches against existing named persons. Truly new people still need to be named manually in the UI.
