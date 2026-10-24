# ADR-003: Person age splitting strategy

## Status

Accepted

## Context

A key challenge in personal photo collections is recognizing the same person across wide age ranges -- particularly children who change dramatically from infancy through adolescence. ArcFace embeddings are trained primarily on adult faces and encode current appearance, not identity persistence over time.

Observations from testing:

- Babies (0-3 years) have round, relatively featureless faces that produce embeddings in a distinct region of the embedding space. Two different babies often have more similar embeddings than the same baby at age 1 vs age 5.
- From roughly age 3 onward, facial features become distinctive enough for ArcFace to track across moderate age changes (e.g., age 5 to age 12 often clusters together).
- Centroids computed across very wide age ranges (e.g., age 1 to age 30) produce a "meaningless average" that doesn't actually resemble the person at any age, leading to poor matching.

Research literature on face recognition across age gaps supports the observation that recognition accuracy drops sharply for subjects younger than about 3 years.

## Decision

We accept that baby photos (roughly age 0-3) will form separate clusters from the same person's later photos, and handle this through the merge workflow rather than trying to force a single cluster.

The approach:

1. **Clustering operates purely on embedding similarity** -- no age-aware logic
2. Baby faces naturally cluster separately due to their distinct embedding characteristics
3. Users merge baby clusters with the person's later clusters using the web UI's merge feature
4. The auto-assign workflow (ADR-007) uses centroids, so once some baby photos are manually assigned to a person, remaining baby clusters for that person can be auto-matched

We intentionally do not attempt to bridge the baby-to-child gap algorithmically because:

- Lowering the clustering threshold enough to bridge it would merge unrelated babies and siblings
- Age metadata from EXIF is unreliable and not available for all photos
- The merge UI makes this a one-time manual step per person

## Consequences

**Positive:**

- Avoids false merges between different babies/toddlers
- Keeps clustering parameters simple and interpretable
- Manual merge is a one-time cost per person and correctly handles edge cases
- After initial merge, auto-assign can pick up remaining baby photos via centroid matching

**Negative:**

- Users must manually merge baby clusters for each person -- this is extra work for family collections with many young children
- The centroid of a person spanning baby-to-adult ages is still a diluted average, which can reduce auto-assign accuracy for extreme age gaps
- There is no automated way to suggest "this baby cluster might be the same person as this adult" -- the user needs domain knowledge (knowing which baby is which)
