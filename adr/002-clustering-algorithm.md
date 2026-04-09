# ADR-002: Clustering algorithm

## Status

Accepted

## Context

After extracting face embeddings, we need to group them into clusters representing the same person. The initial implementation used DBSCAN (Density-Based Spatial Clustering of Applications with Noise) with cosine distance. While DBSCAN worked for small collections, it suffered from a fundamental problem at scale: **chaining**.

With DBSCAN's reachability-based approach, face A can be similar to face B, and face B similar to face C, causing A and C to end up in the same cluster even if they are not similar to each other at all. This is especially problematic for:

- Family members who share some facial features
- A person aging over decades (baby photos chain through childhood into adulthood)
- Low-quality detections acting as "bridges" between distinct identities

The original DBSCAN parameters were `eps=0.50` (cosine distance) and `min_samples=2`.

## Decision

We switched to **agglomerative hierarchical clustering with complete linkage** using scipy's `linkage` and `fcluster`.

The algorithm:

1. Compute all pairwise cosine distances via `pdist` (after L2-normalizing embeddings)
2. Build the linkage tree using `method="complete"` -- complete linkage means the distance between two clusters is the **maximum** distance between any pair of their members
3. Cut the tree at `threshold=0.45` cosine distance using `fcluster(criterion="distance")`
4. Discard clusters smaller than `min_size=2`

Complete linkage guarantees that every member of a cluster is within the distance threshold of every other member. No chaining is possible.

## Consequences

**Positive:**

- Eliminates chaining: siblings, parents, and look-alikes no longer get merged
- The threshold parameter has a clear interpretation: "every face in this cluster is at most X cosine distance from every other face"
- Results are deterministic (unlike DBSCAN which can vary with point ordering)
- The default threshold of 0.45 works well for ArcFace 512-dim embeddings across tested collections

**Negative:**

- **O(n^2) memory and compute** for pairwise distances: this becomes expensive for large collections (tens of thousands of faces). For ~15k faces the distance matrix is ~900 MB. Iterative or batch approaches are planned for future scaling.
- Complete linkage is conservative -- it tends to split rather than merge, so the same person may end up in multiple clusters. This is mitigated by the auto-assign workflow (ADR-007) and the merge UI.
- Re-clustering clears all cluster assignments (person assignments are preserved), so the workflow encourages naming clusters first, then re-clustering with adjusted thresholds.
