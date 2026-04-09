# ADR-002: Clustering algorithm

## Status

Accepted (updated)

## Context

After extracting face embeddings, we need to group them into clusters representing the same person. The initial implementation used DBSCAN which suffered from **chaining** at scale. We then switched to scipy's agglomerative clustering with complete linkage, which solved chaining but had O(n²) memory for the full pairwise distance matrix (~14GB for 61K faces).

## Decision

We use a **two-phase FAISS-accelerated clustering** approach:

### Phase 1: Candidate neighbor discovery
- Build a FAISS `IndexFlatIP` (inner product on L2-normalized vectors = cosine similarity)
- Run `range_search` to find all pairs within the cosine distance threshold
- Build a sparse adjacency graph from these pairs
- Find connected components via BFS

### Phase 2: Exact verification
- For each connected component (candidate cluster), run **exact complete-linkage** hierarchical clustering using scipy `pdist` + `linkage`
- This prevents chaining: connected components give single-linkage behavior, but the verification step enforces complete linkage
- Components larger than 500 faces are split via FAISS k-means first, then verified per sub-group

### Species isolation
- Humans, dogs, and cats are clustered independently (different embedding dimensions: 512 for humans, 768 for pets)
- Cluster IDs are offset per species to avoid namespace collision

### Complexity
- Phase 1 is O(n²) in theory (brute-force index) but much faster in practice due to FAISS's SIMD-optimized native code
- Phase 2 is O(k²) per component where k << n
- No full n×n distance matrix is stored in memory
- The default threshold of 0.45 cosine distance works well for ArcFace 512-dim embeddings

## Consequences

**Positive:**
- Eliminates chaining via complete-linkage verification
- Handles 61K+ faces without memory exhaustion
- Species are fully isolated — pet clustering cannot corrupt human clusters
- Deterministic results

**Negative:**
- Still O(n²) in the FAISS range search phase (could be improved with `IndexIVFFlat` for approximate search)
- Complete linkage is conservative — same person may split across clusters. Mitigated by auto-assign (ADR-007) and merge UI.
