# ADR-004: Pet detection pipeline

## Status

Accepted

## Context

The user wanted to extend the system to recognize pets (dogs and cats) in photos, motivated by a late border collie named Figaro. The existing InsightFace/ArcFace pipeline is trained exclusively on human faces and cannot detect or embed animal faces.

We needed:

- A detection model that can localize dogs and cats in photos
- An embedding model that produces discriminative features for individual animal identity (or at least appearance similarity)
- Integration with the existing database and clustering pipeline without breaking human face recognition

Options considered:

- **Fine-tuning ArcFace on pet images**: would require a large labeled pet dataset and training infrastructure. Overkill for this use case.
- **CLIP/SigLIP vision encoder**: general-purpose visual features that capture appearance. Not trained for identity, but useful for appearance-based clustering.
- **YOLO for detection + separate embedding model**: YOLO provides excellent object detection with COCO-trained classes that include dogs (class 16) and cats (class 15).

## Decision

We use a two-model pipeline:

1. **YOLO v11m** (`yolo11m.pt`) for detection: identifies and localizes dogs and cats in photos using COCO class IDs (15=cat, 16=dog). Default confidence threshold is 0.5 (lower than the 0.65 used for human faces, because YOLO is well-calibrated for these classes).

2. **SigLIP base** (`google/siglip-base-patch16-224`) for embeddings: the detected animal crop is passed through SigLIP's vision encoder to produce a feature vector. We use the pooler output (or mean-pooled patch tokens as fallback). Embeddings are L2-normalized.

Key design choices:

- The `species` column on the faces table (`human`/`dog`/`cat`) separates the three worlds. Clustering operates within a single species at a time.
- Pet photos use a `__pets` suffix on the file_path in the photos table (e.g., `/path/to/photo.jpg__pets`) to enable idempotent scanning: the same photo can be scanned once for human faces and once for pets without collision.
- SigLIP runs on MPS (Metal Performance Shaders) when available on Apple Silicon, falling back to CPU.

## Consequences

**Positive:**

- YOLO is fast and accurate for dog/cat detection -- very few false positives at the 0.5 confidence threshold
- SigLIP embeddings capture visual appearance well enough to cluster photos of the same pet together, especially for distinctive-looking animals
- The species column cleanly separates human and pet clustering -- no risk of a dog face being merged with a human cluster
- The __pets suffix allows the same photo to be processed by both pipelines independently

**Negative:**

- SigLIP embeddings are not trained for animal identity, so two different dogs of the same breed may cluster together. Clustering thresholds for pets need to be tuned separately from humans.
- The two-model pipeline (YOLO + SigLIP) adds significant dependencies: ultralytics, transformers, torch, sentencepiece
- YOLO model file (`yolo11m.pt`, ~40 MB) is stored in the project directory
- The __pets suffix is an implementation hack -- it works but is not the cleanest schema design. A proper solution would use a separate scans table tracking which pipelines have been run on each photo.
