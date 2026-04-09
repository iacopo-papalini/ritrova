# ADR-001: Face detection model choice

## Status

Accepted

## Context

We need a face detection and embedding extraction model for a personal photo collection tool. The model must:

- Detect faces reliably across a wide range of conditions (lighting, angles, group photos, varying resolutions)
- Produce high-quality embeddings for identity clustering
- Run efficiently on a single machine without cloud APIs
- Support Apple Silicon acceleration for the primary development/deployment target

The main options considered were:

- **dlib** (HOG or CNN detector + face_recognition library): widely used in hobby projects, 128-dimensional embeddings, but lower accuracy on challenging photos and no hardware acceleration for Apple Silicon
- **MTCNN + FaceNet**: older pipeline, decent accuracy but complex setup and maintenance
- **InsightFace/ArcFace (buffalo_l)**: state-of-the-art open model suite, 512-dimensional embeddings trained with ArcFace loss, supports multiple execution providers including CoreML

## Decision

We use **InsightFace** with the **buffalo_l** model pack, which bundles a RetinaFace detector and ArcFace embedding extractor.

Key configuration:

- Execution providers: `["CoreMLExecutionProvider", "CPUExecutionProvider"]` -- CoreML is used automatically on Apple Silicon, with CPU fallback on other platforms
- Detection input size: 640x640 pixels
- Embedding dimensionality: 512 float32 values per face
- Embeddings are L2-normalized at extraction time (`face.normed_embedding`)
- Minimum detection confidence defaults to 0.65 for photos and 0.65 for videos

## Consequences

**Positive:**

- 512-dim ArcFace embeddings provide excellent discriminative power for clustering -- sufficient to distinguish family members and similar-looking individuals
- CoreML acceleration on Apple Silicon makes scanning thousands of photos practical (significantly faster than CPU-only)
- The model handles challenging real-world photos well: group shots, varying lighting, partial occlusion
- buffalo_l is a single download (~300 MB) that bundles everything needed

**Negative:**

- The InsightFace library has a heavier dependency footprint than simpler options (onnxruntime, opencv)
- The buffalo_l model is tuned for human faces -- it cannot detect or embed animal faces, which required a separate pipeline (see ADR-004)
- CoreMLExecutionProvider availability depends on platform; the fallback to CPU is significantly slower
- First run requires downloading the model, which needs internet access
