# Reference

> Lookup tables: vocabulary, configuration, path rules and the model inventory. Everything here is a pointer to the authoritative source, not a copy of it.

## Glossary

| Term | Meaning |
|---|---|
| Source | One photo or video file. Immutable; the archive is never written to. |
| Scan | One analysis pass over a source under a named `scan_type`. A zero-finding scan is still recorded. |
| Finding | One detected face or animal: bbox, embedding, species, confidence. Detector output. |
| Cluster | A machine-proposed group of findings believed to be one individual. |
| Subject | A named person or pet. |
| Circle | A user-defined grouping of subjects, such as a side of the family. |
| Singleton | A finding that no cluster claims. |
| Stranger | A finding the user marked as someone they do not want to name. An exclusion, not a deletion. |
| Uncurated | No `finding_assignment` row exists. The default state. |
| Unclustered | No `cluster_findings` row exists. |
| Receipt | What a service returns after a mutation: the undo token plus the sentence the UI shows. |
| `kind` | URL and template vocabulary: `people` / `pets`. |
| subject kind | Database and domain vocabulary: `person` / `pet`. Never used in HTTP code. |
| `species` | Finding vocabulary: `human` / `dog` / `cat` / `other_pet`. `pet` is an umbrella query value the database expands. |

## Configuration

Only two environment variables exist. Both are click options with an `envvar`, so a flag always wins over the environment¹.

| Variable | Flag | Default | Notes |
|---|---|---|---|
| `FACE_DB` | `--db` | `./faces.db` | The example env file sets a different value; see the findings backlog. Several derived paths hang off this file's parent directory. |
| `PHOTOS_DIR` | `--photos-dir` | unset | Archive root. Stored finding paths are relative to it. |

Loaded from an env file by the Makefile, which selects a platform-specific file name on Windows. There is no settings object and no other configuration source.

**Paths derived from the database location** — all siblings of the database file, all regenerable:

- `tmp/frames/` — extracted video frame JPEGs that finding rows point at by relative path.
- `tmp/thumbnails/` — the face-crop cache, created at app startup.
- Analysis run reports — HTML, written next to the database.

## Path conventions

- A stored path beginning with `/` is absolute and used as-is. Everything else resolves against the archive root, except `tmp/...` which resolves against the database directory².
- Source path resolution rejects `..` traversal. Frame-path resolution does not, which the findings backlog records.
- A directory containing a `.fr_exclude` marker file drops itself and its whole subtree from discovery³.
- Photos are `.jpg` / `.jpeg` only. Videos are `.mp4`, `.mov`, `.avi`, `.mkv`.
- Dates come from the path first — directory, then filename — with EXIF as the last fallback. Videos never carry an EXIF date.

## Model inventory

Named here because the choices are load-bearing and ADR-backed; the code is the source of truth for the exact identifiers.

| Role | Model | Output | Install |
|---|---|---|---|
| Face detection and embedding | InsightFace `buffalo_l` (ArcFace), ONNX Runtime, provider order CUDA → CoreML → CPU | 512-dim embedding | detection extra⁴ |
| Pet box detection | YOLO, weights resolved from a working-directory-relative default | dog / cat boxes | detection extra⁵ |
| Pet embedding | SigLIP base patch16-224, device chosen independently of YOLO | 768-dim embedding | detection extra⁵ |
| Captioning | Qwen2.5-VL via MLX, 4-bit | English caption plus tags | captioning extra, Apple Silicon only⁶ |
| Translation | MarianMT en→it, greedy decode | Italian caption and tags | captioning extra⁶ |

**Embedding dimension is the species discriminator** downstream: 512 means human, 768 means pet. Embeddings are stored L2-normalised, so a dot product is cosine similarity everywhere.

## Enumerations

- **Exclusion reasons** — `stranger`, `not_a_face`. Constrained by a CHECK on `finding_assignment`, which also enforces that a row carries either a subject or an exclusion, never both and never neither.
- **Scan types** — a catalog table seeded from a list in the connection module, carrying each type's pipeline description and outputs. Current analysis writes `subjects`, or `subjects+captions` with captioning enabled. A run can be labelled with a custom type to keep an experiment out of the skip logic⁷.
- **Detection strategy** — the `+`-joined step names, stored on the scan row, so the exact step set is recoverable per scan⁸.

## Thresholds

Defaults live in two places and do not agree; the findings backlog records it. The operative values are the per-species ones the CLI passes:

| Operation | Human | Pet |
|---|---|---|
| Clustering similarity | 0.50 | 0.23 (SigLIP space is much denser) |
| Auto-assign minimum | 50% | 50% |
| Auto-merge minimum | 70% | see the clustering chapter — cats embed as a concept, not an individual |

Face detection additionally rejects boxes under 50 px and applies a Laplacian sharpness floor that scales down with face area, so large smooth faces are not read as blur⁹.

## Conventions a contributor must follow

From CLAUDE.md and `docs/design-guide.md`, both authoritative over this page:

- No `<select>` dropdowns for assignment; typeahead with inline create.
- No `confirm()`; use the app dialog. No `location.reload()` except after a successful undo.
- Every write action must be undoable.
- `/{kind}/...` page routes register after all `/api/...` routes.
- No f-strings in logging calls; pass values as arguments.
- Never catch bare `Exception` outside a top-level handler.
- Generated identifiers are UUIDv7.

## References

¹ `src/ritrova/cli.py` — the `cli` group options
² `src/ritrova/db/paths.py` — `resolve_path`, `resolve_finding_image`
³ `src/ritrova/scanner.py` — `EXCLUDE_MARKER`, `find_images`, `find_videos`
⁴ `src/ritrova/detector.py` — `FaceDetector`, `MIN_FACE_SIZE`
⁵ `src/ritrova/pet_detector.py` — `PetDetector`, `_choose_yolo_device`
⁶ `src/ritrova/describer.py` — `DEFAULT_VLM_MODEL`, `DEFAULT_TRANSLATOR_MODEL`
⁷ `src/ritrova/db/connection.py` — `KNOWN_SCAN_TYPES`
⁸ `src/ritrova/analysis.py` — `AnalysisPipeline.strategy_id`
⁹ `src/ritrova/detector.py` — `_BASE_SHARPNESS` and the area scale

---

_Last updated: 2026-10-24 by Kipi_
