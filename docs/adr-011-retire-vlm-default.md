# ADR-011: Retire VLM captioning from the default `analyse` pipeline

**Date:** 2026-04-18
**Status:** Accepted
**Supersedes:** parts of ADR-009 (unified pipeline) and ADR-010 (VLM performance tuning)

## Context

Between early and mid-April 2026, the `analyse` command evolved from two separate passes (`human` for face detection, `pet` for YOLO+SigLIP) into a unified `composite` pipeline that ran a VLM caption step first, then the detection steps, then a translator (en→it via MarianMT). The VLM also provided `has_people` / `has_animals` booleans used as a prefilter to skip detection on scenes with no subjects — validated in ADR-010 §2a on a 500-sample A/B as a precision+throughput win.

On 2026-04-17 the user started a full-archive `--force` re-index with the tuned composite pipeline (~2 s/source after Phase A+B). After two rounds of diagnosis (Maintenance Sleep wakes, then 50/50 stall cadence fixed with `caffeinate -i -w`), the run made it to ~50% of the 32,411-source archive before we stopped it to audit quality.

## Evidence that drove the decision

### 1. Italian translation quality is genuinely bad

MarianMT `opus-mt-tc-big-en-it` was trained on pre-2020 OPUS/EuroParl corpora. It systematically mishandles constructs the VLM produces:

- **Singular *they* → plural *loro*** (most visible): *"A baby on a high chair eats food from their hands"* becomes *"Un bambino su un seggiolone mangia il cibo dalle **loro** mani"*. Correct Italian would be *sue mani* (or simply *le mani*). Italian speakers parse it, but it reads as broken.
- Gender-agreement errors on fixed-gender nouns.
- Literal translations of English idioms.
- Occasional wrong verb forms.

The user's verdict after reading ~20 sampled captions: *"reading these descriptions gives me a bad sensation — the language is wrong."*

A downstream LLM polish sweep (FEAT-26) could fix most of this but only adds another moving part to an already-expensive pipeline.

### 2. Face recall regression (~10%) vs the legacy `human` scanner

On 15,325 sources re-processed by composite where both composite and the legacy `human` scan are present, using a ±10%-bbox match:

|  | In both | Only composite | Only legacy |
|---|---|---|---|
| Humans | 23,204 | **0** | 3,017 (**−11.5%**) |
| Pets | 589 | **0** | 92 (−13.5%) |

Manual eyeball of the 12 largest "missed" faces confirmed these are not all false positives that the VLM prefilter correctly suppressed:

- ~2/12 are legitimately rejected (a pharaoh statue, a dog-on-bench reclassified correctly as a pet).
- ~2/12 are bbox-match artefacts (both pipelines detected the same face but with >10% bbox shift).
- **~8/12 are genuine misses** on clear, large faces (babies, aquarium family shots, recital portraits). The composite pipeline simply failed to reproduce what the legacy pipeline caught.

Root cause not fully traced. Captions on those sources confirm the VLM prefilter was not the cause (captions mention people, `has_people` keyword override forces True). Likely a difference in how the detector is invoked inside the composite pipeline (`detect_image(pil_image)`) vs the legacy path (`detect(path)`). Left as a known-unknown.

### 3. Throughput cost is non-trivial

- **VLM-on:** ~2.0 s/source median (Phase A+B tuned). Full archive ~18 h (assuming no stalls).
- **VLM-off** (subject detection only): ~0.2 s/source in the just-started rescan, 5.7 sources/sec. Full archive ~1 h 30 m.

**~10× throughput gap** for output the user now distrusts.

### 4. Trade-off verdict

> *"1 face lost is not worth 10 statues excluded."* — user, 2026-04-18

For a personal photo-search tool, recall on real human / pet faces is more valuable than precision wins from suppressing a handful of statue / doll false positives.

## Decision

Retire VLM captioning from the default `ritrova analyse` pipeline. Keep the code (Describer, Translator, CaptionStep, TranslationStep) as **opt-in via `--caption`**, gated behind:

- The `[caption]` extra in `pyproject.toml` (`mlx-vlm`, `sentencepiece`, `sacremoses`).
- A runtime check for Apple Silicon (MLX backend is the only supported path; the transformers/CUDA backend is removed entirely).

### New default

| | scan_type | Pipeline | Runtime / source | Output |
|---|---|---|---|---|
| Default | `subjects` | ArcFace + YOLO+SigLIP + dedup | ~0.2 s | findings only |
| Opt-in | `subjects+captions` | + Qwen2.5-VL-7B-4bit (MLX) + MarianMT | ~2.0 s | findings + IT caption + IT tags |

### scan_types catalog

A new documentation-only `scan_types` table records the provenance of every scan_type ever produced: pipeline composition, outputs, introduced-at, retired-at. Logical FK (no hard constraint) so historic `human` / `pet` / `composite` rows keep their lineage even after their pipelines are retired. Populated from `KNOWN_SCAN_TYPES` in `db/connection.py`, synced on every DB open.

## Windows VLM support: removed entirely

The transformers-based Windows VLM backend (Qwen2.5-VL-7B-AWQ via `autoawq`) was measured at ~33 s/source on an RTX 3060 Ti 8 GB — 16× slower than the Mac MLX backend. Combined with:

- `autoawq` having unsatisfiable wheel dependencies on Python 3.14.
- `optimum-quanto` as a fallback being ~3× slower than AWQ at best.
- No Windows user ever being likely to want captioning at those numbers.

The `_load_transformers_backend`, `_describe_image_transformers`, `_extract_transformers_text` helpers and the `DEFAULT_VLM_MODEL_TRANSFORMERS_*` constants are deleted. `_ensure_loaded` now raises a clean `RuntimeError` on non-Apple-Silicon instructing the user to use the face+pet default pipeline.

## Consequences

**Kept:**

- Legacy `human` and `pet` scans (full archive coverage) — authoritative face/pet baseline.
- The entire VLM code path, lazily importable and covered by `--caption`.
- The pipeline / step / persister architecture — untouched. Adding new experimental steps is still a one-class change.
- `describe-eval` and `translate-bench` CLI subcommands for independent VLM experimentation.

**Dropped:**

- 18,485 `composite` scans (and ~29k findings, ~18k descriptions) from the partial re-index — orphan cleanup performed alongside.
- Windows / Linux transformers VLM code path (~100 LOC).
- `mlx-vlm`, `sentencepiece`, `sacremoses`, `autoawq` from core dependencies.
- FEAT-21, FEAT-22, FEAT-23, FEAT-24, FEAT-25, FEAT-26 — all marked withdrawn in `docs/features.md`.

**Revisit when:**

- A better Italian-native VLM ships on Apple Silicon (e.g. Qwen3-VL-IT, Gemma-3-VL-IT).
- An integrated vision+translation model obviates the MarianMT hop entirely.
- The face-recall regression in the composite pipeline is root-caused and fixed (`FaceDetector.detect_image(pil_image)` vs legacy `detect(path)` producing different results on the same source).

If any of those land, the switch is trivial: `uv sync --extra caption && ritrova analyse --caption`. The plumbing is still there.
