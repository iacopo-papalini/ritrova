# ADR-010: Analyse Performance Optimization

## Context

The `analyse` pipeline processes sources at **0.4/s** (2.5s per source) on an M1 Max with 48GB. At 32K sources that's ~22 hours for the full archive. The pipeline runs 4 steps sequentially per source: face detection (ArcFace/CoreML), pet detection (YOLO+SigLIP/MPS), VLM captioning (Qwen2.5-VL-7B-4bit/MLX), and translation (NLLB-600M/CPU).

Multi-threading failed — Metal doesn't support concurrent GPU submissions. The bottleneck is sequential model inference on a single GPU.

## Step 1: Instrument to find the bottleneck

Before optimizing, measure per-step timings. Add a `--profile` flag to `analyse` that logs per-step wall time for each source.

### Implementation

Add timing to `AnalysisPipeline.analyse_source()`:

```python
# In AnalysisPipeline.analyse_source(), around the step loop:
for step in self.steps:
    t0 = time.monotonic()
    state = step.analyse(frame, state)
    elapsed = time.monotonic() - t0
    step_times[step.name] = step_times.get(step.name, 0) + elapsed
```

Return step_times alongside the SourceAnalysis (or attach to it). The CLI aggregates and prints a summary at the end:

```
Step timings (avg per source):
  arcface:    0.35s (14%)
  siglip:     0.40s (16%)
  vlm+trans:  1.65s (66%)
  dedup:      0.01s (0%)
  overhead:   0.09s (4%)
```

### Files to modify
- `src/ritrova/analysis.py` — add timing to pipeline loop
- `src/ritrova/cli.py` — add `--profile` flag, print summary

## Step 2: Optimize based on measurements

Likely outcome: VLM captioning dominates (60-80%). Optimization strategies ranked by effort/impact:

### 2a. Caption pre-filter ✅ implemented 2026-04-17

VLM prompt requests JSON with `"has_people": bool, "has_animals": bool`. CaptionStep runs first; `FaceDetectionStep` / `PetDetectionStep` skip when the corresponding flag is false. No CLI flag — implicit whenever captioning is enabled; `--no-caption` disables both the caption and the filter.

**Accuracy-first wiring**:
- Booleans default to True on parse failure or missing fields (fail-open).
- After parsing the JSON, the parser overrides `has_people` / `has_animals` back to True whenever the caption or a tag contains a matching keyword. Qwen2.5-VL-7B-4bit will sometimes describe a cat in the caption yet emit `has_animals: false` — the override catches this.

**Validation (seed 2024, 500 random sources across the archive):**
- Baseline: 737 findings (702 human + 35 animal), 2.788s/source.
- Prefilter: 707 findings (673 human + 34 animal), 2.641s/source (−5.3%).
- Every one of the 30 "missed" findings was a false positive:
  - 29 ArcFace false positives on statues, busts, dolls, and illustrations.
  - 1 YOLO false positive on a video still where the bbox covered almost the whole frame.
- Net effect: prefilter is a **precision win AND a speed win**, not a trade-off. 0 true misses on 500 sources.

**Files touched**: `src/ritrova/describer.py` (JSON prompt, DescribeOutput, keyword override in `_parse_vlm_response`), `src/ritrova/analysis_steps.py` (prefilter_enabled on detection steps, booleans populated from describer output), `src/ritrova/analysis.py` (SourceAnalysis.has_people/has_animals, default True), `src/ritrova/cli.py` (step ordering).

### Running measurements (seed 42, 50 photos from 2024/)

| Phase | s/source | VLM | translator | arcface | siglip | findings |
|---|---|---|---|---|---|---|
| Baseline (off, line prompt) | 2.735 | 2.188 (bundled) | — | 0.378 | 0.123 | 85 |
| 2a prefilter-on | 2.932 | 2.499 (bundled) | — | 0.343 | 0.043 | 83 |
| +A max_tokens 256→128, split translator | 2.754 | **1.989** | **0.341** | 0.335 | 0.043 | 83 |
| +B max_side 1024→896 | 2.611 | **1.815** | 0.372 | 0.335 | 0.043 | 83 |
| D trimmed prompt (74w/100t) | 2.337 | 1.646 | 0.268 | 0.334 | 0.042 | 82 |
| D middle prompt (101w/130t) | 2.772 | 1.925 | 0.424 | 0.332 | 0.046 | 84 |

Translation is 12% of wall time — larger than expected, promotes §2f (translator swap) from last-priority to mid-priority.

Phase B A/B details: on a 20-photo high-complexity sample (seed 2024), captions at 896 were differently worded from 1024 in 17/18 shared sources but none exhibited content regression. At 768 we began to see hallucinations (e.g. IMG_2553 "a man in jeans with his arms around her" not in 1024) and detail loss; 640 regressed further with frequent generic "a group of people". 896 therefore wins on quality-neutrality at the cost of a modest 8.7% VLM speedup (below the original 10% threshold but above the "no regression" criterion that matters more for a personal searchable archive).

Phase D outcome: **rejected as a speed optimisation** — both variants miss or hurt. The trimmed 74-word prompt shortens captions (avg 8.5 vs 16.6 words), losing detail ("red tracksuit on a wooden ladder" → "one person dancing"), for a −10.5% total. The middle 101-word prompt that restored the RIGHT example *added* detail ("red plates, drinks in white cups and bottles" vs "table with food and drinks") but produced longer captions (24.9 words), so generation + translation cost rose, net +6% slower. The old prompt is the best accuracy/speed point; shorter prompts do not monotonically speed things up because they reshape output length. Keep the current prompt as default; the middle variant is a *quality* upgrade option to consider if captioning detail becomes more important than 6% throughput.

### 2b. Smaller/faster VLM model

The current model is Qwen2.5-VL-7B-4bit. Options:
- Qwen2.5-VL-3B-4bit — roughly 2x faster, slightly lower quality
- Test with `--vlm-model` on a sample, compare caption quality

**Expected impact**: if VLM is 66% of time and 3B is 2x faster, total time drops ~33%.

**Files**: none (already configurable via `--vlm-model`)

### 2c. Batch YOLO detection

YOLO supports batch inference — process N images in one forward pass. Currently we call it once per source. Batching 4-8 images could reduce per-image overhead.

**Expected impact**: depends on YOLO's share. If 15% of total and batching gives 2x, saves ~7%.

**Files**: `src/ritrova/pet_detector.py` (batch detect_images method), `src/ritrova/analysis_steps.py` (batch step variant)

### 2d. Skip pet detection entirely when not needed

Most personal photo archives have <1% pet photos. Running YOLO on every source is waste. With the caption pre-filter (2a), this becomes automatic.

**Expected impact**: combined with 2a, eliminates ~15% of inference time on 99% of sources.

### 2e. Image resize before detection

ArcFace already uses `det_size=(640,640)` internally. But we pass the full-resolution image (4368x2912 for a DSLR photo). Resizing to 1280px longest side before passing to detectors would reduce preprocessing time.

**Expected impact**: small — InsightFace resizes internally. But PIL resize before numpy conversion might help for very large images.

**Files**: `src/ritrova/analysis.py` (photo_frames could yield a resized copy for detection)

### 2f. Translation model optimization

NLLB-600M runs on CPU. Options:
- Use a lighter model (Helsinki-NLP opus-mt-en-it, ~300MB)
- Quantize NLLB to int8
- Skip translation entirely and translate at display time in the UI

**Expected impact**: translation is likely <10% of total. Small gains.

## Recommended order

1. **Instrument** (step 1) — 30 min, essential for everything else
2. **Smaller VLM** (2b) — 0 code change, just test with `--vlm-model`
3. **Caption pre-filter** (2a) — 2 hours, validated design, 10-30% savings
4. **Skip pet detection** (2d) — comes free with 2a
5. **Batch YOLO** (2c) — only if YOLO is a significant share after 1-3
6. **Image resize** (2e) — only if profiling shows image loading is significant
7. **Translation** (2f) — last, smallest impact

## Verification

After each optimization, run on a sample and compare:
```bash
ritrova analyse --sample 100 --scan-dir <test_dir> --force --profile
```
Compare rate (images/s) and step breakdown before/after.
