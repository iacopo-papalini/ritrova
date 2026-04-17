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

### 2a. Caption pre-filter (skip detection on 30% of sources)

Already backlogged and validated. Add `"has_people": bool, "has_animals": bool` to the VLM prompt. Skip face/pet detection when the VLM says no subjects. Zero extra inference cost — just a few more tokens in the existing caption call.

**Expected impact**: 30% fewer detection calls. If detection is 30% of total time, saves ~10% overall. If detection is 15%, saves ~5%.

**Files**: `src/ritrova/describer.py` (prompt), `src/ritrova/analysis_steps.py` (conditional step), `src/ritrova/cli.py` (step ordering in builder)

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
