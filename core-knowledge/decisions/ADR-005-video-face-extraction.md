# ADR-005: Video face extraction

## Status

Accepted

## Context

Photo collections often contain videos alongside still images. People appearing only in videos would be invisible to the face recognition system if only photos were scanned. We needed a strategy to extract faces from videos and integrate them into the same database and clustering pipeline.

Challenges:

- Videos contain many redundant frames -- processing every frame would be slow and produce thousands of duplicate detections of the same person
- Video frames are often lower quality than photos (motion blur, compression artifacts)
- We need to store a representative image for each detected face (for thumbnail display in the UI)
- The database schema and UI were built around photos; videos need to fit into this model

## Decision

The video scanning pipeline works as follows:

1. **Frame sampling**: extract one frame every 2 seconds (configurable via `--interval`). This balances coverage against processing time. For a 1-minute video at 30fps, we process 30 frames instead of 1800.

2. **Per-video deduplication**: within each video, we track unique identities using cosine similarity (threshold 0.6). When a new face detection is similar to an already-seen face in the same video, we keep only the detection with the highest confidence score. This ensures one entry per person per video.

3. **Frame storage**: for each unique face, the frame where it was best detected is saved as a JPEG file in `tmp/frames/` (next to the database). The filename encodes a hash of the video path plus an index: `vid_{hash}_{index}.jpg`.

4. **Database integration**: each saved frame is stored in the photos table with:
   - `file_path`: path to the extracted JPEG frame
   - `video_path`: path to the original video file (used for idempotent scanning -- if `video_path` already exists in the DB, the video is skipped)
   - Standard width/height from the video resolution

5. **Supported formats**: MP4, MOV, AVI, MKV (via OpenCV's VideoCapture)

## Consequences

**Positive:**

- People who only appear in videos are now included in clustering and can be named
- The deduplication strategy keeps the database compact: one face per person per video rather than hundreds
- Extracted frames are viewable in the UI as regular photos, with face bounding boxes and thumbnails working identically
- The `video_path` column enables clean idempotent scanning

**Negative:**

- Sampling at 1 frame per 2 seconds may miss people who appear only briefly (less than 2 seconds on screen)
- The extracted JPEG frames consume disk space in `tmp/frames/` -- for large video collections this can add up
- Video processing is significantly slower than photo scanning due to frame decoding and the per-frame detection overhead
- The deduplication threshold of 0.6 is hardcoded and may not be optimal for all cases -- it could miss a second detection that is actually a better quality frame of a different angle of the same person
- Videos where no faces are detected still get a placeholder entry in the photos table (`__nofaces_{hash}`) to enable skip-on-rerun behavior
