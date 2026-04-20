"""CLI entry point for ritrova."""

import logging
from pathlib import Path

import click
from dotenv import load_dotenv

from .describer import DEFAULT_TRANSLATOR_MODEL, DEFAULT_VLM_MODEL

load_dotenv()


@click.group()
@click.option(
    "--db",
    default="./faces.db",
    help="Path to SQLite database",
    envvar="FACE_DB",
)
@click.option(
    "--photos-dir",
    default=None,
    help="Root directory for photos (paths stored relative to this)",
    envvar="PHOTOS_DIR",
    type=click.Path(file_okay=False),
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, db: str, photos_dir: str | None, verbose: bool) -> None:
    """Ritrova — find again: face and pet recognition for photo collections."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(name)s %(levelname)s: %(message)s",
    )
    logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db
    ctx.obj["photos_dir"] = photos_dir


def _require_photos_dir(ctx: click.Context) -> str:
    photos_dir: str | None = ctx.obj["photos_dir"]
    if not photos_dir:
        raise click.UsageError("Set --photos-dir or PHOTOS_DIR environment variable")
    return photos_dir


@cli.command()
@click.option("--min-confidence", default=0.65, help="Minimum detection confidence")
@click.pass_context
def scan(ctx: click.Context, min_confidence: float) -> None:
    """Scan photos for human faces. Legacy — delegates to analyse."""
    ctx.invoke(analyse, no_pets=True, no_videos=True, min_face_confidence=min_confidence)


@cli.command()
@click.option("--min-confidence", default=0.7, help="Minimum YOLO detection confidence")
@click.pass_context
def scan_pets(ctx: click.Context, min_confidence: float) -> None:
    """Scan photos for pets. Legacy — delegates to analyse."""
    ctx.invoke(analyse, no_faces=True, no_videos=True, min_pet_confidence=min_confidence)


@cli.command()
@click.option("--no-faces", is_flag=True, help="Skip human face detection")
@click.option("--no-pets", is_flag=True, help="Skip pet detection")
@click.option(
    "--caption",
    is_flag=True,
    default=False,
    help="Opt-in: run VLM captioning + Italian translation after subject detection. "
    "Requires `uv sync --extra caption` and Apple Silicon (MLX backend). Off by default.",
)
@click.option("--no-videos", is_flag=True, help="Skip video files (photos only)")
@click.option("--min-face-confidence", default=0.65, help="Min confidence for face detection")
@click.option("--min-pet-confidence", default=0.7, help="Min confidence for pet detection")
@click.option(
    "--vlm-model",
    default=None,
    help=f"VLM model ID (only with --caption; default: {DEFAULT_VLM_MODEL})",
)
@click.option(
    "--translator-model",
    default=None,
    help=f"Translation model ID (only with --caption; default: {DEFAULT_TRANSLATOR_MODEL})",
)
@click.option(
    "--no-translate", is_flag=True, help="With --caption: keep English output (no translation)"
)
@click.option("--force", is_flag=True, help="Re-analyse already-scanned sources")
@click.option("--dry-run", is_flag=True, help="Run pipeline but don't persist to DB")
@click.option("--sample", default=0, type=int, help="Process only N random sources (0 = all)")
@click.option(
    "--sample-seed",
    default=None,
    type=int,
    help="Seed for --sample selection (reproducible across runs). Default: OS entropy.",
)
@click.option(
    "--interval",
    default=2.0,
    help="Seconds between sampled video frames. Higher = faster but misses "
    "brief subject appearances. Default 2s matches the legacy scan-videos "
    "behaviour — 5s was found to lose faces on short clips.",
)
@click.option(
    "--workers", "-j", default=1, help="Number of parallel workers (default 1; GPU serialised)"
)
@click.option(
    "--scan-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Subdirectory to scan (defaults to --photos-dir). Paths are always stored relative to --photos-dir.",
)
@click.option(
    "--profile",
    is_flag=True,
    help="Measure per-step wall time and print a summary at the end.",
)
@click.option(
    "--scan-type",
    default=None,
    help="Scan type label used to de-dup and to tag stored scans. "
    "Defaults to 'subjects' (or 'subjects+captions' with --caption). "
    "Override to let experiment runs coexist alongside the production data.",
)
@click.option(
    "--max-tokens",
    default=128,
    type=int,
    help="Upper bound on VLM generation length (tokens). Lower = faster; "
    "default 128 fits observed 20-24 word captions with ~2x headroom.",
)
@click.option(
    "--vlm-max-side",
    default=896,
    type=int,
    help="Longest side (pixels) the image is resized to before the VLM. "
    "Smaller = faster, but may lose fine detail on complex scenes.",
)
@click.pass_context
def analyse(
    ctx: click.Context,
    no_faces: bool,
    no_pets: bool,
    caption: bool,
    no_videos: bool,
    min_face_confidence: float,
    min_pet_confidence: float,
    vlm_model: str | None,
    translator_model: str | None,
    no_translate: bool,
    force: bool,
    dry_run: bool,
    sample: int,
    sample_seed: int | None,
    interval: float,
    workers: int,
    scan_dir: str | None,
    profile: bool,
    scan_type: str | None,
    max_tokens: int,
    vlm_max_side: int,
) -> None:
    """Unified source analysis: detect faces, pets, and generate captions in one pass."""
    import random
    import warnings

    # Suppress noisy third-party loggers during model loading
    for noisy in (
        "insightface",
        "onnxruntime",
        "ultralytics",
        "transformers",
        "huggingface_hub",
        "mlx_vlm",
    ):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*estimate.*deprecated.*SimilarityTransform.*")

    from .analysis import (
        AnalysisPersister,
        AnalysisPipelineBuilder,
        SourceAnalysis,
        photo_frames,
        video_frames,
    )
    from .analysis_steps import (
        CaptionStep,
        DeduplicationStep,
        FaceDetectionStep,
        PetDetectionStep,
        TranslationStep,
    )
    from .db import FaceDB
    from .scanner import find_images, find_videos, get_exif_date, get_exif_gps

    photos_dir = _require_photos_dir(ctx)
    db = FaceDB(ctx.obj["db_path"], base_dir=photos_dir)
    # scan_dir: where to discover files. base_dir (photos_dir): prefix to strip for relative paths.
    discovery_dir = scan_dir or photos_dir
    frames_dir = Path(ctx.obj["db_path"]).parent / "tmp" / "frames"

    builder = AnalysisPipelineBuilder()

    # Default scan_type derives from the pipeline composition. See ADR-011 +
    # the `scan_types` catalog table for the taxonomy.
    effective_scan_type = scan_type or ("subjects+captions" if caption else "subjects")

    # --caption (opt-in) adds VLM captioning + translation steps. Face/pet
    # detection always runs unconditionally — the VLM subject-prefilter was
    # removed after it caused real face-recall losses on clear portraits.
    # Without --caption, the pipeline is just face + pet + dedup, unified
    # under scan_type `subjects`. See ADR-011 for the full rationale.
    translator_obj = None
    if caption:
        from .describer import _prefers_mlx_backend

        if not _prefers_mlx_backend():
            raise click.ClickException(
                "--caption requires Apple Silicon (MLX backend). "
                "The transformers/CUDA path was retired in ADR-011; "
                "see docs/adr-011-retire-vlm-default.md."
            )
        try:
            from .describer import Describer, Translator
        except ImportError as exc:
            raise click.ClickException(
                "--caption requires optional dependencies. Install with:\n  uv sync --extra caption"
            ) from exc
        vlm_id = vlm_model or DEFAULT_VLM_MODEL
        trans_id = translator_model or DEFAULT_TRANSLATOR_MODEL
        print(f"Loading VLM: {vlm_id}")
        describer = Describer(model_id=vlm_id, max_tokens=max_tokens, max_side=vlm_max_side)
        translator_obj = None if no_translate else Translator(model_id=trans_id)
        if translator_obj:
            print(f"Loading translator: {trans_id}")
        builder.add_step(CaptionStep(describer))

    if not no_faces:
        from .detector import FaceDetector

        print("Loading face detection model...")
        builder.add_step(FaceDetectionStep(FaceDetector(), min_face_confidence))

    if not no_pets:
        from .pet_detector import PetDetector

        print("Loading pet detection models...")
        builder.add_step(PetDetectionStep(PetDetector(), min_pet_confidence))

    # Translation is CPU-only and parse of VLM output is English; we run it
    # after detection so the profile records it as a distinct step.
    if translator_obj is not None:
        builder.add_step(TranslationStep(translator_obj))

    builder.add_step(DeduplicationStep())
    pipeline = builder.build()

    # Discover sources: photos + optionally videos
    candidates: list[tuple[Path, str]] = []  # (path, source_type)
    for img_path in find_images(Path(discovery_dir)):
        stored = db.to_relative(str(img_path.resolve()))
        if not force and db.is_scanned(stored, effective_scan_type):
            continue
        candidates.append((img_path, "photo"))

    if not no_videos:
        for vid_path in find_videos(Path(discovery_dir)):
            stored = db.to_relative(str(vid_path.resolve()))
            if not force and db.is_scanned(stored, effective_scan_type):
                continue
            candidates.append((vid_path, "video"))

    if sample > 0 and len(candidates) > sample:
        # Sort first so --sample-seed is reproducible regardless of filesystem order.
        candidates.sort(key=lambda pair: str(pair[0]))
        rng = random.Random(sample_seed) if sample_seed is not None else random
        candidates = rng.sample(candidates, sample)

    print(f"Database: {ctx.obj['db_path']}")
    print(f"scan_type: {effective_scan_type}")
    print(f"Pipeline:  {pipeline.strategy_id}")
    n_photos = sum(1 for _, t in candidates if t == "photo")
    n_videos = sum(1 for _, t in candidates if t == "video")
    print(
        f"Sources to analyse: {len(candidates)} ({n_photos} photos, {n_videos} videos)"
        f" — {workers} workers"
    )

    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    persister = AnalysisPersister(db, frames_dir=frames_dir) if not dry_run else None
    strategy_id = pipeline.strategy_id

    # Shared counters protected by a lock
    lock = threading.Lock()
    counters = {"processed": 0, "errors": 0, "findings": 0, "done": 0}

    # Profile buckets split by source_type. Dataclass (not dict) so mypy can
    # track the per-field types cleanly.
    from dataclasses import dataclass
    from dataclasses import field as _dc_field

    @dataclass
    class _ProfileBucket:
        steps: dict[str, float] = _dc_field(default_factory=dict)
        exif: float = 0.0
        io: float = 0.0
        persist: float = 0.0
        total: float = 0.0
        n: int = 0

    profile_buckets: dict[str, _ProfileBucket] = {
        "photo": _ProfileBucket(),
        "video": _ProfileBucket(),
    }
    t_start = time.monotonic()
    total = len(candidates)

    def _process_one(source_path: Path, source_type: str) -> None:
        stored = db.to_relative(str(source_path.resolve()))
        if source_type == "video":
            frame_iter = video_frames(source_path, interval_sec=interval)
        else:
            frame_iter = photo_frames(source_path)

        per_source_times: dict[str, float] | None = {} if profile else None
        t_source_start = time.monotonic() if profile else 0.0
        t_exif = 0.0
        t_pipeline = 0.0
        t_persist = 0.0

        try:
            initial = SourceAnalysis(source_path=stored, source_type=source_type)
            if source_type == "photo":
                if profile:
                    t0 = time.monotonic()
                initial.taken_at = get_exif_date(source_path)
                gps = get_exif_gps(source_path)
                if gps:
                    initial.latitude, initial.longitude = gps
                if profile:
                    t_exif = time.monotonic() - t0

            if profile:
                t0 = time.monotonic()
            result = pipeline.analyse_source(
                source_path,
                source_type=source_type,
                frames=frame_iter,
                initial_state=initial,
                step_times=per_source_times,
            )
            if profile:
                t_pipeline = time.monotonic() - t0
        except OSError:
            with lock:
                counters["errors"] += 1
                counters["done"] += 1
            return

        n_findings = len(result.findings)

        # Persistence is serialised by the DB lock
        if persister is not None and (n_findings > 0 or result.caption):
            if profile:
                t0 = time.monotonic()
            if force:
                existing = db.conn.execute(
                    "SELECT sc.id FROM scans sc JOIN sources s ON s.id = sc.source_id "
                    "WHERE s.file_path = ? AND sc.scan_type = ?",
                    (stored, effective_scan_type),
                ).fetchone()
                if existing:
                    db.delete_scan(existing[0])
            persister.persist(result, strategy_id=strategy_id, scan_type=effective_scan_type)
            if profile:
                t_persist = time.monotonic() - t0

        with lock:
            counters["processed"] += 1
            counters["findings"] += n_findings
            counters["done"] += 1
            if per_source_times is not None:
                bucket = profile_buckets[source_type]
                step_sum = 0.0
                for name, elapsed in per_source_times.items():
                    bucket.steps[name] = bucket.steps.get(name, 0.0) + elapsed
                    step_sum += elapsed
                # I/O time inside the pipeline = pipeline wall minus step work.
                # Captures PIL.open + EXIF-transpose + RGB convert for photos,
                # and cv2 VideoCapture frame decoding for videos.
                bucket.io += max(t_pipeline - step_sum, 0.0)
                bucket.exif += t_exif
                bucket.persist += t_persist
                bucket.total += time.monotonic() - t_source_start
                bucket.n += 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_one, src, stype): i for i, (src, stype) in enumerate(candidates, 1)
        }
        for future in as_completed(futures):
            future.result()  # propagate unexpected exceptions
            with lock:
                done = counters["done"]
            if done % 10 == 0 or done == total:
                elapsed = time.monotonic() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (total - done) / rate if rate > 0 else 0
                eta_min, eta_sec = divmod(int(remaining), 60)
                eta_h, eta_min = divmod(eta_min, 60)
                eta = f"{eta_h}h{eta_min:02d}m" if eta_h else f"{eta_min}m{eta_sec:02d}s"
                print(
                    f"\r  [{done}/{total}] processed={counters['processed']} "
                    f"findings={counters['findings']} errors={counters['errors']} "
                    f"{rate:.1f}/s ETA {eta}   ",
                    end="",
                    flush=True,
                )

    print()
    mode = " (dry run)" if dry_run else ""
    print(
        f"Done!{mode} processed={counters['processed']}  "
        f"findings={counters['findings']}  errors={counters['errors']}"
    )

    if profile and counters["processed"] > 0:
        for bucket_name, bucket in profile_buckets.items():
            if bucket.n == 0:
                continue

            # Name rows in display order: exif → frame I/O → each step → persist
            rows: list[tuple[str, float]] = []
            if bucket.exif > 0:
                rows.append(("exif", bucket.exif))
            if bucket.io > 0:
                rows.append(("frame I/O", bucket.io))
            # Steps sorted by total descending so bottlenecks are obvious.
            rows.extend(sorted(bucket.steps.items(), key=lambda kv: kv[1], reverse=True))
            if bucket.persist > 0:
                rows.append(("persist", bucket.persist))
            accounted = sum(v for _, v in rows)
            overhead = max(bucket.total - accounted, 0.0)
            if overhead > 0:
                rows.append(("overhead", overhead))

            denom = bucket.total if bucket.total > 0 else accounted if accounted > 0 else 1.0
            name_w = max(len(k) for k, _ in rows)
            print()
            print(f"=== {bucket_name.upper()}S ({bucket.n} sources, avg per source) ===")
            for name, total_t in rows:
                pct = 100 * total_t / denom
                print(f"  {name:<{name_w}}  {total_t / bucket.n:7.3f}s  ({pct:5.1f}%)")
            print(f"  {'TOTAL':<{name_w}}  {bucket.total / bucket.n:7.3f}s")

    db.close()


@cli.command()
@click.option(
    "--model",
    default=None,
    help=f"HuggingFace model ID for the VLM (default: {DEFAULT_VLM_MODEL})",
)
@click.option(
    "--translator",
    default=None,
    help=f"HuggingFace model ID for en→it translation (default: {DEFAULT_TRANSLATOR_MODEL})",
)
@click.option("--no-translate", is_flag=True, help="Skip translation — store English output")
@click.option("--sample", default=0, type=int, help="Process only N random sources (0 = all)")
@click.option("--force", is_flag=True, help="Re-describe sources that already have descriptions")
@click.option(
    "--source-id",
    "source_ids_arg",
    multiple=True,
    type=int,
    help="Specific source IDs (repeatable)",
)
@click.pass_context
def describe(
    ctx: click.Context,
    model: str | None,
    translator: str | None,
    no_translate: bool,
    sample: int,
    force: bool,
    source_ids_arg: tuple[int, ...],
) -> None:
    """Generate scene descriptions and tags for photos via VLM + translation."""
    import random

    from .db import FaceDB
    from .describer import (
        DEFAULT_TRANSLATOR_MODEL,
        DEFAULT_VLM_MODEL,
        Describer,
        Translator,
        describe_sources,
    )

    model = model or DEFAULT_VLM_MODEL
    translator = translator or DEFAULT_TRANSLATOR_MODEL

    _require_photos_dir(ctx)
    db = FaceDB(ctx.obj["db_path"], base_dir=ctx.obj["photos_dir"])

    if source_ids_arg:
        source_ids = list(source_ids_arg)
    elif force:
        source_ids = db.get_all_source_ids()
    else:
        source_ids = db.get_undescribed_source_ids()

    if not source_ids:
        print("All sources already have descriptions. Use --force to re-describe.")
        db.close()
        return

    if sample > 0 and not source_ids_arg:
        source_ids = random.sample(source_ids, min(sample, len(source_ids)))

    print(f"Database: {ctx.obj['db_path']}")
    print(f"VLM: {model}")
    if no_translate:
        print("Translation: disabled (English output)")
    else:
        print(f"Translator: {translator}")
    print(f"Sources to describe: {len(source_ids)}")
    print("Loading models (first run downloads them)...")

    describer = Describer(model_id=model)
    translator_obj = None if no_translate else Translator(model_id=translator)
    result = describe_sources(db, source_ids, describer, translator_obj, force=force)

    print(f"Done! described={result.described}  errors={result.errors}  tags={result.total_tags}")
    db.close()


@cli.command("describe-eval")
@click.option(
    "--model",
    default=None,
    help=f"HuggingFace model ID for the VLM (default: {DEFAULT_VLM_MODEL})",
)
@click.option(
    "--translator",
    default=None,
    help=f"HuggingFace model ID for en→it translation (default: {DEFAULT_TRANSLATOR_MODEL})",
)
@click.option("--count", default=100, type=int, help="Number of random photos to evaluate")
@click.option("--output", "-o", default="describe_eval.csv", help="Output CSV path")
@click.option(
    "--max-tokens",
    default=128,
    type=int,
    help="VLM generation cap (tokens). Must match across A/B runs.",
)
@click.option(
    "--vlm-max-side",
    default=896,
    type=int,
    help="Longest image side (pixels) fed to the VLM. Varied between A/B runs.",
)
@click.option(
    "--high-complexity",
    is_flag=True,
    help="Bias the random sample to sources with ≥4 existing findings — the "
    "busy, worst-case photos used to stress-test image resize / prompt trims.",
)
@click.option(
    "--sample-seed",
    default=None,
    type=int,
    help="Seed for the random sample (reproducible across A/B runs).",
)
@click.option(
    "--no-translate",
    is_flag=True,
    help="Skip translation — write only source_id / path / en_caption / en_tags. "
    "Use to build a corpus for the translate-bench rig.",
)
@click.pass_context
def describe_eval(
    ctx: click.Context,
    model: str | None,
    translator: str | None,
    count: int,
    output: str,
    max_tokens: int,
    vlm_max_side: int,
    high_complexity: bool,
    sample_seed: int | None,
    no_translate: bool,
) -> None:
    """Generate a CSV of sample descriptions for human review.

    Picks random photos, runs VLM + translation, writes both English and
    Italian output to a CSV with a blank 'review' column for annotation.
    Does NOT store results in the DB.
    """
    import csv
    import random

    from .db import FaceDB
    from .describer import (
        DEFAULT_TRANSLATOR_MODEL,
        DEFAULT_VLM_MODEL,
        Describer,
        Translator,
    )

    model = model or DEFAULT_VLM_MODEL
    translator = translator or DEFAULT_TRANSLATOR_MODEL

    _require_photos_dir(ctx)
    db = FaceDB(ctx.obj["db_path"], base_dir=ctx.obj["photos_dir"])

    if high_complexity:
        rows = db.conn.execute(
            """
            SELECT s.id
            FROM sources s
            JOIN scans sc ON sc.source_id = s.id
            LEFT JOIN findings f ON f.scan_id = sc.id
            JOIN descriptions d ON d.scan_id = sc.id
            WHERE sc.scan_type LIKE 'subjects%' AND d.caption IS NOT NULL
            GROUP BY s.id
            HAVING COUNT(f.id) >= 4
            """
        ).fetchall()
        all_ids = [r[0] for r in rows]
        if not all_ids:
            raise click.UsageError(
                "--high-complexity found no sources with ≥4 findings; "
                "run `ritrova analyse` first or drop the flag."
            )
    else:
        all_ids = db.get_all_source_ids()

    all_ids.sort()  # stable order so --sample-seed is reproducible.
    rng = random.Random(sample_seed) if sample_seed is not None else random
    source_ids = rng.sample(all_ids, min(count, len(all_ids)))

    print(f"VLM: {model}  max_tokens={max_tokens}  max_side={vlm_max_side}")
    print(f"Translator: {translator}")
    selector = "high-complexity" if high_complexity else "all"
    print(f"Evaluating {len(source_ids)} random photos ({selector}) → {output}")
    print("Loading models...")

    describer = Describer(model_id=model, max_tokens=max_tokens, max_side=vlm_max_side)
    translator_obj = None if no_translate else Translator(model_id=translator)

    columns = ["source_id", "path", "en_caption", "en_tags"]
    if not no_translate:
        columns.extend(["it_caption", "it_tags", "review"])

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        errors = 0
        for i, source_id in enumerate(source_ids, 1):
            source = db.get_source(source_id)
            if not source or source.type != "photo":
                continue
            resolved = db.resolve_path(source.file_path)
            if not resolved.exists():
                errors += 1
                continue

            vlm_output = describer.describe(resolved)
            if not vlm_output.caption:
                errors += 1
                continue

            en_caption, en_tags = vlm_output.caption, vlm_output.tags
            row = [source_id, str(resolved), en_caption, ", ".join(sorted(en_tags))]
            if translator_obj is not None:
                it_caption, it_tags = translator_obj.translate(en_caption, en_tags)
                row.extend([it_caption, ", ".join(sorted(it_tags)), ""])
            writer.writerow(row)

            if i % 10 == 0 or i == len(source_ids):
                print(f"\r  [{i}/{len(source_ids)}] errors={errors}", end="", flush=True)

    print(f"\nDone! Wrote {len(source_ids) - errors} rows to {output} ({errors} errors)")
    db.close()


@cli.command("translate-bench")
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="CSV produced by `describe-eval --no-translate` (columns: "
    "source_id, path, en_caption, en_tags).",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output CSV with en + it columns. Defaults to <input>.it.csv.",
)
@click.option(
    "--translator",
    default=None,
    help=f"HuggingFace model ID (default: {DEFAULT_TRANSLATOR_MODEL}).",
)
@click.option(
    "--num-beams",
    default=1,
    type=int,
    help="Beam-search width. 1 = greedy (current default).",
)
@click.option(
    "--max-new-tokens",
    default=128,
    type=int,
    help="Max tokens per translation (safety cap; EOS usually triggers earlier).",
)
@click.option(
    "--device",
    default="cpu",
    type=click.Choice(["cpu", "cuda", "mps"]),
    help="Torch device for translation benchmarking.",
)
@click.pass_context
def translate_bench(
    ctx: click.Context,
    input_path: str,
    output: str | None,
    translator: str | None,
    num_beams: int,
    max_new_tokens: int,
    device: str,
) -> None:
    """Replay-only translation benchmark.

    Reads pre-recorded English captions from ``--input`` and runs ONLY the
    translator with the supplied config. No VLM, no face/pet detection — the
    idea is to iterate quickly on translator variants without re-paying the
    ~20 min captioning cost for each A/B.
    """
    import csv
    import time

    from .describer import DEFAULT_TRANSLATOR_MODEL, Translator

    translator_id = translator or DEFAULT_TRANSLATOR_MODEL
    out_path = output or f"{input_path}.it.csv"

    with open(input_path) as f:
        rows = list(csv.DictReader(f))

    print(
        f"Translator: {translator_id}  num_beams={num_beams}  "
        f"max_new_tokens={max_new_tokens}  device={device}"
    )
    print(f"Input:  {input_path} ({len(rows)} rows)")
    print(f"Output: {out_path}")
    print("Loading model...")

    t_obj = Translator(model_id=translator_id)
    t_obj._ensure_loaded()  # noqa: SLF001 — explicit warm-up so load time isn't counted.
    if device != "cpu":
        # Move the already-loaded seq2seq onto the requested accelerator.
        t_obj._translation_model = t_obj._translation_model.to(device)  # noqa: SLF001
    t_obj._device = device  # noqa: SLF001 — read inside _translate_text

    # Patch the generation kwargs to the bench values. _translate_text's
    # current hard-coded kwargs are the shipped defaults; for bench we
    # override from the CLI.
    t_obj._bench_gen_kwargs = {  # noqa: SLF001
        "num_beams": num_beams,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }

    with open(out_path, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["source_id", "path", "en_caption", "en_tags", "it_caption", "it_tags"])

        t_start = time.monotonic()
        per_row: list[float] = []
        for i, row in enumerate(rows, 1):
            en_caption = row["en_caption"]
            en_tags = {t.strip() for t in row["en_tags"].split(",") if t.strip()}
            t0 = time.monotonic()
            it_caption, it_tags = t_obj.translate(en_caption, en_tags)
            per_row.append(time.monotonic() - t0)
            writer.writerow(
                [
                    row["source_id"],
                    row["path"],
                    en_caption,
                    row["en_tags"],
                    it_caption,
                    ", ".join(sorted(it_tags)),
                ]
            )
            if i % 50 == 0 or i == len(rows):
                elapsed = time.monotonic() - t_start
                rate = i / elapsed if elapsed > 0 else 0
                print(
                    f"\r  [{i}/{len(rows)}] {rate:.1f}/s  avg {elapsed / i * 1000:.0f}ms/row",
                    end="",
                    flush=True,
                )

    total = time.monotonic() - t_start
    per_row.sort()
    p50 = per_row[len(per_row) // 2] * 1000
    p95 = per_row[int(len(per_row) * 0.95)] * 1000
    print()
    print(
        f"Done! {len(rows)} rows in {total:.1f}s  "
        f"avg {total / len(rows) * 1000:.0f}ms  p50 {p50:.0f}ms  p95 {p95:.0f}ms"
    )


@cli.command()
@click.option("--min-confidence", default=0.65, help="Minimum detection confidence")
@click.option("--interval", default=2.0, help="Seconds between sampled frames")
@click.pass_context
def scan_videos(ctx: click.Context, min_confidence: float, interval: float) -> None:
    """Scan videos for human faces. Legacy — delegates to analyse."""
    ctx.invoke(
        analyse,
        no_pets=True,
        no_caption=True,
        min_face_confidence=min_confidence,
        interval=interval,
    )


SPECIES_THRESHOLDS = {
    "human": 0.50,  # ArcFace 512-dim. Bumped from 0.45 — no intruders observed at 0.45
    "dog": 0.23,  # SigLIP 768-dim: much denser, needs tighter threshold. Bumped from 0.20
    "cat": 0.23,
}

# Per-kind auto-merge thresholds (%). SigLIP embeds cat-as-a-concept, not individuals:
# any two cats' centroids sit at 0.92-0.98 cosine sim, so any threshold <100 cascades
# into a single mega-cluster. Disable pet auto-merge entirely — rely on curation + the
# `suggest_merges` UI for pets.
AUTO_MERGE_THRESHOLDS = {"person": 70.0, "pet": 101.0}


@cli.command()
@click.option(
    "--threshold",
    default=None,
    type=float,
    help="Override cosine distance threshold (default: per-species)",
)
@click.option("--min-size", default=2, help="Minimum faces per cluster")
@click.option(
    "--auto-merge-threshold",
    default=None,
    type=float,
    help=(
        "Override per-kind auto-merge similarity %. "
        "Default: 70 for people, disabled for pets (SigLIP centroids too close). "
        "Set to 100 (or higher) to disable for all kinds."
    ),
)
@click.pass_context
def cluster(
    ctx: click.Context,
    threshold: float | None,
    min_size: int,
    auto_merge_threshold: float | None,
) -> None:
    """Cluster all detected faces by embedding similarity (humans + pets)."""
    from .cluster import auto_merge_clusters, cluster_faces
    from .db import FaceDB

    db = FaceDB(ctx.obj["db_path"], base_dir=ctx.obj["photos_dir"])

    for species, default_thresh in SPECIES_THRESHOLDS.items():
        t = threshold if threshold is not None else default_thresh
        print(f"\n── {species} (threshold={t}) ──")
        result = cluster_faces(db, threshold=t, min_size=min_size, species=species)
        print(f"  Total faces:      {result['total_faces']}")
        print(f"  Clusters formed:  {result['clusters']}")
        print(f"  Noise (outliers): {result['noise']}")
        if result.get("largest_cluster"):
            print(f"  Largest cluster:  {result['largest_cluster']} faces")

    # Per-kind (person, pet) rather than per-species: pet auto-merge spans
    # dog + cat because they share the SigLIP embedding space.
    for kind, default_merge in AUTO_MERGE_THRESHOLDS.items():
        t = auto_merge_threshold if auto_merge_threshold is not None else default_merge
        if t >= 100.0:
            print(f"\n── auto-merge {kind} skipped (threshold {t:.0f}%) ──")
            continue
        print(f"\n── auto-merge {kind} (≥ {t:.0f}%) ──")
        total_merged = 0
        total_moved = 0
        iteration = 0
        max_iterations = 20  # safety cap — convergence is typically 1-3 passes
        while iteration < max_iterations:
            result = auto_merge_clusters(db, min_similarity=t / 100, kind=kind)
            iteration += 1
            print(
                f"  pass {iteration}: merged {result['merged']} pairs, "
                f"moved {result['faces_moved']} faces, "
                f"{result['remaining_clusters']} clusters remain"
            )
            if result["merged"] == 0:
                break
            total_merged += result["merged"]
            total_moved += result["faces_moved"]
        print(f"  total: {total_merged} merges, {total_moved} faces moved")

    db.close()


@cli.command()
@click.option("--min-similarity", default=50.0, help="Minimum centroid similarity %")
@click.option("--kind", default="person", help="Subject kind: person or pet")
@click.pass_context
def auto_assign(ctx: click.Context, min_similarity: float, kind: str) -> None:
    """Bulk-assign unnamed clusters to existing named subjects."""
    from .cluster import auto_assign as _auto_assign
    from .db import FaceDB

    db = FaceDB(ctx.obj["db_path"], base_dir=ctx.obj["photos_dir"])
    result = _auto_assign(db, min_similarity=min_similarity / 100, kind=kind)

    print(
        f"\nAssigned {result['assigned_clusters']} clusters "
        f"({result['assigned_faces']} faces), "
        f"{result['unmatched']} unmatched"
    )
    db.close()


@cli.command()
@click.option("--min-similarity", default=70.0, help="Minimum centroid similarity %")
@click.option("--kind", default="person", help="Subject kind: person or pet")
@click.pass_context
def auto_merge(ctx: click.Context, min_similarity: float, kind: str) -> None:
    """Auto-merge unnamed clusters whose centroids are highly similar."""
    from .cluster import auto_merge_clusters
    from .db import FaceDB

    db = FaceDB(ctx.obj["db_path"], base_dir=ctx.obj["photos_dir"])
    result = auto_merge_clusters(db, min_similarity=min_similarity / 100, kind=kind)

    print(
        f"\nMerged {result['merged']} cluster pairs "
        f"({result['faces_moved']} faces moved), "
        f"{result['remaining_clusters']} clusters remaining"
    )
    db.close()


@cli.command()
@click.option("--min-size", default=50, help="Minimum face width/height in pixels")
@click.option("--min-sharpness", default=30.0, help="Minimum Laplacian variance (focus blur)")
@click.option("--min-edges", default=2.0, help="Minimum Canny edge density % (motion blur)")
@click.option("--dry-run", is_flag=True, help="Show what would be dismissed without acting")
@click.pass_context
def cleanup(
    ctx: click.Context, min_size: int, min_sharpness: float, min_edges: float, dry_run: bool
) -> None:
    """Dismiss tiny and blurry faces from the database."""
    import cv2
    from PIL import Image, ImageFile, ImageOps

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    from .db import FaceDB

    db = FaceDB(ctx.obj["db_path"], base_dir=ctx.obj["photos_dir"])

    # Find uncurated faces (no assignment row at all, excluded included).
    rows = db.query(
        "SELECT f.id, f.source_id, f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h "
        "FROM findings f "
        "LEFT JOIN finding_assignment fa ON fa.finding_id = f.id "
        "WHERE fa.finding_id IS NULL"
    )
    total = len(rows)
    print(f"Checking {total} unassigned faces...")

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import numpy as np

    tiny_ids = []
    check_rows = []
    for r in rows:
        fid, pid, bx, by, bw, bh = r[0], r[1], r[2], r[3], r[4], r[5]
        if bw < min_size or bh < min_size:
            tiny_ids.append(fid)
        else:
            check_rows.append((fid, pid, bx, by, bw, bh))

    print(f"  Tiny (<{min_size}px): {len(tiny_ids)} — instant dismiss")
    print(f"  Checking sharpness on {len(check_rows)} faces...")

    bad_quality_ids = []
    lock = threading.Lock()
    done = [0]
    focus_blur = [0]
    motion_blur = [0]

    def check_quality(row: tuple[int, ...]) -> tuple[int | None, str | None]:
        fid, pid, bx, by, bw, bh = row
        photo = db.get_source(pid)
        if not photo:
            return None, None
        resolved = db.resolve_path(photo.file_path)
        if not resolved.exists():
            return None, None
        try:
            raw_img = Image.open(resolved)
            oriented = ImageOps.exif_transpose(raw_img)
            crop = oriented.crop((bx, by, bx + bw, by + bh))
            gray = np.array(crop.convert("L"))
            lap = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lap < min_sharpness:
                return fid, "focus"
            edge_pct = (cv2.Canny(gray, 50, 150) > 0).mean() * 100
            if edge_pct < min_edges:
                return fid, "motion"
            return None, None
        except OSError:
            return None, None

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(check_quality, r): r for r in check_rows}
        for future in as_completed(futures):
            fid, reason = future.result()
            if fid is not None:
                bad_quality_ids.append(fid)
                if reason == "focus":
                    focus_blur[0] += 1
                else:
                    motion_blur[0] += 1
            with lock:
                done[0] += 1
                if done[0] % 500 == 0:
                    print(
                        f"\r  [{done[0]}/{len(check_rows)}] out_of_focus={focus_blur[0]} featureless={motion_blur[0]}",
                        end="",
                        flush=True,
                    )

    to_dismiss = tiny_ids + bad_quality_ids
    print(f"\n  Tiny (<{min_size}px): {len(tiny_ids)}")
    print(f"  Out of focus (laplacian <{min_sharpness}): {focus_blur[0]}")
    print(f"  Featureless (edges <{min_edges}%): {motion_blur[0]}")
    print(f"  Total to dismiss: {len(to_dismiss)}")

    if dry_run:
        print("Dry run — no changes made.")
    elif to_dismiss:
        db.dismiss_findings(to_dismiss)
        print(f"Dismissed {len(to_dismiss)} faces.")

    db.close()


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8787, type=int, help="Port to listen on")
@click.pass_context
def serve(ctx: click.Context, host: str, port: int) -> None:
    """Start the web UI for browsing and naming faces."""
    import uvicorn

    from .app import create_app

    app = create_app(ctx.obj["db_path"], photos_dir=ctx.obj["photos_dir"])
    print(f"Ritrova → http://localhost:{port}")
    uvicorn.run(app, host=host, port=port)


@cli.command()
@click.option("--output", "-o", default="-", help="Output file (- for stdout)")
@click.pass_context
def export(ctx: click.Context, output: str) -> None:
    """Export database as JSON."""
    from .db import FaceDB

    db = FaceDB(ctx.obj["db_path"], base_dir=ctx.obj["photos_dir"])
    data = db.export_json()

    if output == "-":
        print(data)
    else:
        Path(output).write_text(data)
        print(f"Exported to {output}")
    db.close()


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show database statistics."""
    from .db import FaceDB

    db = FaceDB(ctx.obj["db_path"], base_dir=ctx.obj["photos_dir"])
    s = db.get_stats()

    print(f"Sources scanned:   {s['total_sources']}")
    print(f"Findings detected: {s['total_findings']}")
    print(f"Subjects named:    {s['total_subjects']}")
    print(f"Named findings:    {s['named_findings']}")
    print(f"Unnamed clusters:  {s['unnamed_clusters']}")
    print(f"Unclustered:       {s['unclustered_findings']}")
    db.close()


@cli.group()
def scans() -> None:
    """Inspect and prune scan records (and the findings they own)."""


@scans.command("list")
@click.option("--source-pattern", default=None, help="GLOB pattern on source path (e.g. '2024/*')")
@click.pass_context
def scans_list(ctx: click.Context, source_pattern: str | None) -> None:
    """List scans, optionally filtered by a source path pattern."""
    from .db import FaceDB

    db = FaceDB(ctx.obj["db_path"], base_dir=ctx.obj["photos_dir"])
    rows = db.find_scans(source_pattern=source_pattern)
    if not rows:
        click.echo("(no scans match)")
    else:
        click.echo(f"{'id':>6}  {'type':6s}  {'findings':>8}  {'scanned_at':24s}  source")
        for r in rows:
            click.echo(
                f"{r['id']:>6}  {r['scan_type']:6s}  {r['finding_count']:>8}  "
                f"{r['scanned_at']:24s}  {r['source_path']}"
            )
        click.echo(f"\n{len(rows)} scan(s)")
    db.close()


@scans.command("prune")
@click.option("--scan-id", type=int, default=None, help="Prune a single scan by id")
@click.option(
    "--source-pattern",
    default=None,
    help="GLOB pattern on source path; prunes every matching scan",
)
@click.option("-y", "--yes", is_flag=True, help="Skip the confirmation prompt")
@click.pass_context
def scans_prune(
    ctx: click.Context, scan_id: int | None, source_pattern: str | None, yes: bool
) -> None:
    """Prune scans (and their findings via cascade). Requires --scan-id and/or --source-pattern."""
    if scan_id is None and source_pattern is None:
        msg = "Provide --scan-id and/or --source-pattern (refusing to prune everything)."
        raise click.UsageError(msg)

    from .db import FaceDB

    db = FaceDB(ctx.obj["db_path"], base_dir=ctx.obj["photos_dir"])
    targets = db.find_scans(scan_id=scan_id, source_pattern=source_pattern)
    if not targets:
        click.echo("(no scans match)")
        db.close()
        return

    n_findings = sum(t["finding_count"] for t in targets)
    n_assigned = db.query(
        f"SELECT COUNT(*) FROM findings f "  # noqa: S608
        f"JOIN finding_assignment fa ON fa.finding_id = f.id "
        f"WHERE fa.subject_id IS NOT NULL AND f.scan_id IN "
        f"({','.join(str(t['id']) for t in targets)})"
    )[0][0]

    click.echo(f"About to prune {len(targets)} scan(s):")
    for t in targets[:10]:
        click.echo(f"  scan {t['id']:>6} [{t['scan_type']}]  {t['source_path']}")
    if len(targets) > 10:
        click.echo(f"  ... and {len(targets) - 10} more")
    click.echo(
        f"\nThis will delete {n_findings} finding(s), of which "
        f"{n_assigned} have manual subject assignments."
    )
    if not yes and not click.confirm("Proceed?", default=False):
        click.echo("Aborted.")
        db.close()
        return

    for t in targets:
        db.delete_scan(t["id"])
    click.echo(f"Pruned {len(targets)} scan(s) ({n_findings} findings).")
    db.close()


@cli.command()
@click.argument("source")
@click.option(
    "--scan-type",
    default="all",
    type=click.Choice(["all", "human", "pet"]),
    help="Which scan to redo (default all existing on the source)",
)
@click.option("-y", "--yes", is_flag=True, help="Skip the confirmation prompt")
@click.pass_context
def rescan(ctx: click.Context, source: str, scan_type: str, yes: bool) -> None:
    """Rescan a single source: delete existing scans + findings, run fresh detection."""
    from .db import FaceDB

    photos_dir = _require_photos_dir(ctx)
    db = FaceDB(ctx.obj["db_path"], base_dir=photos_dir)

    # Resolve to a stored path (relative to PHOTOS_DIR) and the absolute path.
    abs_path = Path(source)
    if not abs_path.is_absolute():
        abs_path = (Path(photos_dir) / source).resolve()
    if not abs_path.exists():
        raise click.UsageError(f"File not found: {abs_path}")
    stored_path = db.to_relative(str(abs_path))
    src = db.get_source_by_path(stored_path)
    if src is None:
        raise click.UsageError(
            f"Source not in DB: {stored_path}\nRun `ritrova scan` (or scan-pets/scan-videos) "
            "to ingest new files first."
        )

    existing = [s for s in db.find_scans() if s["source_id"] == src.id]
    if scan_type != "all":
        existing = [s for s in existing if s["scan_type"] == scan_type]
    if not existing:
        click.echo(f"(no matching scans on {stored_path})")
        db.close()
        return

    n_findings = sum(t["finding_count"] for t in existing)
    n_assigned = db.query(
        f"SELECT COUNT(*) FROM findings f "  # noqa: S608
        f"JOIN finding_assignment fa ON fa.finding_id = f.id "
        f"WHERE fa.subject_id IS NOT NULL AND f.scan_id IN "
        f"({','.join(str(t['id']) for t in existing)})"
    )[0][0]
    click.echo(
        f"About to rescan {stored_path}\n"
        f"Replacing {len(existing)} scan(s): {sorted({t['scan_type'] for t in existing})}\n"
        f"This will delete {n_findings} finding(s), of which "
        f"{n_assigned} have manual subject assignments."
    )
    if not yes and not click.confirm("Proceed?", default=False):
        click.echo("Aborted.")
        db.close()
        return

    types_to_redo = {t["scan_type"] for t in existing}
    for t in existing:
        db.delete_scan(t["id"])

    # Re-run only the appropriate single-source scanners.
    from .scanner import (
        scan_one_photo_for_human,
        scan_one_photo_for_pets,
        scan_one_video_for_human,
    )

    if src.type == "video":
        if "human" in types_to_redo:
            from .detector import FaceDetector

            detector = FaceDetector()
            frames_dir = Path(ctx.obj["db_path"]).parent / "tmp" / "frames"
            ok, n = scan_one_video_for_human(db, abs_path, detector, frames_dir)
            click.echo(f"  video human scan: {'ok' if ok else 'FAILED'}, {n} face(s)")
    else:
        if "human" in types_to_redo:
            from .detector import FaceDetector

            detector = FaceDetector()
            ok, n = scan_one_photo_for_human(db, abs_path, detector)
            click.echo(f"  photo human scan: {'ok' if ok else 'FAILED'}, {n} face(s)")
        if "pet" in types_to_redo:
            from .pet_detector import PetDetector

            pet_detector = PetDetector()
            ok, n = scan_one_photo_for_pets(db, abs_path, pet_detector)
            click.echo(f"  photo pet scan:   {'ok' if ok else 'FAILED'}, {n} pet(s)")

    db.close()


@cli.command("backfill-gps")
@click.pass_context
def backfill_gps(ctx: click.Context) -> None:
    """Read GPS from EXIF for all photos missing coordinates."""
    from .db import FaceDB
    from .scanner import get_exif_gps

    photos_dir = _require_photos_dir(ctx)
    db = FaceDB(ctx.obj["db_path"], base_dir=photos_dir)
    rows = db.query("SELECT id, file_path FROM sources WHERE latitude IS NULL")
    updated = 0
    total = len(rows)
    for i, r in enumerate(rows, 1):
        pid, fp = r[0], r[1]
        resolved = db.resolve_path(fp)
        if not resolved.exists():
            continue
        gps = get_exif_gps(resolved)
        if gps:
            db.run(
                "UPDATE sources SET latitude = ?, longitude = ? WHERE id = ?",
                (gps[0], gps[1], pid),
            )
            updated += 1
        if i % 500 == 0 or i == total:
            print(f"\r  [{i}/{total}] updated={updated}", end="", flush=True)
    print(f"\nUpdated {updated} of {total} sources with GPS coordinates.")
    db.close()


@cli.command("migrate-paths")
@click.pass_context
def migrate_paths(ctx: click.Context) -> None:
    """Rewrite absolute paths in the DB to relative (using --photos-dir as base)."""
    from .db import FaceDB

    photos_dir = ctx.obj["photos_dir"]
    if not photos_dir:
        print("Error: --photos-dir is required for migration")
        raise SystemExit(1)

    db = FaceDB(ctx.obj["db_path"], base_dir=photos_dir)
    base = str(db.base_dir) + "/"

    rows = db.query("SELECT id, file_path FROM sources")
    migrated = 0
    for r in rows:
        sid, fp = r[0], r[1]
        if fp and fp.startswith(base):
            new_fp = fp[len(base) :]
            db.run("UPDATE sources SET file_path = ? WHERE id = ?", (new_fp, sid))
            migrated += 1

    print(f"Migrated {migrated} of {len(rows)} source paths to relative.")


@cli.command("prune")
@click.option("--dry-run", is_flag=True, help="Report duplicates without deleting")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def prune(ctx: click.Context, dry_run: bool, yes: bool) -> None:
    """Remove duplicate findings on the same source.

    When a source is re-analysed, duplicate findings can accumulate. This
    command collapses them:

    \b
    - (source_id, subject_id): keep newest finding per assigned subject
    - (source_id, cluster_id): keep newest finding per cluster (unassigned)
    """
    from .db import FaceDB

    db = FaceDB(ctx.obj["db_path"], base_dir=ctx.obj["photos_dir"])
    report = db.prune_duplicate_findings(dry_run=True)

    print(f"Database: {ctx.obj['db_path']}")
    print(f"  Duplicate by subject assignment: {report.by_subject}")
    print(f"  Duplicate by cluster membership: {report.by_cluster}")
    print(f"  Total to remove: {report.total}")

    if report.total == 0:
        print("\nNo duplicates found.")
        db.close()
        return

    if dry_run:
        print("\nDry run — no changes made.")
        db.close()
        return

    if not yes and not click.confirm(f"Delete {report.total} duplicate finding(s)?", default=False):
        print("Aborted.")
        db.close()
        return

    report = db.prune_duplicate_findings(dry_run=False)
    print(f"Deleted {report.total} duplicate finding(s).")
    db.close()


@cli.command("doctor")
@click.option(
    "--fix",
    is_flag=True,
    help="Delete the orphans found (otherwise: report-only, exit 0 if clean, 1 if dirty).",
)
@click.option("-y", "--yes", is_flag=True, help="Skip the confirmation prompt (requires --fix)")
@click.pass_context
def doctor(ctx: click.Context, fix: bool, yes: bool) -> None:
    """Find (and optionally delete) orphaned DB rows.

    Orphans are child rows whose parent was deleted from a connection with
    ``PRAGMA foreign_keys=OFF`` (e.g. a plain ``sqlite3`` CLI session). The
    app connection enforces FKs, so only external tools can create these.
    """
    from .db import FaceDB

    db = FaceDB(ctx.obj["db_path"], base_dir=ctx.obj["photos_dir"])
    report = db.find_orphans()

    categories = [
        ("findings with missing source", report.findings_missing_source),
        ("findings with missing scan", report.findings_missing_scan),
        ("scans with missing source", report.scans_missing_source),
        ("dismissed_findings with missing finding", report.dismissed_missing_finding),
    ]

    click.echo(f"DB: {ctx.obj['db_path']}")
    for label, ids in categories:
        click.echo(f"  {label}: {len(ids)}")
        # Show a small sample so the user can cross-reference with their logs
        # / shell history. Full dump would be noisy on a broken DB.
        if ids:
            sample = ids[:8]
            more = "" if len(ids) <= 8 else f" … (+{len(ids) - 8} more)"
            click.echo(f"    sample ids: {sample}{more}")

    if report.total == 0:
        click.echo("\nNo orphans found. DB is healthy.")
        db.close()
        return

    click.echo(f"\nTotal orphans: {report.total}")

    if not fix:
        click.echo("Run with --fix to delete them. Exiting with code 1 (dirty).")
        db.close()
        raise SystemExit(1)

    if not yes and not click.confirm(f"Delete {report.total} orphaned row(s)?", default=False):
        click.echo("Aborted.")
        db.close()
        return

    db.delete_orphans(report)
    click.echo(f"Deleted {report.total} orphan row(s).")
    db.close()
    db.close()
