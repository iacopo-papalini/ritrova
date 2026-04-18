"""VLM-powered scene description and tagging for photos (FEAT-8, opt-in).

Two-stage pipeline, activated by ``ritrova analyse --caption``:
1. ``Describer`` — VLM inference via mlx-vlm, generates English captions + tags.
2. ``Translator`` — text-only translation (en→it) via MarianMT, converts to Italian.

Apple Silicon only. The MLX backend is the only supported VLM runtime;
a non-Apple-Silicon platform raises a clear error at load time. See
ADR-011 for the retirement of the transformers/CUDA Windows backend.
"""

from __future__ import annotations

import json
import logging
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageFile, ImageOps

from .db import FaceDB

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

DEFAULT_VLM_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
DEFAULT_TRANSLATOR_MODEL = "Helsinki-NLP/opus-mt-tc-big-en-it"
MAX_TAGS = 15


def _prefers_mlx_backend() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


DEFAULT_SYSTEM_PROMPT = (
    "You catalog photos.\n\n"
    "Rules for the caption:\n"
    "- State ONLY what is visible. Use 1-2 short sentences.\n"
    "- Never guess, interpret, or speculate. No 'perhaps', 'appears to be', "
    "'suggests', 'seems like', 'possibly', 'likely'.\n"
    '- WRONG: "A man who appears to be a knight at what seems like a festival."\n'
    '- RIGHT: "A man in a leather costume holds a large bird on his arm. '
    'A crowd watches him in a stone courtyard."\n\n'
    "Rules for tags:\n"
    "- 3 to 10 single-word English nouns for visible things.\n"
    "- No adjectives, no verbs, no guesses.\n\n"
    "Reply with a single JSON object, no prose, no markdown fences:\n"
    '{"caption": "...", "tags": ["...", "..."]}\n\n'
    "Example:\n"
    '{"caption": "A dog sits on a beach at sunset.", '
    '"tags": ["dog", "beach", "sunset", "sand", "water"]}'
)

DEFAULT_USER_PROMPT_WITH_VOCAB = (
    "Describe this image.\n"
    "When possible, reuse these existing tags: {vocab}\n"
    "You may add new tags if needed."
)

DEFAULT_USER_PROMPT_NO_VOCAB = "Describe this image."


@dataclass
class DescribeOutput:
    """VLM output for one image."""

    caption: str
    tags: set[str]


def _resize_for_vlm(img: Image.Image, max_side: int = 1024) -> Image.Image:
    """Downscale so the longest side is at most ``max_side`` pixels."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


# ── Stage 1: VLM ────────────────────────────────────────────────────


class Describer:
    """Generate English captions and tags for images using a VLM.

    The model is loaded lazily on first call and reused. Thread-unsafe —
    intended for single-threaded CLI batch processing.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_VLM_MODEL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt_with_vocab: str = DEFAULT_USER_PROMPT_WITH_VOCAB,
        user_prompt_no_vocab: str = DEFAULT_USER_PROMPT_NO_VOCAB,
        *,
        max_tokens: int = 128,
        max_side: int = 896,
    ) -> None:
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.user_prompt_with_vocab = user_prompt_with_vocab
        self.user_prompt_no_vocab = user_prompt_no_vocab
        # Observed captions are ~20-24 words (≈50-80 tokens); 128 keeps a
        # comfortable headroom while saving ~half of the old 256-token budget.
        # Image side 896 shaves ~9% off VLM time vs 1024 with no quality loss
        # on a 20-photo high-complexity A/B (ADR-010 §2a Phase B).
        self.max_tokens = max_tokens
        self.max_side = max_side
        # Untyped because mlx-vlm's model/processor types would drag the
        # heavy MLX import into this module's import-time surface.
        self._model: Any = None
        self._processor: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not _prefers_mlx_backend():
            raise RuntimeError(
                "VLM captioning requires Apple Silicon (MLX backend). "
                "The transformers/CUDA backend was retired in ADR-011; "
                "see docs/adr-011-retire-vlm-default.md for the rationale."
            )
        try:
            from mlx_vlm import load
        except ImportError as exc:
            raise RuntimeError(
                "mlx-vlm is not installed. Install the `caption` extras: `uv sync --extra caption`."
            ) from exc
        logger.info("Loading VLM via MLX: %s", self.model_id)
        self._model, self._processor = load(self.model_id)

    def describe(self, image_path: Path, vocab_hint: str | None = None) -> DescribeOutput:
        """Generate an English caption and tags from a file path.

        Use ``describe_image`` when you already have a loaded PIL Image.
        """
        try:
            raw = Image.open(image_path)
            img = ImageOps.exif_transpose(raw)
            if img is None:
                img = raw
            img = img.convert("RGB")
        except OSError:
            logger.warning("Could not read image: %s", image_path)
            return DescribeOutput(caption="", tags=set())
        return self.describe_image(img, vocab_hint=vocab_hint)

    def describe_image(
        self, pil_image: Image.Image, vocab_hint: str | None = None
    ) -> DescribeOutput:
        """Generate an English caption and tags from an already-loaded PIL RGB Image."""
        self._ensure_loaded()

        if vocab_hint:
            user_text = self.user_prompt_with_vocab.format(vocab=vocab_hint)
        else:
            user_text = self.user_prompt_no_vocab

        img = _resize_for_vlm(pil_image, max_side=self.max_side)
        assert self._model is not None
        assert self._processor is not None

        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        # mlx_vlm's family-aware helper inserts the image token and picks
        # the right template for qwen / gemma3 / pixtral / llama-vision.
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]
        formatted = apply_chat_template(self._processor, self._model.config, messages, num_images=1)

        output = generate(
            self._model,
            self._processor,
            formatted,
            image=[img],
            max_tokens=self.max_tokens,
            temperature=0.0,
            repetition_penalty=1.2,
            verbose=False,
        )

        text = output.text if hasattr(output, "text") else str(output)
        return _parse_vlm_response(text)


def _parse_vlm_response(text: str) -> DescribeOutput:
    """Extract caption and tags from the VLM response.

    Supports two formats:
    1. JSON: ``{"caption": "...", "tags": [...]}`` (with optional fences).
    2. Line-based fallback: first line is caption, subsequent lines are tags.

    Tags are lowercased and capped at MAX_TAGS.
    """
    cleaned = text.strip()

    # Try JSON first (may come with markdown fences)
    json_text = re.sub(r"```json\s*", "", cleaned)
    json_text = re.sub(r"```\s*", "", json_text).strip()
    try:
        data = json.loads(json_text)
        caption = str(data.get("caption", "")).strip()
        raw_tags = data.get("tags", [])
        json_tags: set[str] = set()
        for t in raw_tags:
            tag = str(t).strip().lower()
            if tag:
                json_tags.add(tag)
            if len(json_tags) >= MAX_TAGS:
                break
        return DescribeOutput(caption=caption, tags=json_tags)
    except json.JSONDecodeError, AttributeError, TypeError:
        pass

    # Line-based fallback: first line is caption, rest are tags.
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if not lines:
        return DescribeOutput(caption="", tags=set())
    caption = lines[0]
    line_tags: set[str] = set()
    for line in lines[1:]:
        tag = line.strip().lower().rstrip(".")
        if tag and len(tag.split()) == 1:
            line_tags.add(tag)
        if len(line_tags) >= MAX_TAGS:
            break
    return DescribeOutput(caption=caption, tags=line_tags)


# ── Stage 2: Translation ────────────────────────────────────────────


class Translator:
    """Translate English text to Italian using MarianMT.

    Loaded lazily on first call. Uses the ``transformers`` pipeline API
    which is already a project dependency.
    """

    def __init__(self, model_id: str = DEFAULT_TRANSLATOR_MODEL) -> None:
        self.model_id = model_id
        # Lazy-loaded transformers objects — untyped because transformers
        # doesn't ship py.typed and mypy can't resolve the dynamic types.
        self._tokenizer: Any = None
        self._translation_model: Any = None
        self._target_lang_id: int | None = None
        # Overridable by the translate-bench harness; None = use shipped
        # defaults (num_beams=1 greedy, CPU).
        self._device: str = "cpu"
        self._bench_gen_kwargs: dict[str, object] | None = None

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading translator: %s", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._translation_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        # NLLB models need a target language token; MarianMT doesn't.
        self._target_lang_id = self._tokenizer.convert_tokens_to_ids("ita_Latn")
        # Falls back to None for non-NLLB models (MarianMT ignores forced_bos_token_id)
        if self._target_lang_id == self._tokenizer.unk_token_id:
            self._target_lang_id = None

    def _translate_text(self, text: str) -> str:
        """Translate a single English text to Italian."""
        assert self._tokenizer is not None
        assert self._translation_model is not None
        logger.debug("Translator input: %s", text)
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True)
        if self._device != "cpu":
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        # opus-mt-tc-big-en-it ships with num_beams=4 by default (4-way beam
        # search). For photo captions the quality difference vs greedy is
        # negligible — switching to num_beams=1 cuts the decoder cost by
        # roughly 3-4x. See ADR-010 §2a Phase E. translate-bench may override
        # via _bench_gen_kwargs.
        gen_kwargs: dict[str, object] = {
            "max_new_tokens": 128,
            "max_length": None,
            "num_beams": 1,
            "do_sample": False,
        }
        if self._bench_gen_kwargs is not None:
            gen_kwargs.update(self._bench_gen_kwargs)
            gen_kwargs["max_length"] = None  # preserve our no-max-length default
        if self._target_lang_id is not None:
            gen_kwargs["forced_bos_token_id"] = self._target_lang_id
        outputs = self._translation_model.generate(**inputs, **gen_kwargs)
        result: str = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        logger.debug("Translator output: %s", result)
        return result

    def translate(self, caption: str, tags: set[str]) -> tuple[str, set[str]]:
        """Translate an English caption and tag set to Italian.

        Sends caption and tags as a single block separated by a newline
        marker. Faithful translation models (MarianMT) preserve the
        structure; the tags provide context for disambiguation.

        Returns ``(italian_caption, italian_tags)``.
        """
        if not caption and not tags:
            return ("", set())

        self._ensure_loaded()

        tag_list = sorted(tags)
        combined = caption
        if tag_list:
            combined += "\nTags: " + ", ".join(tag_list)

        translated = self._translate_text(combined)

        # Split on the translated "Tags:" marker
        it_caption = translated
        it_tags: set[str] = set()
        for marker in ("\nTags:", "\nTag:", "\ntag:", "Tags:", "Tag:"):
            if marker in translated:
                parts = translated.split(marker, 1)
                it_caption = parts[0].strip()
                for raw in parts[1].split(","):
                    word = raw.strip().rstrip(".").lower()
                    if word:
                        it_tags.add(word)
                break

        return (it_caption, it_tags)


# ── Pipeline ─────────────────────────────────────────────────────────


def _build_vocab_hint(db: FaceDB) -> str | None:
    """Build a sorted comma-separated vocabulary string from existing tags."""
    tags = db.get_all_tags()
    if not tags:
        return None
    return ", ".join(sorted(tags))


@dataclass
class DescribeResult:
    described: int
    errors: int
    total_tags: int


def describe_sources(
    db: FaceDB,
    source_ids: list[int],
    describer: Describer,
    translator: Translator | None = None,
    *,
    force: bool = False,
    vocab_refresh_interval: int = 500,
) -> DescribeResult:
    """Run the describe pipeline on a list of sources.

    Stage 1 (VLM) produces English captions + tags.
    Stage 2 (translation, optional) converts to Italian.
    If ``translator`` is None, English output is stored directly.
    """
    vocab_hint = _build_vocab_hint(db)
    described = 0
    errors = 0
    total = len(source_ids)

    strategy = describer.model_id
    if translator is not None:
        strategy = f"{describer.model_id}+{translator.model_id}"

    for i, source_id in enumerate(source_ids, 1):
        source = db.get_source(source_id)
        if not source or source.type != "photo":
            continue
        resolved = db.resolve_path(source.file_path)
        if not resolved.exists():
            errors += 1
            continue

        if i > 1 and (i - 1) % vocab_refresh_interval == 0:
            vocab_hint = _build_vocab_hint(db)

        output = describer.describe(resolved, vocab_hint=vocab_hint)
        if not output.caption:
            errors += 1
            continue

        caption, tags = output.caption, output.tags
        if translator is not None:
            caption, tags = translator.translate(caption, tags)

        if force:
            db.delete_describe_scans(source_id)

        scan_id = db.record_scan(source_id, "describe", detection_strategy=strategy)
        db.add_description(source_id, scan_id, caption, tags)
        described += 1

        tag_str = ", ".join(sorted(tags)) if tags else "(no tags)"
        logger.info("%s", resolved)
        logger.info("  %s", caption)
        logger.info("  [%s]", tag_str)

    if total > 0 and described == 0 and errors == 0:
        logger.info("(no sources to process)")

    return DescribeResult(
        described=described,
        errors=errors,
        total_tags=len(db.get_all_tags()),
    )
