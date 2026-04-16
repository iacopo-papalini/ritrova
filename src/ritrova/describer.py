"""VLM-powered scene description and tagging for photos (FEAT-8).

Two-stage pipeline:
1. ``Describer`` — VLM inference via mlx-vlm, generates English captions + tags.
2. ``Translator`` — text-only translation (en→it) via MarianMT, converts to Italian.

Each stage is independently configurable and testable. The ``describe_sources``
pipeline function orchestrates both.
"""

from __future__ import annotations

import json
import logging
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
    "Reply with the caption on the first line, then each tag on its own line.\n"
    "Example:\n"
    "A dog sits on a beach at sunset.\n"
    "dog\n"
    "beach\n"
    "sunset\n"
    "sand\n"
    "water"
)

DEFAULT_USER_PROMPT_WITH_VOCAB = (
    "Describe this image.\n"
    "When possible, reuse these existing tags: {vocab}\n"
    "You may add new tags if needed."
)

DEFAULT_USER_PROMPT_NO_VOCAB = "Describe this image."


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
    ) -> None:
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.user_prompt_with_vocab = user_prompt_with_vocab
        self.user_prompt_no_vocab = user_prompt_no_vocab
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from mlx_vlm import load

        logger.info("Loading VLM: %s", self.model_id)
        self._model, self._processor = load(self.model_id)

    def describe(self, image_path: Path, vocab_hint: str | None = None) -> tuple[str, set[str]]:
        """Generate an English caption and tags for a single image.

        Returns ``(caption, tags)``. Returns ``("", set())`` if the image
        is unreadable.
        """
        self._ensure_loaded()
        assert self._model is not None
        assert self._processor is not None

        from mlx_vlm import generate

        if vocab_hint:
            user_text = self.user_prompt_with_vocab.format(vocab=vocab_hint)
        else:
            user_text = self.user_prompt_no_vocab

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        formatted = self._processor.apply_chat_template(messages, add_generation_prompt=True)

        try:
            raw = Image.open(image_path)
            img = ImageOps.exif_transpose(raw)
            if img is None:
                img = raw
            img = img.convert("RGB")
        except OSError:
            logger.warning("Could not read image: %s", image_path)
            return ("", set())

        img = _resize_for_vlm(img, max_side=1024)

        output = generate(
            self._model,
            self._processor,
            formatted,
            image=[img],
            max_tokens=256,
            temperature=0.0,
            repetition_penalty=1.2,
            verbose=False,
        )

        text = output.text if hasattr(output, "text") else str(output)
        return _parse_vlm_response(text)


def _parse_vlm_response(text: str) -> tuple[str, set[str]]:
    """Extract caption and tags from the VLM response.

    Supports two formats:
    1. Line-based: first line is caption, subsequent lines are tags.
    2. JSON: ``{"caption": "...", "tags": [...]}`` (with optional fences).

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
        return (caption, json_tags)
    except json.JSONDecodeError, AttributeError:
        pass

    # Line-based: first line is caption, rest are tags
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if not lines:
        return ("", set())
    caption = lines[0]
    line_tags: set[str] = set()
    for line in lines[1:]:
        tag = line.strip().lower().rstrip(".")
        if tag and len(tag.split()) == 1:
            line_tags.add(tag)
        if len(line_tags) >= MAX_TAGS:
            break
    return (caption, line_tags)


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
        gen_kwargs: dict[str, object] = {"max_new_tokens": 256}
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

        caption, tags = describer.describe(resolved, vocab_hint=vocab_hint)
        if not caption:
            errors += 1
            continue

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
