"""VLM-powered scene description and tagging for photos (FEAT-8).

Two-stage pipeline:
1. ``Describer`` — VLM inference via mlx-vlm, generates English captions + tags.
2. ``Translator`` — text-only translation (en→it) via MarianMT, converts to Italian.

Each stage is independently configurable and testable. The ``describe_sources``
pipeline function orchestrates both.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFile, ImageOps

from .db import FaceDB

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

DEFAULT_VLM_MODEL_MLX = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
# AWQ-quantized 7B keeps the same base model as the Mac (~6 GB on CUDA —
# fits an 8 GB card with room for image-encoder activations). The earlier
# default of the 3B fp16 variant was a quality regression — on a 20-photo
# A/B (seed 2024, high-complexity) 3B systematically lost scene context
# ("softball dugout" → "sports venue"; "inflatable bee costume" →
# "person in costume"), which hurts a searchable archive. See ADR-010.
DEFAULT_VLM_MODEL_TRANSFORMERS_AWQ = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
DEFAULT_VLM_MODEL_TRANSFORMERS_FALLBACK = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_TRANSLATOR_MODEL = "Helsinki-NLP/opus-mt-tc-big-en-it"
MAX_TAGS = 15


def _prefers_mlx_backend() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def _has_autoawq_runtime() -> bool:
    return importlib.util.find_spec("awq") is not None


def _default_transformers_vlm_model() -> str:
    # AWQ-quantized 7B keeps the same base model as the Mac (~6 GB on CUDA and
    # fits an 8 GB card). When the AWQ runtime is unavailable, fall back to
    # the portable 3B transformers model so `uv run` remains installable.
    if _has_autoawq_runtime():
        return DEFAULT_VLM_MODEL_TRANSFORMERS_AWQ
    return DEFAULT_VLM_MODEL_TRANSFORMERS_FALLBACK


DEFAULT_VLM_MODEL = (
    DEFAULT_VLM_MODEL_MLX if _prefers_mlx_backend() else _default_transformers_vlm_model()
)

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
    "Rules for has_people / has_animals:\n"
    "- has_people: true if ANY human face, head, or identifiable person "
    "(even partial, even blurry) is visible. Only false when there are "
    "clearly no people at all.\n"
    "- has_animals: true if ANY dog, cat, or other pet is visible (even "
    "partial, even peripheral). Only false when there are no animals.\n"
    "- When in doubt, answer true. False positives cost nothing; false "
    "negatives cause people and pets to be missed.\n\n"
    "Reply with a single JSON object, no prose, no markdown fences:\n"
    '{"caption": "...", "tags": ["...", "..."], '
    '"has_people": true, "has_animals": false}\n\n'
    "Example:\n"
    '{"caption": "A dog sits on a beach at sunset.", '
    '"tags": ["dog", "beach", "sunset", "sand", "water"], '
    '"has_people": false, "has_animals": true}'
)

DEFAULT_USER_PROMPT_WITH_VOCAB = (
    "Describe this image.\n"
    "When possible, reuse these existing tags: {vocab}\n"
    "You may add new tags if needed."
)

DEFAULT_USER_PROMPT_NO_VOCAB = "Describe this image."


@dataclass
class DescribeOutput:
    """VLM output for one image.

    ``has_people`` / ``has_animals`` default to True so that unparsable or
    absent fields fail open — detection still runs and nothing is missed.
    """

    caption: str
    tags: set[str]
    has_people: bool = True
    has_animals: bool = True


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
        # Untyped because the backend can be either an mlx-vlm model or a
        # transformers pipeline — both concrete types drag in heavy imports.
        self._model: Any = None
        self._processor: Any = None
        self._backend_name: str | None = None

    def _load_mlx_backend(self) -> None:
        from mlx_vlm import load

        logger.info("Loading VLM via MLX: %s", self.model_id)
        self._model, self._processor = load(self.model_id)
        self._backend_name = "mlx"

    def _load_transformers_backend(self) -> None:
        from transformers import pipeline

        logger.info("Loading VLM via transformers: %s", self.model_id)
        if torch.cuda.is_available():
            device: str | int = 0
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = -1
        # sdpa attention ships with PyTorch ≥2.1 and gives a ~30% speed-up
        # over the default eager attention on Qwen2.5-VL with no quality
        # change. flash_attention_2 would be faster still but requires a
        # separate install, so stick with sdpa.
        self._model = pipeline(
            task="image-text-to-text",
            model=self.model_id,
            device=device,
            dtype="auto",
            model_kwargs={"attn_implementation": "sdpa"},
        )
        self._processor = None
        self._backend_name = "transformers"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        errors: list[BaseException] = []
        looks_like_mlx_model = self.model_id.startswith("mlx-community/")
        # On non-Apple-Silicon platforms mlx-vlm is not installed (see
        # pyproject.toml). Skip the MLX branch entirely so import failures
        # don't pollute error messages, and transformers is the only path.
        loaders: tuple[Any, ...]
        if _prefers_mlx_backend():
            loaders = (
                (self._load_mlx_backend, self._load_transformers_backend)
                if looks_like_mlx_model
                else (self._load_transformers_backend, self._load_mlx_backend)
            )
        else:
            loaders = (self._load_transformers_backend,)
        for loader in loaders:
            try:
                loader()
                return
            except Exception as exc:  # pragma: no cover - depends on local ML stack
                errors.append(exc)

        raise RuntimeError(
            "Could not load any VLM backend for this platform. "
            "Apple Silicon prefers MLX; Windows/Linux fall back to transformers. "
            f"Tried model `{self.model_id}` and got: "
            + " | ".join(f"{type(err).__name__}: {err}" for err in errors)
        ) from errors[-1]

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
        if self._backend_name == "mlx":
            return self._describe_image_mlx(img, user_text)
        if self._backend_name == "transformers":
            return self._describe_image_transformers(img, user_text)
        raise RuntimeError("VLM backend was not initialized correctly")

    def _describe_image_mlx(self, pil_image: Image.Image, user_text: str) -> DescribeOutput:
        assert self._model is not None
        assert self._processor is not None

        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        # Use mlx_vlm's family-aware helper: it inserts the image token and
        # picks the right template for gemma3 / qwen / pixtral / llama-vision /
        # anything else mlx_vlm understands. The HF processor's own
        # apply_chat_template does not work for models (e.g. gemma-3) that
        # ship without a chat template in their processor config.
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]
        formatted = apply_chat_template(self._processor, self._model.config, messages, num_images=1)

        output = generate(
            self._model,
            self._processor,
            formatted,
            image=[pil_image],
            max_tokens=self.max_tokens,
            temperature=0.0,
            repetition_penalty=1.2,
            verbose=False,
        )

        text = output.text if hasattr(output, "text") else str(output)
        return _parse_vlm_response(text)

    def _describe_image_transformers(
        self, pil_image: Image.Image, user_text: str
    ) -> DescribeOutput:
        assert self._model is not None

        generation_config = self._model.model.generation_config.__class__.from_dict(
            self._model.model.generation_config.to_dict()
        )
        generation_config.max_new_tokens = self.max_tokens
        generation_config.max_length = None

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        output = self._model(
            text=messages,
            generation_config=generation_config,
            return_full_text=False,
        )
        return _parse_vlm_response(_extract_transformers_text(output))


def _extract_transformers_text(output: object) -> str:
    """Extract generated text from transformers image-text-to-text pipeline output."""
    if isinstance(output, list) and output:
        output = output[0]
    generated = output.get("generated_text", output) if isinstance(output, dict) else output

    if isinstance(generated, list) and generated:
        generated = generated[-1]
    content = generated.get("content", generated) if isinstance(generated, dict) else generated

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return "\n".join(part for part in text_parts if part).strip()
    return str(content).strip()


def _parse_vlm_response(text: str) -> DescribeOutput:
    """Extract caption, tags and subject booleans from the VLM response.

    Supports two formats:
    1. JSON: ``{"caption": "...", "tags": [...], "has_people": bool,
       "has_animals": bool}`` (with optional fences).
    2. Line-based fallback: first line is caption, subsequent lines are tags.
       In line-mode ``has_people`` / ``has_animals`` are left at the fail-open
       default (True), so no detection is ever skipped by a mis-parse.

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
        has_people = _coerce_bool_fail_open(data.get("has_people"))
        has_animals = _coerce_bool_fail_open(data.get("has_animals"))
        # Override: if the VLM contradicts itself (caption mentions a cat
        # yet has_animals=false), trust the caption. Accuracy-first.
        if not has_people and _text_mentions(_PEOPLE_KEYWORDS, caption, json_tags):
            has_people = True
        if not has_animals and _text_mentions(_ANIMAL_KEYWORDS, caption, json_tags):
            has_animals = True
        return DescribeOutput(
            caption=caption,
            tags=json_tags,
            has_people=has_people,
            has_animals=has_animals,
        )
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    # Line-based fallback: first line is caption, rest are tags.
    # Booleans stay at fail-open defaults.
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


def _coerce_bool_fail_open(value: Any) -> bool:
    """Return False ONLY when the VLM explicitly said false; True otherwise.

    Anything unparseable, missing, or ambiguous defaults to True so detection
    still runs. Accepts real booleans and the common string forms.
    """
    if isinstance(value, bool):
        return value
    return not (isinstance(value, str) and value.strip().lower() in {"false", "no", "0"})


# Qwen2.5-VL sometimes describes a subject in the caption / tags yet emits
# the opposite boolean in the same JSON (observed: "A person sitting with a
# cat on their lap" with has_animals=false). Treat the boolean as advisory
# and force-true whenever the caption or tags mention a relevant noun.
# English only — the describer runs before translation.
_PEOPLE_KEYWORDS = frozenset(
    {
        "person",
        "people",
        "man",
        "men",
        "woman",
        "women",
        "boy",
        "girl",
        "child",
        "children",
        "kid",
        "kids",
        "baby",
        "infant",
        "adult",
        "human",
        "face",
        "crowd",
    }
)
_ANIMAL_KEYWORDS = frozenset(
    {
        "cat",
        "cats",
        "kitten",
        "kittens",
        "feline",
        "dog",
        "dogs",
        "puppy",
        "puppies",
        "canine",
        "pet",
        "pets",
        "animal",
        "animals",
    }
)


def _text_mentions(keywords: frozenset[str], caption: str, tags: set[str]) -> bool:
    """True if any keyword appears as a whole word in caption or as a tag."""
    if any(t.lower() in keywords for t in tags):
        return True
    tokens = re.findall(r"[a-zA-Z]+", caption.lower())
    return any(tok in keywords for tok in tokens)


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
