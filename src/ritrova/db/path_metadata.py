"""Path-derived source metadata.

This module deliberately treats archive paths as structured data:
directory/event names and filenames often carry better dates and search
tokens than EXIF. The parser is pure so we can regression-test real path
shapes without touching SQLite; the mixin below owns persistence.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal, cast

from ._base import _DBAccessor, _locked
from .models import SourcePathMetadata

DatePrecision = Literal["day", "month", "year", "unknown"]
DateSource = Literal["filename", "directory", "exif", "unknown"]

_YMD_DASH_RE = re.compile(
    r"(?<!\d)(19\d{2}|20\d{2})[-.](0[1-9]|1[0-2])[-.](0[1-9]|[12]\d|3[01])(?!\d)"
)
_YMD_UNDERSCORE_RE = re.compile(
    r"(?<!\d)(19\d{2}|20\d{2})_(0[1-9]|1[0-2])_(0[1-9]|[12]\d|3[01])(?!\d)"
)
_YMD_COMPACT_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")
_YM_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})[-.](0[1-9]|1[0-2])(?!\d)")
_YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")

_DATE_PATTERNS = (_YMD_DASH_RE, _YMD_UNDERSCORE_RE, _YMD_COMPACT_RE, _YM_RE, _YEAR_RE)
_NOISE_TAGS = {
    "copia",
    "copy",
    "dsc",
    "dscf",
    "dscn",
    "foto",
    "heic",
    "image",
    "img",
    "immagini",
    "jpeg",
    "jpg",
    "mov",
    "mvi",
    "photo",
    "photos",
    "pic",
    "pict",
    "pxl",
    "video",
    "whatsapp",
}


@dataclass(frozen=True)
class ParsedDate:
    text: str
    precision: DatePrecision


@dataclass(frozen=True)
class ParsedPathMetadata:
    date_text: str | None
    date_precision: DatePrecision
    date_source: DateSource
    date_conflict: bool
    filename_date: str | None
    directory_date: str | None
    exif_date: str | None
    path_tags: set[str]


def parse_source_path_metadata(file_path: str, taken_at: str | None = None) -> ParsedPathMetadata:
    """Extract chosen date + path tags from a source path.

    Conflict policy:
    - directory dates beat filename dates when the overlapping parts disagree;
    - directory dates beat suspicious filename placeholders like Jan 1;
    - otherwise the most precise path date wins;
    - EXIF is fallback only.
    """
    path = PurePosixPath(file_path)
    filename_date = _first_date(path.name)
    directory_date = _first_date("/".join(path.parts[:-1]))
    exif_date = _parse_exif_date(taken_at)
    conflict = _dates_conflict(filename_date, directory_date)

    chosen: tuple[DateSource, ParsedDate] | None = None
    if filename_date and directory_date:
        if conflict or _is_suspicious_filename_date(filename_date):
            chosen = ("directory", directory_date)
        elif _precision_rank(filename_date.precision) >= _precision_rank(directory_date.precision):
            chosen = ("filename", filename_date)
        else:
            chosen = ("directory", directory_date)
    elif filename_date:
        chosen = ("filename", filename_date)
    elif directory_date:
        chosen = ("directory", directory_date)
    elif exif_date:
        chosen = ("exif", exif_date)

    return ParsedPathMetadata(
        date_text=chosen[1].text if chosen else None,
        date_precision=chosen[1].precision if chosen else "unknown",
        date_source=chosen[0] if chosen else "unknown",
        date_conflict=conflict,
        filename_date=filename_date.text if filename_date else None,
        directory_date=directory_date.text if directory_date else None,
        exif_date=exif_date.text if exif_date else None,
        path_tags=_extract_path_tags(path),
    )


def _first_date(text: str) -> ParsedDate | None:
    patterns: list[tuple[re.Pattern[str], DatePrecision]] = [
        (_YMD_DASH_RE, "day"),
        (_YMD_UNDERSCORE_RE, "day"),
        (_YMD_COMPACT_RE, "day"),
        (_YM_RE, "month"),
        (_YEAR_RE, "year"),
    ]
    for pattern, precision in patterns:
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groups()
        if precision == "day":
            return ParsedDate(f"{groups[0]}-{groups[1]}-{groups[2]}", "day")
        if precision == "month":
            return ParsedDate(f"{groups[0]}-{groups[1]}", "month")
        return ParsedDate(groups[0], "year")
    return None


def _parse_exif_date(raw: str | None) -> ParsedDate | None:
    if not raw:
        return None
    normalized = raw.strip().replace(":", "-", 2)
    match = _YMD_DASH_RE.search(normalized)
    if match:
        return ParsedDate(f"{match.group(1)}-{match.group(2)}-{match.group(3)}", "day")
    match = _YM_RE.search(normalized)
    if match:
        return ParsedDate(f"{match.group(1)}-{match.group(2)}", "month")
    match = _YEAR_RE.search(normalized)
    if match:
        return ParsedDate(match.group(1), "year")
    return None


def _dates_conflict(left: ParsedDate | None, right: ParsedDate | None) -> bool:
    if left is None or right is None:
        return False
    left_parts = left.text.split("-")
    right_parts = right.text.split("-")
    n = min(len(left_parts), len(right_parts))
    return left_parts[:n] != right_parts[:n]


def _is_suspicious_filename_date(date: ParsedDate) -> bool:
    return date.precision == "day" and date.text[5:] == "01-01"


def _precision_rank(precision: DatePrecision) -> int:
    return {"unknown": 0, "year": 1, "month": 2, "day": 3}[precision]


def _extract_path_tags(path: PurePosixPath) -> set[str]:
    tags: set[str] = set()
    for raw_part in (*path.parts[:-1], path.stem):
        part = raw_part
        for pattern in _DATE_PATTERNS:
            part = pattern.sub(" ", part)
        for match in _WORD_RE.finditer(part):
            tag = _normalize_tag(match.group(0))
            if tag:
                tags.add(tag)
    return tags


def _normalize_tag(raw: str) -> str | None:
    tag = unicodedata.normalize("NFKD", raw.lower().strip("'"))
    tag = "".join(ch for ch in tag if not unicodedata.combining(ch))
    tag = tag.replace("'", "")
    if len(tag) < 3 or tag in _NOISE_TAGS:
        return None
    return tag


def normalize_path_tag(raw: str) -> str | None:
    """Normalize user-entered path-tag filters like parser-produced tags."""
    return _normalize_tag(raw)


class PathMetadataMixin(_DBAccessor):
    @staticmethod
    def _encode_path_tags(tags: set[str]) -> str:
        return ":" + ":".join(sorted(tags)) + ":" if tags else ""

    @staticmethod
    def _decode_path_tags(raw: str) -> set[str]:
        return {t for t in raw.split(":") if t}

    @_locked
    def upsert_source_path_metadata(self, source_id: int) -> SourcePathMetadata | None:
        source = self.get_source(source_id)
        if source is None:
            return None
        parsed = parse_source_path_metadata(source.file_path, source.taken_at)
        indexed_at = self._now()
        self.conn.execute(
            """
            INSERT OR REPLACE INTO source_path_metadata
            (source_id, date_text, date_precision, date_source, date_conflict,
             filename_date, directory_date, exif_date, path_tags, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_id,
                parsed.date_text,
                parsed.date_precision,
                parsed.date_source,
                1 if parsed.date_conflict else 0,
                parsed.filename_date,
                parsed.directory_date,
                parsed.exif_date,
                self._encode_path_tags(parsed.path_tags),
                indexed_at,
            ),
        )
        self.conn.commit()
        return SourcePathMetadata(
            source_id=source_id,
            date_text=parsed.date_text,
            date_precision=parsed.date_precision,
            date_source=parsed.date_source,
            date_conflict=parsed.date_conflict,
            filename_date=parsed.filename_date,
            directory_date=parsed.directory_date,
            exif_date=parsed.exif_date,
            path_tags=parsed.path_tags,
            indexed_at=indexed_at,
        )

    @_locked
    def get_source_path_metadata(self, source_id: int) -> SourcePathMetadata | None:
        row = self.conn.execute(
            "SELECT * FROM source_path_metadata WHERE source_id = ?", (source_id,)
        ).fetchone()
        if row is None:
            return None
        return SourcePathMetadata(
            source_id=row["source_id"],
            date_text=row["date_text"],
            date_precision=cast(DatePrecision, row["date_precision"]),
            date_source=cast(DateSource, row["date_source"]),
            date_conflict=bool(row["date_conflict"]),
            filename_date=row["filename_date"],
            directory_date=row["directory_date"],
            exif_date=row["exif_date"],
            path_tags=self._decode_path_tags(row["path_tags"]),
            indexed_at=row["indexed_at"],
        )

    @_locked
    def backfill_source_path_metadata(self, source_type: str | None = None) -> int:
        if source_type:
            rows = self.conn.execute(
                """
                SELECT s.id FROM sources s
                LEFT JOIN source_path_metadata spm ON spm.source_id = s.id
                WHERE s.type = ? AND spm.source_id IS NULL
                ORDER BY s.id
                """,
                (source_type,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT s.id FROM sources s
                LEFT JOIN source_path_metadata spm ON spm.source_id = s.id
                WHERE spm.source_id IS NULL
                ORDER BY s.id
                """
            ).fetchall()
        count = 0
        with self.transaction():
            for row in rows:
                if self.upsert_source_path_metadata(int(row["id"])) is not None:
                    count += 1
        return count

    @_locked
    def get_all_path_tags(self, limit: int | None = None) -> list[tuple[str, int]]:
        rows = self.conn.execute("SELECT path_tags FROM source_path_metadata").fetchall()
        counts: dict[str, int] = {}
        for row in rows:
            for tag in self._decode_path_tags(row["path_tags"]):
                counts[tag] = counts.get(tag, 0) + 1
        result = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return result[:limit] if limit is not None else result
