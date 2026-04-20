"""Pure helpers used across the HTTP layer.

These were all closure-local inside ``create_app`` before M1. Pulling them
out lets the per-aggregate routers share them without either re-defining
them or importing from a sibling router.

Kinds and species
-----------------
Per ADR-012 M0.5 the HTTP layer speaks only two vocabularies:

- URL kind ``"people"`` / ``"pets"`` — what shows up in paths like ``/{kind}/…``
- Finding species ``"human"`` / ``"dog"`` / ``"cat"`` / ``"other_pet"``
  — the DB value on ``findings.species`` (a legitimate DB→HTTP crossing value)

The singular subject kind (``"person"`` / ``"pet"``) lives inside the DB and
domain layers; HTTP code never holds it. When HTTP needs a plural URL kind
from a subject row, it inlines the one-line mapping rather than importing a
named helper.

``db/paths.py`` owns ``_species_for_kind`` and ``_kind_for_species`` for
domain/DB consumers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

KindType = Literal["people", "pets"]

# URL kind -> finding species (for finding-level filtering)
_KIND_TO_SPECIES: dict[str, str] = {"people": "human", "pets": "pet"}


def species_for_kind(kind: KindType) -> str:
    """Map URL kind to finding species for finding-level queries."""
    return _KIND_TO_SPECIES[kind]


def kind_for_species(species: str) -> KindType:
    """Map a finding species string to the URL kind."""
    if species in ("pet", "cat", "dog"):
        return "pets"
    return "people"


_DATE_RE = re.compile(r"(\d{4})-(\d{2})")


def month_from_path(file_path: str) -> str:
    """Extract YYYY-MM from directory names in a source path (most reliable date source)."""
    for part in reversed(Path(file_path).parts):
        m = _DATE_RE.search(part)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
    return "Unknown"


def group_by_month(
    items: Sequence[tuple[object, str]], key: str = "items"
) -> list[dict[str, str | list[object]]]:
    """Group (item, file_path) pairs by month extracted from path."""
    groups: list[dict[str, str | list[object]]] = []
    current_month = None
    for item, path in items:
        month = month_from_path(path)
        if month != current_month:
            current_month = month
            groups.append({"month": month, key: []})
        last = groups[-1][key]
        assert isinstance(last, list)
        last.append(item)
    return groups


# ── Undo-toast description builders ───────────────────────────────────────


def _noun(n: int) -> str:
    return "face" if n == 1 else "faces"


def describe_cluster_dismiss(cluster_id: int, n: int) -> str:
    return f"Dismissed {n} {_noun(n)} in cluster #{cluster_id}"


def describe_cluster_merge(source_id: int, target_id: int, n: int) -> str:
    return f"Merged cluster #{source_id} into #{target_id} ({n} {_noun(n)})"


def describe_cluster_assign(subject_name: str, n: int) -> str:
    return f"Assigned {n} {_noun(n)} to {subject_name}"


def describe_cluster_name(subject_name: str, n: int) -> str:
    return f"Created {subject_name} and assigned {n} {_noun(n)}"


def describe_subject_delete(subject_name: str, n: int) -> str:
    return f"Deleted {subject_name} ({n} {_noun(n)} unassigned)"


def describe_subject_merge(source_name: str, target_name: str, n: int) -> str:
    return f"Merged {source_name} into {target_name} ({n} {_noun(n)})"


def describe_findings_dismiss(n: int) -> str:
    return f"Dismissed {n} {_noun(n)}"


def describe_findings_exclude(cluster_id: int, n: int) -> str:
    return f"Excluded {n} {_noun(n)} from cluster #{cluster_id}"


def describe_findings_reassign(verb: str, subject_name: str, n: int) -> str:
    return f"{verb} {n} {_noun(n)} for {subject_name}"


# ── Misc ──────────────────────────────────────────────────────────────────


def undo_hx_trigger(message: str, token: str) -> dict[str, str]:
    """Build the HX-Trigger header that fires the client-side undo toast."""
    payload = {"undoToast": {"message": message, "token": token}}
    return {"HX-Trigger": json.dumps(payload)}


def normalize_source_type(raw: str) -> str | None:
    """Accept 'photo' / 'video' as filters; anything else (default 'either') → None."""
    return raw if raw in ("photo", "video") else None


# Images are content-addressed by immutable keys: finding_id → bbox →
# source file, or source_id → file path. Source files are read-only per
# design principle #5 ("Respect the archive"), so the rendered bytes for
# a given URL never change. Tell the browser it can cache forever.
IMMUTABLE_IMAGE_HEADERS = {"Cache-Control": "public, max-age=31536000, immutable"}
