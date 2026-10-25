"""Whether the optional model runtimes are installed.

The server image installs the base dependencies only: detection and
embedding live behind the ``scan`` extra, which pulls torch, insightface,
ultralytics and transformers — around 2 GB onto a box that only serves
pages. Everything the web UI does apart from drawing a face by hand works
without them, clustering included, because that reads embeddings already
in the database rather than producing new ones.

Code that needs a model asks here first, so a base install refuses with a
sentence the reader can act on instead of a ``ModuleNotFoundError``.
"""

from __future__ import annotations

from functools import cache
from importlib.util import find_spec

INSTALL_HINT = "Install the model runtimes with `uv sync --extra scan`."

_FACE_MODULES = ("insightface", "onnxruntime")
_PET_MODULES = ("torch", "ultralytics", "transformers")


def _installed(modules: tuple[str, ...]) -> bool:
    return all(find_spec(m) is not None for m in modules)


@cache
def face_models_available() -> bool:
    return _installed(_FACE_MODULES)


@cache
def pet_models_available() -> bool:
    return _installed(_PET_MODULES)


@cache
def scan_available() -> bool:
    """True when every runtime ``ritrova analyse`` needs is importable."""
    return face_models_available() and pet_models_available()


def models_for_species(species: str) -> bool:
    return face_models_available() if species == "human" else pet_models_available()
