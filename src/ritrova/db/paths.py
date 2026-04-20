"""Path resolution and species/kind mapping mixin."""

from __future__ import annotations

from pathlib import Path

from ._base import _DBAccessor
from .models import Finding


class PathMixin(_DBAccessor):
    """Path resolution, species/kind mapping, and dimension filtering."""

    PET_SPECIES = ("dog", "cat", "other_pet")
    KIND_TO_SPECIES: dict[str, str] = {"person": "human", "pet": "pet"}

    def resolve_path(self, stored_path: str) -> Path:
        """Resolve a DB-stored path to an absolute filesystem path."""
        if stored_path.startswith("/"):
            return Path(stored_path)
        if ".." in stored_path.split("/"):
            msg = f"Path contains '..': {stored_path}"
            raise ValueError(msg)
        # tmp/ paths are app-generated (video frames, etc.) — relative to DB directory
        if stored_path.startswith("tmp/"):
            return self.db_path.parent / stored_path
        if self.base_dir is not None:
            return self.base_dir / stored_path
        return Path(stored_path)

    def resolve_finding_image(self, finding: Finding) -> Path | None:
        """Resolve the image file to use for a finding's thumbnail.

        Photo findings: the source file.
        Video findings: the extracted frame JPEG.

        Returns ``None`` when the finding is orphaned (its source row is gone) —
        endpoints then 404 alongside their existing ``not resolved.exists()``
        check.
        """
        if finding.frame_path:
            return self.db_path.parent / finding.frame_path
        source = self.get_source(finding.source_id)
        if source:
            return self.resolve_path(source.file_path)
        return None

    def to_relative(self, absolute_path: str) -> str:
        """Convert an absolute path to a relative path (stripping base_dir prefix)."""
        if self.base_dir is None:
            return absolute_path
        try:
            return str(Path(absolute_path).resolve().relative_to(self.base_dir))
        except ValueError:
            return absolute_path

    def species_filter(self, species: str) -> tuple[str, tuple[str, ...]]:
        """Return SQL clause and params for species filtering."""
        if species == "pet":
            placeholders = ",".join("?" * len(self.PET_SPECIES))
            return f"species IN ({placeholders})", self.PET_SPECIES
        return "species = ?", (species,)

    def _species_for_kind(self, kind: str) -> str:
        """Map subject kind to the finding species filter value."""
        return self.KIND_TO_SPECIES[kind]

    def _kind_for_species(self, species: str) -> str:
        """Map a finding species to the matching subject kind (DB-internal).

        Returns singular subject-kind ("person" / "pet"). Callers in the
        domain layer (cluster, services, DB mixins) may use this freely;
        HTTP code must not — it keeps the plural URL kind / species
        vocabulary strictly separate; see ADR-012 M0.5.
        """
        return "pet" if species in self.PET_SPECIES or species == "pet" else "person"

    def _is_species_kind_compatible(self, finding_species: str, subject_kind: str) -> bool:
        """Check if a finding's species is compatible with a subject's kind."""
        if subject_kind == "person":
            return finding_species == "human"
        if subject_kind == "pet":
            return finding_species in self.PET_SPECIES
        return True

    def _dim_filter(self, embedding_dim: int | None) -> tuple[str, tuple[int, ...]]:
        """Return SQL clause and params for embedding dimension filtering."""
        if embedding_dim is None:
            return "1=1", ()
        return "embedding_dim = ?", (embedding_dim,)
