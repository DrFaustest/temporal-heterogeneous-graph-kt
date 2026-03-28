"""Protocol interfaces for the data pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from thgkt.data.artifacts import SplitArtifacts
from thgkt.schemas.canonical import CanonicalBundle
from thgkt.schemas.validators import ValidationReport


class BaseDatasetAdapter(Protocol):
    def load_raw(self, raw_path: str | Path) -> Any:
        """Load raw dataset assets from disk."""

    def to_canonical(self, raw_obj: Any) -> CanonicalBundle:
        """Convert raw inputs into the canonical bundle."""

    def validate_canonical(self, bundle: CanonicalBundle) -> ValidationReport:
        """Validate the canonical bundle."""


class DatasetPreprocessor(Protocol):
    def clean(self, bundle: CanonicalBundle) -> CanonicalBundle:
        """Drop malformed rows and normalize basic types."""

    def normalize(self, bundle: CanonicalBundle) -> CanonicalBundle:
        """Normalize values without changing dataset semantics."""

    def filter(self, bundle: CanonicalBundle) -> CanonicalBundle:
        """Apply configurable filtering to the canonical bundle."""

    def add_sequence_indices(self, bundle: CanonicalBundle) -> CanonicalBundle:
        """Recompute within-student sequence indices."""


class Splitter(Protocol):
    def make_splits(self, bundle: CanonicalBundle) -> SplitArtifacts:
        """Create serializable train/val/test split artifacts."""
