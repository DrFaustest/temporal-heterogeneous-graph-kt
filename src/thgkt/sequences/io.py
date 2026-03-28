"""Sequence artifact serialization."""

from __future__ import annotations

from pathlib import Path

from thgkt.data.io import load_json, save_json
from thgkt.sequences.artifacts import SequenceArtifacts


def save_sequence_artifacts(artifacts: SequenceArtifacts, path: str | Path) -> Path:
    return save_json(artifacts.to_dict(), path)


def load_sequence_artifacts(path: str | Path) -> SequenceArtifacts:
    return SequenceArtifacts.from_dict(load_json(path))
