"""Graph artifact serialization."""

from __future__ import annotations

from pathlib import Path

from thgkt.data.io import load_json, save_json
from thgkt.graph.artifacts import HeteroGraphArtifacts


def save_graph_artifacts(graph: HeteroGraphArtifacts, path: str | Path) -> Path:
    return save_json(graph.to_dict(), path)


def load_graph_artifacts(path: str | Path) -> HeteroGraphArtifacts:
    return HeteroGraphArtifacts.from_dict(load_json(path))
