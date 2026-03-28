"""Artifact serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from thgkt.data.artifacts import SplitArtifacts
from thgkt.data.bundles import make_canonical_bundle
from thgkt.schemas.canonical import CanonicalBundle


def _table_to_dict(table: Any) -> dict[str, Any]:
    return {
        "name": table.name,
        "columns": list(table.columns),
        "rows": [dict(row) for row in table.rows],
    }


def save_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_canonical_bundle(bundle: CanonicalBundle, path: str | Path) -> Path:
    payload = {
        "interactions": _table_to_dict(bundle.interactions),
        "questions": _table_to_dict(bundle.questions),
        "concepts": _table_to_dict(bundle.concepts),
        "question_concept_map": _table_to_dict(bundle.question_concept_map),
        "concept_relations": _table_to_dict(bundle.concept_relations),
        "metadata": dict(bundle.metadata),
    }
    return save_json(payload, path)


def load_canonical_bundle(path: str | Path) -> CanonicalBundle:
    payload = load_json(path)
    return make_canonical_bundle(
        interactions=payload["interactions"]["rows"],
        questions=payload["questions"]["rows"],
        concepts=payload["concepts"]["rows"],
        question_concept_map=payload["question_concept_map"]["rows"],
        concept_relations=payload["concept_relations"]["rows"],
        metadata=payload.get("metadata", {}),
    )


def save_split_artifacts(artifacts: SplitArtifacts, path: str | Path) -> Path:
    return save_json(artifacts.to_dict(), path)


def load_split_artifacts(path: str | Path) -> SplitArtifacts:
    return SplitArtifacts.from_dict(load_json(path))
