"""Concept relation generation variants."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from thgkt.data.bundles import make_canonical_bundle
from thgkt.schemas.canonical import CanonicalBundle


@dataclass(frozen=True, slots=True)
class RelationConfig:
    mode: str = "none"
    min_weight: int = 1


def apply_concept_relation_mode(
    bundle: CanonicalBundle,
    config: RelationConfig | None = None,
) -> CanonicalBundle:
    relation_config = config or RelationConfig()
    relations = generate_concept_relations(bundle, relation_config)
    return make_canonical_bundle(
        interactions=bundle.interactions.rows,
        questions=bundle.questions.rows,
        concepts=bundle.concepts.rows,
        question_concept_map=bundle.question_concept_map.rows,
        concept_relations=relations,
        metadata={
            **bundle.metadata,
            "relation_generation": {
                "mode": relation_config.mode,
                "min_weight": relation_config.min_weight,
                "num_relations": len(relations),
            },
        },
    )


def generate_concept_relations(
    bundle: CanonicalBundle,
    config: RelationConfig | None = None,
) -> list[dict[str, Any]]:
    relation_config = config or RelationConfig()
    if relation_config.mode == "none":
        return []
    if relation_config.mode == "cooccurrence":
        return _cooccurrence_relations(bundle, relation_config.min_weight)
    if relation_config.mode == "transition":
        return _transition_relations(bundle, relation_config.min_weight)
    raise ValueError(f"Unsupported relation generation mode: {relation_config.mode}")


def _cooccurrence_relations(bundle: CanonicalBundle, min_weight: int) -> list[dict[str, Any]]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    for row in bundle.interactions.rows:
        concepts = sorted({str(item) for item in row["concept_ids"]})
        for left_index, source in enumerate(concepts):
            for target in concepts[left_index + 1 :]:
                pair_counts[(source, target)] += 1
                pair_counts[(target, source)] += 1
    return [
        {
            "source_concept_id": source,
            "target_concept_id": target,
            "relation_type": "cooccurs_with",
        }
        for (source, target), weight in sorted(pair_counts.items())
        if weight >= min_weight and source != target
    ]


def _transition_relations(bundle: CanonicalBundle, min_weight: int) -> list[dict[str, Any]]:
    per_student: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bundle.interactions.rows:
        per_student[str(row["student_id"])].append(dict(row))

    transition_counts: Counter[tuple[str, str]] = Counter()
    for rows in per_student.values():
        ordered = sorted(rows, key=lambda item: int(item["seq_idx"]))
        for left_row, right_row in zip(ordered, ordered[1:]):
            left_concepts = {str(item) for item in left_row["concept_ids"]}
            right_concepts = {str(item) for item in right_row["concept_ids"]}
            for source in left_concepts:
                for target in right_concepts:
                    if source != target:
                        transition_counts[(source, target)] += 1

    return [
        {
            "source_concept_id": source,
            "target_concept_id": target,
            "relation_type": "transition_to",
        }
        for (source, target), weight in sorted(transition_counts.items())
        if weight >= min_weight
    ]
