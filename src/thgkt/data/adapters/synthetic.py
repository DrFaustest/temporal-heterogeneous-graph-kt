"""Deterministic synthetic toy dataset adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from thgkt.data.bundles import make_canonical_bundle
from thgkt.data.summaries import summarize_bundle
from thgkt.schemas.canonical import CanonicalBundle
from thgkt.schemas.validators import ValidationReport, validate_bundle


@dataclass(frozen=True, slots=True)
class SyntheticToyConfig:
    num_students: int = 4
    num_questions: int = 6
    num_concepts: int = 3
    interactions_per_student: int = 4
    source_dataset: str = "synthetic_toy"


class SyntheticToyAdapter:
    """Generate a deterministic, smoke-test-friendly canonical bundle."""

    def __init__(self, config: SyntheticToyConfig | None = None) -> None:
        self.config = config or SyntheticToyConfig()

    def load_raw(self, raw_path: str | None = None) -> dict[str, Any]:
        return {"config": self.config, "raw_path": raw_path}

    def to_canonical(self, raw_obj: Any | None = None) -> CanonicalBundle:
        config = self.config
        questions = [
            {
                "question_id": f"q-{question_idx}",
                "source_dataset": config.source_dataset,
            }
            for question_idx in range(config.num_questions)
        ]
        concepts = [
            {
                "concept_id": f"c-{concept_idx}",
                "source_dataset": config.source_dataset,
            }
            for concept_idx in range(config.num_concepts)
        ]

        question_concept_map: list[dict[str, Any]] = []
        for question_idx in range(config.num_questions):
            question_id = f"q-{question_idx}"
            primary = f"c-{question_idx % config.num_concepts}"
            secondary = f"c-{(question_idx + 1) % config.num_concepts}"
            for concept_id in (primary, secondary):
                pair = {"question_id": question_id, "concept_id": concept_id}
                if pair not in question_concept_map:
                    question_concept_map.append(pair)

        concept_relations = [
            {
                "source_concept_id": f"c-{concept_idx}",
                "target_concept_id": f"c-{concept_idx + 1}",
                "relation_type": "prerequisite_of",
            }
            for concept_idx in range(config.num_concepts - 1)
        ]

        interactions: list[dict[str, Any]] = []
        for student_idx in range(config.num_students):
            student_id = f"s-{student_idx}"
            for seq_idx in range(config.interactions_per_student):
                question_idx = (student_idx + seq_idx) % config.num_questions
                question_id = f"q-{question_idx}"
                concept_ids = sorted(
                    {
                        mapping["concept_id"]
                        for mapping in question_concept_map
                        if mapping["question_id"] == question_id
                    }
                )
                interactions.append(
                    {
                        "student_id": student_id,
                        "interaction_id": f"{student_id}-i-{seq_idx}",
                        "seq_idx": seq_idx,
                        "timestamp": f"2024-01-{student_idx + 1:02d}T00:{seq_idx:02d}:00",
                        "question_id": question_id,
                        "correct": int((student_idx + seq_idx) % 2 == 0),
                        "concept_ids": concept_ids,
                        "elapsed_time": float(8 + seq_idx),
                        "attempt_count": 1 + (seq_idx % 2),
                        "source_dataset": config.source_dataset,
                    }
                )

        bundle = make_canonical_bundle(
            interactions=interactions,
            questions=questions,
            concepts=concepts,
            question_concept_map=question_concept_map,
            concept_relations=concept_relations,
            metadata={
                "dataset_name": "synthetic_toy",
                "source_dataset": config.source_dataset,
                "generator": "SyntheticToyAdapter",
            },
        )
        summary = summarize_bundle(bundle)
        bundle.metadata["summary"] = summary.to_dict()
        return bundle

    def validate_canonical(self, bundle: CanonicalBundle) -> ValidationReport:
        return validate_bundle(bundle)
