"""Cleaning and filtering for canonical bundles."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from thgkt.data.bundles import make_canonical_bundle
from thgkt.schemas.canonical import CanonicalBundle


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    min_interactions_per_student: int = 3
    drop_interactions_without_concepts: bool = True


class CanonicalPreprocessor:
    """Prepare canonical bundles for safe splitting."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()

    def run(self, bundle: CanonicalBundle) -> CanonicalBundle:
        return self.add_sequence_indices(self.filter(self.normalize(self.clean(bundle))))

    def clean(self, bundle: CanonicalBundle) -> CanonicalBundle:
        cleaned_interactions: list[dict[str, Any]] = []
        valid_question_ids = {row["question_id"] for row in bundle.questions.rows}

        for row in bundle.interactions.rows:
            concept_ids = row.get("concept_ids")
            if self.config.drop_interactions_without_concepts and not concept_ids:
                continue
            if row.get("question_id") not in valid_question_ids:
                continue
            correct = row.get("correct")
            if correct not in (0, 1, False, True):
                continue
            cleaned_interactions.append(dict(row))

        return make_canonical_bundle(
            interactions=cleaned_interactions,
            questions=bundle.questions.rows,
            concepts=bundle.concepts.rows,
            question_concept_map=bundle.question_concept_map.rows,
            concept_relations=bundle.concept_relations.rows,
            metadata={**bundle.metadata, "preprocessing": {"cleaned": True}},
        )

    def normalize(self, bundle: CanonicalBundle) -> CanonicalBundle:
        normalized_interactions: list[dict[str, Any]] = []
        question_to_concepts: dict[str, set[str]] = defaultdict(set)

        for row in bundle.interactions.rows:
            concept_ids = sorted({str(concept_id) for concept_id in row["concept_ids"]})
            normalized = {
                **dict(row),
                "student_id": str(row["student_id"]),
                "interaction_id": str(row["interaction_id"]),
                "question_id": str(row["question_id"]),
                "seq_idx": int(row["seq_idx"]),
                "correct": int(row["correct"]),
                "concept_ids": concept_ids,
                "elapsed_time": float(row["elapsed_time"]),
                "attempt_count": int(row["attempt_count"]),
                "source_dataset": str(row["source_dataset"]),
            }
            normalized_interactions.append(normalized)
            for concept_id in concept_ids:
                question_to_concepts[normalized["question_id"]].add(concept_id)

        normalized_map = [
            {"question_id": question_id, "concept_id": concept_id}
            for question_id, concept_ids in sorted(question_to_concepts.items())
            for concept_id in sorted(concept_ids)
        ]
        normalized_concepts = [
            {
                "concept_id": concept_row["concept_id"],
                "source_dataset": concept_row["source_dataset"],
            }
            for concept_row in bundle.concepts.rows
            if any(concept_row["concept_id"] in row["concept_ids"] for row in normalized_interactions)
        ]

        return make_canonical_bundle(
            interactions=normalized_interactions,
            questions=bundle.questions.rows,
            concepts=normalized_concepts,
            question_concept_map=normalized_map,
            concept_relations=bundle.concept_relations.rows,
            metadata={**bundle.metadata, "preprocessing": {"normalized": True}},
        )

    def filter(self, bundle: CanonicalBundle) -> CanonicalBundle:
        student_counts = Counter(str(row["student_id"]) for row in bundle.interactions.rows)
        kept_students = {
            student_id
            for student_id, count in student_counts.items()
            if count >= self.config.min_interactions_per_student
        }
        filtered_interactions = [
            dict(row)
            for row in bundle.interactions.rows
            if str(row["student_id"]) in kept_students
        ]
        kept_question_ids = {row["question_id"] for row in filtered_interactions}
        kept_concept_ids = {concept for row in filtered_interactions for concept in row["concept_ids"]}

        filtered_questions = [
            dict(row) for row in bundle.questions.rows if row["question_id"] in kept_question_ids
        ]
        filtered_concepts = [
            dict(row) for row in bundle.concepts.rows if row["concept_id"] in kept_concept_ids
        ]
        filtered_map = [
            dict(row)
            for row in bundle.question_concept_map.rows
            if row["question_id"] in kept_question_ids and row["concept_id"] in kept_concept_ids
        ]
        filtered_relations = [
            dict(row)
            for row in bundle.concept_relations.rows
            if row["source_concept_id"] in kept_concept_ids
            and row["target_concept_id"] in kept_concept_ids
        ]

        return make_canonical_bundle(
            interactions=filtered_interactions,
            questions=filtered_questions,
            concepts=filtered_concepts,
            question_concept_map=filtered_map,
            concept_relations=filtered_relations,
            metadata={
                **bundle.metadata,
                "preprocessing": {
                    "filtered": True,
                    "min_interactions_per_student": self.config.min_interactions_per_student,
                },
            },
        )

    def add_sequence_indices(self, bundle: CanonicalBundle) -> CanonicalBundle:
        by_student: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in bundle.interactions.rows:
            by_student[str(row["student_id"])] .append(dict(row))

        reindexed: list[dict[str, Any]] = []
        for student_id, rows in by_student.items():
            rows_sorted = sorted(
                rows,
                key=lambda row: (str(row["timestamp"]), int(row.get("seq_idx", 0)), str(row["interaction_id"])),
            )
            for seq_idx, row in enumerate(rows_sorted):
                row["student_id"] = student_id
                row["seq_idx"] = seq_idx
                reindexed.append(row)

        reindexed.sort(key=lambda row: (str(row["student_id"]), int(row["seq_idx"])))
        return make_canonical_bundle(
            interactions=reindexed,
            questions=bundle.questions.rows,
            concepts=bundle.concepts.rows,
            question_concept_map=bundle.question_concept_map.rows,
            concept_relations=bundle.concept_relations.rows,
            metadata={**bundle.metadata, "preprocessing": {"reindexed": True}},
        )
