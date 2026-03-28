"""Summary statistics for canonical bundles."""

from __future__ import annotations

from collections import Counter

from thgkt.data.artifacts import DatasetSummary
from thgkt.schemas.canonical import CanonicalBundle


def summarize_bundle(bundle: CanonicalBundle) -> DatasetSummary:
    interactions = bundle.interactions.rows
    student_counts = Counter(str(row["student_id"]) for row in interactions)
    num_students = len(student_counts)
    num_interactions = len(interactions)
    avg_per_student = (
        float(num_interactions) / float(num_students) if num_students > 0 else 0.0
    )
    missing_concepts = sum(1 for row in interactions if not row.get("concept_ids"))

    dataset_name = str(bundle.metadata.get("dataset_name", "unknown"))
    source_dataset = str(bundle.metadata.get("source_dataset", dataset_name))

    return DatasetSummary(
        dataset_name=dataset_name,
        source_dataset=source_dataset,
        num_interactions=num_interactions,
        num_students=num_students,
        num_questions=len(bundle.questions),
        num_concepts=len(bundle.concepts),
        num_question_concept_links=len(bundle.question_concept_map),
        num_concept_relations=len(bundle.concept_relations),
        avg_interactions_per_student=avg_per_student,
        missing_concept_interactions=missing_concepts,
        extra={"student_interaction_counts": dict(sorted(student_counts.items()))},
    )
