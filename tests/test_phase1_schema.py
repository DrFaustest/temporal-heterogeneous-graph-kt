from __future__ import annotations

import pytest

from thgkt.schemas.canonical import CanonicalBundle
from thgkt.schemas.validators import SchemaValidationError, validate_bundle


def _valid_bundle() -> CanonicalBundle:
    return CanonicalBundle.from_raw(
        interactions=[
            {
                "student_id": "stu-1",
                "interaction_id": "i-1",
                "seq_idx": 0,
                "timestamp": "2024-01-01T10:00:00",
                "question_id": "q-1",
                "correct": 1,
                "concept_ids": ["c-1", "c-2"],
                "elapsed_time": 12.5,
                "attempt_count": 1,
                "source_dataset": "assistments",
            },
            {
                "student_id": "stu-1",
                "interaction_id": "i-2",
                "seq_idx": 1,
                "timestamp": "2024-01-01T10:05:00",
                "question_id": "q-2",
                "correct": 0,
                "concept_ids": ["c-2"],
                "elapsed_time": 9.0,
                "attempt_count": 2,
                "source_dataset": "assistments",
            },
        ],
        questions=[
            {"question_id": "q-1", "source_dataset": "assistments"},
            {"question_id": "q-2", "source_dataset": "assistments"},
        ],
        concepts=[
            {"concept_id": "c-1", "source_dataset": "assistments"},
            {"concept_id": "c-2", "source_dataset": "assistments"},
        ],
        question_concept_map=[
            {"question_id": "q-1", "concept_id": "c-1"},
            {"question_id": "q-1", "concept_id": "c-2"},
            {"question_id": "q-2", "concept_id": "c-2"},
        ],
        concept_relations=[
            {
                "source_concept_id": "c-1",
                "target_concept_id": "c-2",
                "relation_type": "prerequisite_of",
            }
        ],
        metadata={"dataset_name": "assistments-toy"},
    )


def test_valid_canonical_bundle_passes_validation() -> None:
    bundle = _valid_bundle()

    report = validate_bundle(bundle)

    assert report.row_counts["interactions"] == 2
    assert report.row_counts["question_concept_map"] == 3


def test_missing_required_interactions_column_fails_clearly() -> None:
    bundle = _valid_bundle()
    broken = CanonicalBundle.from_raw(
        interactions=[
            {key: value for key, value in bundle.interactions.rows[0].items() if key != "correct"}
        ],
        questions=bundle.questions.rows,
        concepts=bundle.concepts.rows,
        question_concept_map=bundle.question_concept_map.rows,
        concept_relations=bundle.concept_relations.rows,
    )

    with pytest.raises(
        SchemaValidationError,
        match=r"Table 'interactions' is missing required columns: correct",
    ):
        validate_bundle(broken)


def test_foreign_key_violation_fails_clearly() -> None:
    bundle = CanonicalBundle.from_raw(
        interactions=[
            {
                "student_id": "stu-1",
                "interaction_id": "i-1",
                "seq_idx": 0,
                "timestamp": "2024-01-01T10:00:00",
                "question_id": "q-1",
                "correct": 1,
                "concept_ids": ["c-404"],
                "elapsed_time": 5.0,
                "attempt_count": 1,
                "source_dataset": "assistments",
            }
        ],
        questions=[{"question_id": "q-1", "source_dataset": "assistments"}],
        concepts=[{"concept_id": "c-1", "source_dataset": "assistments"}],
        question_concept_map=[{"question_id": "q-1", "concept_id": "c-1"}],
        concept_relations=[
            {
                "source_concept_id": "c-1",
                "target_concept_id": "c-1",
                "relation_type": "prerequisite_of",
            }
        ],
    )

    with pytest.raises(
        SchemaValidationError,
        match=r"Table 'concept_relations' row 0 has a self-loop concept relation: c-1",
    ):
        validate_bundle(bundle)


def test_invalid_binary_target_fails_clearly() -> None:
    bundle = _valid_bundle()
    broken = CanonicalBundle.from_raw(
        interactions=[
            {
                **bundle.interactions.rows[0],
                "correct": 2,
            }
        ],
        questions=bundle.questions.rows,
        concepts=bundle.concepts.rows,
        question_concept_map=bundle.question_concept_map.rows,
        concept_relations=bundle.concept_relations.rows,
    )

    with pytest.raises(
        SchemaValidationError,
        match=r"Table 'interactions' row 0 expects binary 'correct', got 2",
    ):
        validate_bundle(broken)
