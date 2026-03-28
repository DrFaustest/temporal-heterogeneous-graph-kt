"""Canonical schema validators."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from numbers import Number
from typing import Any, Iterable

from thgkt.schemas.canonical import CanonicalBundle, CanonicalTable


INTERACTIONS_REQUIRED = (
    "student_id",
    "interaction_id",
    "seq_idx",
    "timestamp",
    "question_id",
    "correct",
    "concept_ids",
    "elapsed_time",
    "attempt_count",
    "source_dataset",
)
QUESTIONS_REQUIRED = ("question_id", "source_dataset")
CONCEPTS_REQUIRED = ("concept_id", "source_dataset")
QUESTION_CONCEPT_MAP_REQUIRED = ("question_id", "concept_id")
CONCEPT_RELATIONS_REQUIRED = ("source_concept_id", "target_concept_id", "relation_type")


class SchemaValidationError(ValueError):
    """Raised when a canonical bundle fails validation."""


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Minimal validation result metadata."""

    row_counts: dict[str, int]


def validate_bundle(bundle: CanonicalBundle) -> ValidationReport:
    """Validate a canonical bundle and raise on failure."""

    _validate_table_shape(bundle.interactions, INTERACTIONS_REQUIRED)
    _validate_table_shape(bundle.questions, QUESTIONS_REQUIRED)
    _validate_table_shape(bundle.concepts, CONCEPTS_REQUIRED)
    _validate_table_shape(bundle.question_concept_map, QUESTION_CONCEPT_MAP_REQUIRED)
    _validate_table_shape(bundle.concept_relations, CONCEPT_RELATIONS_REQUIRED)

    question_ids = _validate_questions(bundle.questions)
    concept_ids = _validate_concepts(bundle.concepts)
    _validate_question_concept_map(bundle.question_concept_map, question_ids, concept_ids)
    _validate_concept_relations(bundle.concept_relations, concept_ids)
    _validate_interactions(bundle.interactions, question_ids, concept_ids)

    return ValidationReport(row_counts=bundle.table_sizes())


def _validate_table_shape(table: CanonicalTable, required: Iterable[str]) -> None:
    required_columns = tuple(required)
    missing = [column for column in required_columns if column not in table.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise SchemaValidationError(
            f"Table '{table.name}' is missing required columns: {missing_text}"
        )

    for idx, row in enumerate(table.rows):
        missing_in_row = [column for column in required_columns if column not in row]
        if missing_in_row:
            missing_text = ", ".join(missing_in_row)
            raise SchemaValidationError(
                f"Table '{table.name}' row {idx} is missing required values for: {missing_text}"
            )


def _validate_questions(table: CanonicalTable) -> set[Any]:
    question_ids: set[Any] = set()
    for idx, row in enumerate(table.rows):
        question_id = row["question_id"]
        source_dataset = row["source_dataset"]
        _require_non_null(question_id, "questions", idx, "question_id")
        _require_non_null(source_dataset, "questions", idx, "source_dataset")
        if question_id in question_ids:
            raise SchemaValidationError(
                f"Table 'questions' contains duplicate question_id: {question_id}"
            )
        question_ids.add(question_id)
    return question_ids


def _validate_concepts(table: CanonicalTable) -> set[Any]:
    concept_ids: set[Any] = set()
    for idx, row in enumerate(table.rows):
        concept_id = row["concept_id"]
        source_dataset = row["source_dataset"]
        _require_non_null(concept_id, "concepts", idx, "concept_id")
        _require_non_null(source_dataset, "concepts", idx, "source_dataset")
        if concept_id in concept_ids:
            raise SchemaValidationError(
                f"Table 'concepts' contains duplicate concept_id: {concept_id}"
            )
        concept_ids.add(concept_id)
    return concept_ids


def _validate_question_concept_map(
    table: CanonicalTable,
    question_ids: set[Any],
    concept_ids: set[Any],
) -> None:
    seen_pairs: set[tuple[Any, Any]] = set()
    for idx, row in enumerate(table.rows):
        question_id = row["question_id"]
        concept_id = row["concept_id"]
        _require_known_id(question_id, question_ids, "question_concept_map", idx, "question_id")
        _require_known_id(concept_id, concept_ids, "question_concept_map", idx, "concept_id")
        pair = (question_id, concept_id)
        if pair in seen_pairs:
            raise SchemaValidationError(
                "Table 'question_concept_map' contains duplicate question_id/concept_id pair: "
                f"{pair}"
            )
        seen_pairs.add(pair)


def _validate_concept_relations(table: CanonicalTable, concept_ids: set[Any]) -> None:
    seen_relations: set[tuple[Any, Any, Any]] = set()
    for idx, row in enumerate(table.rows):
        source = row["source_concept_id"]
        target = row["target_concept_id"]
        relation_type = row["relation_type"]
        _require_known_id(source, concept_ids, "concept_relations", idx, "source_concept_id")
        _require_known_id(target, concept_ids, "concept_relations", idx, "target_concept_id")
        _require_non_null(relation_type, "concept_relations", idx, "relation_type")
        if source == target:
            raise SchemaValidationError(
                f"Table 'concept_relations' row {idx} has a self-loop concept relation: {source}"
            )
        relation_key = (source, target, relation_type)
        if relation_key in seen_relations:
            raise SchemaValidationError(
                "Table 'concept_relations' contains duplicate relation: "
                f"{relation_key}"
            )
        seen_relations.add(relation_key)


def _validate_interactions(
    table: CanonicalTable,
    question_ids: set[Any],
    concept_ids: set[Any],
) -> None:
    interaction_ids: set[Any] = set()
    per_student_seq: dict[Any, set[int]] = defaultdict(set)

    for idx, row in enumerate(table.rows):
        student_id = row["student_id"]
        interaction_id = row["interaction_id"]
        seq_idx = row["seq_idx"]
        timestamp = row["timestamp"]
        question_id = row["question_id"]
        correct = row["correct"]
        interaction_concept_ids = row["concept_ids"]
        elapsed_time = row["elapsed_time"]
        attempt_count = row["attempt_count"]
        source_dataset = row["source_dataset"]

        _require_non_null(student_id, "interactions", idx, "student_id")
        _require_non_null(interaction_id, "interactions", idx, "interaction_id")
        _require_non_null(timestamp, "interactions", idx, "timestamp")
        _require_known_id(question_id, question_ids, "interactions", idx, "question_id")
        _require_non_null(source_dataset, "interactions", idx, "source_dataset")
        _require_int(seq_idx, "interactions", idx, "seq_idx", minimum=0)
        _require_binary(correct, "interactions", idx, "correct")
        _require_number(elapsed_time, "interactions", idx, "elapsed_time", minimum=0)
        _require_int(attempt_count, "interactions", idx, "attempt_count", minimum=0)
        _require_concept_id_list(
            interaction_concept_ids,
            concept_ids,
            "interactions",
            idx,
            "concept_ids",
        )

        if interaction_id in interaction_ids:
            raise SchemaValidationError(
                f"Table 'interactions' contains duplicate interaction_id: {interaction_id}"
            )
        interaction_ids.add(interaction_id)

        if seq_idx in per_student_seq[student_id]:
            raise SchemaValidationError(
                f"Table 'interactions' contains duplicate seq_idx={seq_idx} for student_id={student_id}"
            )
        per_student_seq[student_id].add(seq_idx)


def _require_non_null(value: Any, table: str, row_idx: int, column: str) -> None:
    if value is None or value == "":
        raise SchemaValidationError(
            f"Table '{table}' row {row_idx} has null or empty value for '{column}'"
        )


def _require_known_id(
    value: Any,
    valid_ids: set[Any],
    table: str,
    row_idx: int,
    column: str,
) -> None:
    _require_non_null(value, table, row_idx, column)
    if value not in valid_ids:
        raise SchemaValidationError(
            f"Table '{table}' row {row_idx} references unknown {column}: {value}"
        )


def _require_int(
    value: Any,
    table: str,
    row_idx: int,
    column: str,
    *,
    minimum: int | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaValidationError(
            f"Table '{table}' row {row_idx} expects integer '{column}', got {value!r}"
        )
    if minimum is not None and value < minimum:
        raise SchemaValidationError(
            f"Table '{table}' row {row_idx} expects '{column}' >= {minimum}, got {value}"
        )


def _require_number(
    value: Any,
    table: str,
    row_idx: int,
    column: str,
    *,
    minimum: float | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, Number):
        raise SchemaValidationError(
            f"Table '{table}' row {row_idx} expects numeric '{column}', got {value!r}"
        )
    if minimum is not None and float(value) < minimum:
        raise SchemaValidationError(
            f"Table '{table}' row {row_idx} expects '{column}' >= {minimum}, got {value}"
        )


def _require_binary(value: Any, table: str, row_idx: int, column: str) -> None:
    if value not in (0, 1, False, True):
        raise SchemaValidationError(
            f"Table '{table}' row {row_idx} expects binary '{column}', got {value!r}"
        )


def _require_concept_id_list(
    value: Any,
    valid_ids: set[Any],
    table: str,
    row_idx: int,
    column: str,
) -> None:
    if not isinstance(value, (list, tuple)):
        raise SchemaValidationError(
            f"Table '{table}' row {row_idx} expects '{column}' to be a list or tuple, got {value!r}"
        )
    if len(value) == 0:
        raise SchemaValidationError(
            f"Table '{table}' row {row_idx} expects non-empty '{column}'"
        )
    for concept_id in value:
        if concept_id not in valid_ids:
            raise SchemaValidationError(
                f"Table '{table}' row {row_idx} references unknown concept_id in '{column}': {concept_id}"
            )
