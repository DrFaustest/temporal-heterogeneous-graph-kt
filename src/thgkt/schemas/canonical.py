"""Canonical schema data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


Row = dict[str, Any]


@dataclass(frozen=True, slots=True)
class CanonicalTable:
    """A light-weight table representation for schema validation."""

    name: str
    columns: tuple[str, ...]
    rows: tuple[Row, ...]

    @classmethod
    def from_rows(
        cls,
        name: str,
        rows: Iterable[Mapping[str, Any]],
        columns: Iterable[str] | None = None,
    ) -> "CanonicalTable":
        materialized_rows = tuple(dict(row) for row in rows)
        inferred_columns: tuple[str, ...]
        if columns is not None:
            inferred_columns = tuple(columns)
        else:
            ordered_columns: list[str] = []
            seen: set[str] = set()
            for row in materialized_rows:
                for key in row:
                    if key not in seen:
                        seen.add(key)
                        ordered_columns.append(key)
            inferred_columns = tuple(ordered_columns)

        return cls(name=name, columns=inferred_columns, rows=materialized_rows)

    def __len__(self) -> int:
        return len(self.rows)


@dataclass(frozen=True, slots=True)
class CanonicalBundle:
    """The canonical cross-dataset schema used by the project."""

    interactions: CanonicalTable
    questions: CanonicalTable
    concepts: CanonicalTable
    question_concept_map: CanonicalTable
    concept_relations: CanonicalTable
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(
        cls,
        *,
        interactions: Iterable[Mapping[str, Any]],
        questions: Iterable[Mapping[str, Any]],
        concepts: Iterable[Mapping[str, Any]],
        question_concept_map: Iterable[Mapping[str, Any]],
        concept_relations: Iterable[Mapping[str, Any]],
        metadata: Mapping[str, Any] | None = None,
    ) -> "CanonicalBundle":
        return cls(
            interactions=CanonicalTable.from_rows("interactions", interactions),
            questions=CanonicalTable.from_rows("questions", questions),
            concepts=CanonicalTable.from_rows("concepts", concepts),
            question_concept_map=CanonicalTable.from_rows(
                "question_concept_map", question_concept_map
            ),
            concept_relations=CanonicalTable.from_rows(
                "concept_relations", concept_relations
            ),
            metadata=dict(metadata or {}),
        )

    def table_sizes(self) -> dict[str, int]:
        return {
            "interactions": len(self.interactions),
            "questions": len(self.questions),
            "concepts": len(self.concepts),
            "question_concept_map": len(self.question_concept_map),
            "concept_relations": len(self.concept_relations),
        }
