"""Helpers for constructing canonical bundles with explicit table schemas."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from thgkt.schemas.canonical import CanonicalBundle, CanonicalTable
from thgkt.schemas.validators import (
    CONCEPTS_REQUIRED,
    CONCEPT_RELATIONS_REQUIRED,
    INTERACTIONS_REQUIRED,
    QUESTION_CONCEPT_MAP_REQUIRED,
    QUESTIONS_REQUIRED,
)


def make_canonical_bundle(
    *,
    interactions: Iterable[Mapping[str, Any]],
    questions: Iterable[Mapping[str, Any]],
    concepts: Iterable[Mapping[str, Any]],
    question_concept_map: Iterable[Mapping[str, Any]],
    concept_relations: Iterable[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> CanonicalBundle:
    return CanonicalBundle(
        interactions=CanonicalTable.from_rows("interactions", interactions, INTERACTIONS_REQUIRED),
        questions=CanonicalTable.from_rows("questions", questions, QUESTIONS_REQUIRED),
        concepts=CanonicalTable.from_rows("concepts", concepts, CONCEPTS_REQUIRED),
        question_concept_map=CanonicalTable.from_rows(
            "question_concept_map", question_concept_map, QUESTION_CONCEPT_MAP_REQUIRED
        ),
        concept_relations=CanonicalTable.from_rows(
            "concept_relations", concept_relations, CONCEPT_RELATIONS_REQUIRED
        ),
        metadata=dict(metadata or {}),
    )
