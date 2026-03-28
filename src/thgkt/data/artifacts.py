"""Serializable data artifact types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """Light-weight summary statistics for a canonical bundle."""

    dataset_name: str
    source_dataset: str
    num_interactions: int
    num_students: int
    num_questions: int
    num_concepts: int
    num_question_concept_links: int
    num_concept_relations: int
    avg_interactions_per_student: float
    missing_concept_interactions: int
    created_at: str = field(default_factory=utc_now_iso)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "source_dataset": self.source_dataset,
            "num_interactions": self.num_interactions,
            "num_students": self.num_students,
            "num_questions": self.num_questions,
            "num_concepts": self.num_concepts,
            "num_question_concept_links": self.num_question_concept_links,
            "num_concept_relations": self.num_concept_relations,
            "avg_interactions_per_student": self.avg_interactions_per_student,
            "missing_concept_interactions": self.missing_concept_interactions,
            "created_at": self.created_at,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatasetSummary":
        return cls(
            dataset_name=str(payload["dataset_name"]),
            source_dataset=str(payload["source_dataset"]),
            num_interactions=int(payload["num_interactions"]),
            num_students=int(payload["num_students"]),
            num_questions=int(payload["num_questions"]),
            num_concepts=int(payload["num_concepts"]),
            num_question_concept_links=int(payload["num_question_concept_links"]),
            num_concept_relations=int(payload["num_concept_relations"]),
            avg_interactions_per_student=float(payload["avg_interactions_per_student"]),
            missing_concept_interactions=int(payload["missing_concept_interactions"]),
            created_at=str(payload.get("created_at", utc_now_iso())),
            extra=dict(payload.get("extra", {})),
        )


@dataclass(frozen=True, slots=True)
class SplitIndices:
    """Interaction IDs and student IDs for a split."""

    name: str
    interaction_ids: tuple[str, ...]
    student_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "interaction_ids": list(self.interaction_ids),
            "student_ids": list(self.student_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SplitIndices":
        return cls(
            name=str(payload["name"]),
            interaction_ids=tuple(str(item) for item in payload["interaction_ids"]),
            student_ids=tuple(str(item) for item in payload["student_ids"]),
        )


@dataclass(frozen=True, slots=True)
class SplitArtifacts:
    """Serialized split definition and metadata."""

    split_strategy: str
    random_seed: int
    train: SplitIndices
    val: SplitIndices
    test: SplitIndices
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_strategy": self.split_strategy,
            "random_seed": self.random_seed,
            "train": self.train.to_dict(),
            "val": self.val.to_dict(),
            "test": self.test.to_dict(),
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SplitArtifacts":
        return cls(
            split_strategy=str(payload["split_strategy"]),
            random_seed=int(payload["random_seed"]),
            train=SplitIndices.from_dict(payload["train"]),
            val=SplitIndices.from_dict(payload["val"]),
            test=SplitIndices.from_dict(payload["test"]),
            created_at=str(payload.get("created_at", utc_now_iso())),
            metadata=dict(payload.get("metadata", {})),
        )
