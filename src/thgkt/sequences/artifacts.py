"""Serializable sequence artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class StudentSequenceHistory:
    student_id: str
    interaction_ids: tuple[str, ...]
    question_indices: tuple[int, ...]
    correctness: tuple[int, ...]
    elapsed_times: tuple[float, ...]
    attempt_counts: tuple[int, ...]
    concept_indices: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "student_id": self.student_id,
            "interaction_ids": list(self.interaction_ids),
            "question_indices": list(self.question_indices),
            "correctness": list(self.correctness),
            "elapsed_times": list(self.elapsed_times),
            "attempt_counts": list(self.attempt_counts),
            "concept_indices": [list(items) for items in self.concept_indices],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StudentSequenceHistory":
        return cls(
            student_id=str(payload["student_id"]),
            interaction_ids=tuple(str(item) for item in payload["interaction_ids"]),
            question_indices=tuple(int(item) for item in payload["question_indices"]),
            correctness=tuple(int(item) for item in payload["correctness"]),
            elapsed_times=tuple(float(item) for item in payload["elapsed_times"]),
            attempt_counts=tuple(int(item) for item in payload["attempt_counts"]),
            concept_indices=tuple(
                tuple(int(concept_index) for concept_index in concept_indices)
                for concept_indices in payload["concept_indices"]
            ),
        )


@dataclass(frozen=True, slots=True)
class SequenceExample:
    split_name: str
    student_id: str
    target_interaction_id: str
    history_start: int
    target_position: int
    target_question_index: int
    target_concept_indices: tuple[int, ...]
    target_correct: int
    history_source: StudentSequenceHistory = field(repr=False, compare=False)

    @property
    def history_interaction_ids(self) -> tuple[str, ...]:
        return self.history_source.interaction_ids[self.history_start : self.target_position]

    @property
    def history_question_indices(self) -> tuple[int, ...]:
        return self.history_source.question_indices[self.history_start : self.target_position]

    @property
    def history_correctness(self) -> tuple[int, ...]:
        return self.history_source.correctness[self.history_start : self.target_position]

    @property
    def history_elapsed_times(self) -> tuple[float, ...]:
        return self.history_source.elapsed_times[self.history_start : self.target_position]

    @property
    def history_attempt_counts(self) -> tuple[int, ...]:
        return self.history_source.attempt_counts[self.history_start : self.target_position]

    @property
    def history_concept_indices(self) -> tuple[tuple[int, ...], ...]:
        return self.history_source.concept_indices[self.history_start : self.target_position]

    @property
    def history_length(self) -> int:
        return max(0, self.target_position - self.history_start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_name": self.split_name,
            "student_id": self.student_id,
            "target_interaction_id": self.target_interaction_id,
            "history_start": self.history_start,
            "target_position": self.target_position,
            "target_question_index": self.target_question_index,
            "target_concept_indices": list(self.target_concept_indices),
            "target_correct": self.target_correct,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        histories: Mapping[str, StudentSequenceHistory],
    ) -> "SequenceExample":
        student_id = str(payload["student_id"])
        history_source = histories[student_id]
        if "target_position" in payload and "history_start" in payload:
            return cls(
                split_name=str(payload["split_name"]),
                student_id=student_id,
                target_interaction_id=str(payload["target_interaction_id"]),
                history_start=int(payload["history_start"]),
                target_position=int(payload["target_position"]),
                target_question_index=int(payload["target_question_index"]),
                target_concept_indices=tuple(int(item) for item in payload["target_concept_indices"]),
                target_correct=int(payload["target_correct"]),
                history_source=history_source,
            )

        standalone_history = StudentSequenceHistory(
            student_id=student_id,
            interaction_ids=tuple(str(item) for item in payload["history_interaction_ids"]),
            question_indices=tuple(int(item) for item in payload["history_question_indices"]),
            correctness=tuple(int(item) for item in payload["history_correctness"]),
            elapsed_times=tuple(float(item) for item in payload["history_elapsed_times"]),
            attempt_counts=tuple(int(item) for item in payload["history_attempt_counts"]),
            concept_indices=tuple(
                tuple(int(concept_index) for concept_index in concept_indices)
                for concept_indices in payload["history_concept_indices"]
            ),
        )
        return cls(
            split_name=str(payload["split_name"]),
            student_id=student_id,
            target_interaction_id=str(payload["target_interaction_id"]),
            history_start=0,
            target_position=len(standalone_history.interaction_ids),
            target_question_index=int(payload["target_question_index"]),
            target_concept_indices=tuple(int(item) for item in payload["target_concept_indices"]),
            target_correct=int(payload["target_correct"]),
            history_source=standalone_history,
        )


@dataclass(frozen=True, slots=True)
class SequenceSplit:
    name: str
    examples: tuple[SequenceExample, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "examples": [example.to_dict() for example in self.examples]}

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        histories: Mapping[str, StudentSequenceHistory],
    ) -> "SequenceSplit":
        return cls(
            name=str(payload["name"]),
            examples=tuple(SequenceExample.from_dict(item, histories) for item in payload["examples"]),
        )


@dataclass(frozen=True, slots=True)
class SequenceArtifacts:
    question_id_map: dict[str, int]
    concept_id_map: dict[str, int]
    student_histories: dict[str, StudentSequenceHistory]
    train: SequenceSplit
    val: SequenceSplit
    test: SequenceSplit
    metadata: dict[str, Any] = field(default_factory=dict)

    def split_counts(self) -> dict[str, int]:
        return {
            "train": len(self.train.examples),
            "val": len(self.val.examples),
            "test": len(self.test.examples),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id_map": dict(self.question_id_map),
            "concept_id_map": dict(self.concept_id_map),
            "student_histories": {
                student_id: history.to_dict() for student_id, history in self.student_histories.items()
            },
            "train": self.train.to_dict(),
            "val": self.val.to_dict(),
            "test": self.test.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SequenceArtifacts":
        histories = {
            str(student_id): StudentSequenceHistory.from_dict(history_payload)
            for student_id, history_payload in payload.get("student_histories", {}).items()
        }
        return cls(
            question_id_map={str(key): int(value) for key, value in payload["question_id_map"].items()},
            concept_id_map={str(key): int(value) for key, value in payload["concept_id_map"].items()},
            student_histories=histories,
            train=SequenceSplit.from_dict(payload["train"], histories),
            val=SequenceSplit.from_dict(payload["val"], histories),
            test=SequenceSplit.from_dict(payload["test"], histories),
            metadata=dict(payload.get("metadata", {})),
        )
