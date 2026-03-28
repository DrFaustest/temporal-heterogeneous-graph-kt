"""Build next-response sequence artifacts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from thgkt.data.artifacts import SplitArtifacts
from thgkt.schemas.canonical import CanonicalBundle
from thgkt.sequences.artifacts import SequenceArtifacts, SequenceExample, SequenceSplit, StudentSequenceHistory


@dataclass(frozen=True, slots=True)
class SequenceBuilderConfig:
    allow_empty_history: bool = False
    max_history_length: int | None = None


def build_sequence_artifacts(
    bundle: CanonicalBundle,
    split_artifacts: SplitArtifacts,
    config: SequenceBuilderConfig | None = None,
) -> SequenceArtifacts:
    sequence_config = config or SequenceBuilderConfig()
    question_id_map = {
        str(row["question_id"]): index
        for index, row in enumerate(sorted(bundle.questions.rows, key=lambda item: str(item["question_id"])))
    }
    concept_id_map = {
        str(row["concept_id"]): index
        for index, row in enumerate(sorted(bundle.concepts.rows, key=lambda item: str(item["concept_id"])))
    }

    split_by_interaction = _split_map(split_artifacts)
    by_student: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(bundle.interactions.rows, key=lambda item: (str(item["student_id"]), int(item["seq_idx"]))):
        by_student[str(row["student_id"])].append(dict(row))

    student_histories: dict[str, StudentSequenceHistory] = {}
    split_examples: dict[str, list[SequenceExample]] = {"train": [], "val": [], "test": []}
    for student_id, rows in by_student.items():
        ordered_rows = sorted(rows, key=lambda item: int(item["seq_idx"]))
        history = StudentSequenceHistory(
            student_id=student_id,
            interaction_ids=tuple(str(row["interaction_id"]) for row in ordered_rows),
            question_indices=tuple(question_id_map[str(row["question_id"])] for row in ordered_rows),
            correctness=tuple(int(row["correct"]) for row in ordered_rows),
            elapsed_times=tuple(float(row["elapsed_time"]) for row in ordered_rows),
            attempt_counts=tuple(int(row["attempt_count"]) for row in ordered_rows),
            concept_indices=tuple(
                tuple(concept_id_map[str(concept_id)] for concept_id in row["concept_ids"])
                for row in ordered_rows
            ),
        )
        student_histories[student_id] = history

        for target_idx, target_row in enumerate(ordered_rows):
            if not sequence_config.allow_empty_history and target_idx == 0:
                continue
            history_start = 0
            if sequence_config.max_history_length is not None and sequence_config.max_history_length > 0:
                history_start = max(0, target_idx - sequence_config.max_history_length)

            target_interaction_id = str(target_row["interaction_id"])
            target_split = split_by_interaction[target_interaction_id]
            example = SequenceExample(
                split_name=target_split,
                student_id=student_id,
                target_interaction_id=target_interaction_id,
                history_start=history_start,
                target_position=target_idx,
                target_question_index=question_id_map[str(target_row["question_id"])],
                target_concept_indices=tuple(
                    concept_id_map[str(concept_id)] for concept_id in target_row["concept_ids"]
                ),
                target_correct=int(target_row["correct"]),
                history_source=history,
            )
            split_examples[target_split].append(example)

    for split_name in split_examples:
        split_examples[split_name].sort(key=lambda example: (example.student_id, example.target_interaction_id))

    return SequenceArtifacts(
        question_id_map=question_id_map,
        concept_id_map=concept_id_map,
        student_histories=student_histories,
        train=SequenceSplit("train", tuple(split_examples["train"])),
        val=SequenceSplit("val", tuple(split_examples["val"])),
        test=SequenceSplit("test", tuple(split_examples["test"])),
        metadata={
            "sequence_builder": "build_sequence_artifacts",
            "allow_empty_history": sequence_config.allow_empty_history,
            "max_history_length": sequence_config.max_history_length,
            "split_strategy": split_artifacts.split_strategy,
        },
    )


def collate_sequence_batch(examples: list[SequenceExample]) -> dict[str, Any]:
    if not examples:
        raise ValueError("Cannot collate an empty sequence batch.")

    max_length = max(example.history_length for example in examples)
    padded_questions: list[list[int]] = []
    padded_correctness: list[list[int]] = []
    padded_elapsed: list[list[float]] = []
    padded_attempts: list[list[int]] = []
    padded_concepts: list[list[list[int]]] = []
    masks: list[list[int]] = []

    for example in examples:
        pad_length = max_length - example.history_length
        padded_questions.append(list(example.history_question_indices) + [-1] * pad_length)
        padded_correctness.append(list(example.history_correctness) + [0] * pad_length)
        padded_elapsed.append(list(example.history_elapsed_times) + [0.0] * pad_length)
        padded_attempts.append(list(example.history_attempt_counts) + [0] * pad_length)
        padded_concepts.append([list(items) for items in example.history_concept_indices] + [[] for _ in range(pad_length)])
        masks.append([1] * example.history_length + [0] * pad_length)

    return {
        "student_ids": [example.student_id for example in examples],
        "split_names": [example.split_name for example in examples],
        "target_interaction_ids": [example.target_interaction_id for example in examples],
        "history_lengths": [example.history_length for example in examples],
        "history_question_indices": padded_questions,
        "history_correctness": padded_correctness,
        "history_elapsed_times": padded_elapsed,
        "history_attempt_counts": padded_attempts,
        "history_concept_indices": padded_concepts,
        "history_masks": masks,
        "target_question_index": [example.target_question_index for example in examples],
        "target_concept_indices": [list(example.target_concept_indices) for example in examples],
        "targets": [example.target_correct for example in examples],
    }


def _split_map(split_artifacts: SplitArtifacts) -> dict[str, str]:
    split_by_interaction: dict[str, str] = {}
    for split_name, split in (
        ("train", split_artifacts.train),
        ("val", split_artifacts.val),
        ("test", split_artifacts.test),
    ):
        for interaction_id in split.interaction_ids:
            split_by_interaction[str(interaction_id)] = split_name
    return split_by_interaction
