"""Reproducible split creation and leakage checks."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from thgkt.data.artifacts import SplitArtifacts, SplitIndices
from thgkt.schemas.canonical import CanonicalBundle


@dataclass(frozen=True, slots=True)
class SplitConfig:
    strategy: str = "student_chronological"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42


def make_splits(
    bundle: CanonicalBundle,
    config: SplitConfig | None = None,
) -> SplitArtifacts:
    split_config = config or SplitConfig()
    if split_config.strategy == "student_chronological":
        return _student_chronological_split(bundle, split_config)
    if split_config.strategy == "student_holdout":
        return _student_holdout_split(bundle, split_config)
    raise ValueError(f"Unsupported split strategy: {split_config.strategy}")


def check_no_student_overlap(artifacts: SplitArtifacts) -> None:
    train = set(artifacts.train.student_ids)
    val = set(artifacts.val.student_ids)
    test = set(artifacts.test.student_ids)
    if train & val or train & test or val & test:
        raise ValueError("Student overlap detected across split partitions.")


def check_chronological_no_leakage(bundle: CanonicalBundle, artifacts: SplitArtifacts) -> None:
    split_by_interaction_id = {}
    for split_name, split in (
        ("train", artifacts.train),
        ("val", artifacts.val),
        ("test", artifacts.test),
    ):
        for interaction_id in split.interaction_ids:
            split_by_interaction_id[interaction_id] = split_name

    per_student: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in bundle.interactions.rows:
        interaction_id = str(row["interaction_id"])
        if interaction_id in split_by_interaction_id:
            per_student[str(row["student_id"])].append(
                (int(row["seq_idx"]), split_by_interaction_id[interaction_id])
            )

    rank = {"train": 0, "val": 1, "test": 2}
    for student_id, events in per_student.items():
        ordered = sorted(events, key=lambda item: item[0])
        split_ranks = [rank[split_name] for _, split_name in ordered]
        if split_ranks != sorted(split_ranks):
            raise ValueError(f"Temporal leakage detected for student_id={student_id}")


def _student_chronological_split(bundle: CanonicalBundle, config: SplitConfig) -> SplitArtifacts:
    per_student_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    interaction_to_student: dict[str, str] = {}
    for row in bundle.interactions.rows:
        student_id = str(row["student_id"])
        interaction_id = str(row["interaction_id"])
        per_student_rows[student_id].append(dict(row))
        interaction_to_student[interaction_id] = student_id

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    for student_id in sorted(per_student_rows):
        rows = sorted(per_student_rows[student_id], key=lambda item: int(item["seq_idx"]))
        n_rows = len(rows)
        if n_rows < 3:
            raise ValueError(
                f"student_chronological split requires at least 3 interactions per student; "
                f"student_id={student_id} has {n_rows}"
            )

        train_count = max(1, int(n_rows * config.train_ratio))
        val_count = max(1, int(n_rows * config.val_ratio))
        if train_count + val_count >= n_rows:
            val_count = 1
            train_count = max(1, n_rows - 2)
        test_count = n_rows - train_count - val_count
        if test_count <= 0:
            test_count = 1
            if train_count > 1:
                train_count -= 1
            else:
                val_count -= 1

        train_ids.extend(str(row["interaction_id"]) for row in rows[:train_count])
        val_ids.extend(str(row["interaction_id"]) for row in rows[train_count : train_count + val_count])
        test_ids.extend(str(row["interaction_id"]) for row in rows[train_count + val_count :])

    train_students = tuple(sorted({interaction_to_student[item] for item in train_ids}))
    val_students = tuple(sorted({interaction_to_student[item] for item in val_ids}))
    test_students = tuple(sorted({interaction_to_student[item] for item in test_ids}))

    artifacts = SplitArtifacts(
        split_strategy=config.strategy,
        random_seed=config.random_seed,
        train=SplitIndices(name="train", interaction_ids=tuple(train_ids), student_ids=train_students),
        val=SplitIndices(name="val", interaction_ids=tuple(val_ids), student_ids=val_students),
        test=SplitIndices(name="test", interaction_ids=tuple(test_ids), student_ids=test_students),
        metadata={
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
        },
    )
    check_chronological_no_leakage(bundle, artifacts)
    return artifacts


def _student_holdout_split(bundle: CanonicalBundle, config: SplitConfig) -> SplitArtifacts:
    student_ids = sorted({str(row["student_id"]) for row in bundle.interactions.rows})
    shuffled = list(student_ids)
    rng = random.Random(config.random_seed)
    rng.shuffle(shuffled)

    n_students = len(shuffled)
    train_count = max(1, int(n_students * config.train_ratio))
    val_count = max(1, int(n_students * config.val_ratio))
    if train_count + val_count >= n_students:
        val_count = 1
        train_count = max(1, n_students - 2)
    test_count = n_students - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if train_count > 1:
            train_count -= 1
        else:
            val_count -= 1

    train_students = set(shuffled[:train_count])
    val_students = set(shuffled[train_count : train_count + val_count])
    test_students = set(shuffled[train_count + val_count :])

    def _interaction_ids_for(students: set[str]) -> tuple[str, ...]:
        return tuple(
            str(row["interaction_id"])
            for row in bundle.interactions.rows
            if str(row["student_id"]) in students
        )

    artifacts = SplitArtifacts(
        split_strategy=config.strategy,
        random_seed=config.random_seed,
        train=SplitIndices("train", _interaction_ids_for(train_students), tuple(sorted(train_students))),
        val=SplitIndices("val", _interaction_ids_for(val_students), tuple(sorted(val_students))),
        test=SplitIndices("test", _interaction_ids_for(test_students), tuple(sorted(test_students))),
        metadata={
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
        },
    )
    check_no_student_overlap(artifacts)
    return artifacts
