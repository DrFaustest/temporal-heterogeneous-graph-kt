from __future__ import annotations

from pathlib import Path

from thgkt.data.adapters.assistments import AssistmentsAdapter
from thgkt.data.bundles import make_canonical_bundle
from thgkt.data.io import load_split_artifacts, save_split_artifacts
from thgkt.data.preprocessing import CanonicalPreprocessor, PreprocessingConfig
from thgkt.data.splitting import SplitConfig, check_chronological_no_leakage, check_no_student_overlap, make_splits
from thgkt.schemas.validators import validate_bundle


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "assistments_sample.csv"


def _assistments_bundle():
    adapter = AssistmentsAdapter()
    return adapter.to_canonical(adapter.load_raw(FIXTURE_PATH))


def test_preprocessing_cleans_filters_and_reindexes() -> None:
    base = _assistments_bundle()
    broken_interactions = list(base.interactions.rows) + [
        {
            "student_id": "short-student",
            "interaction_id": "short-0",
            "seq_idx": 0,
            "timestamp": "2024-01-05T09:00:00",
            "question_id": "q-1",
            "correct": 1,
            "concept_ids": [],
            "elapsed_time": 1.0,
            "attempt_count": 1,
            "source_dataset": "assistments",
        }
    ]
    dirty = make_canonical_bundle(
        interactions=broken_interactions,
        questions=base.questions.rows,
        concepts=base.concepts.rows,
        question_concept_map=base.question_concept_map.rows,
        concept_relations=base.concept_relations.rows,
        metadata=base.metadata,
    )

    preprocessor = CanonicalPreprocessor(PreprocessingConfig(min_interactions_per_student=4))
    cleaned = preprocessor.run(dirty)
    report = validate_bundle(cleaned)

    assert report.row_counts["interactions"] == 16
    assert all(row["student_id"] != "short-student" for row in cleaned.interactions.rows)
    for student_id in {row["student_id"] for row in cleaned.interactions.rows}:
        seq_indices = [
            int(row["seq_idx"])
            for row in cleaned.interactions.rows
            if row["student_id"] == student_id
        ]
        assert seq_indices == list(range(len(seq_indices)))


def test_chronological_splits_are_reproducible_and_no_leakage(tmp_path) -> None:
    bundle = CanonicalPreprocessor(PreprocessingConfig(min_interactions_per_student=4)).run(_assistments_bundle())

    config = SplitConfig(strategy="student_chronological", random_seed=7)
    split_a = make_splits(bundle, config)
    split_b = make_splits(bundle, config)
    check_chronological_no_leakage(bundle, split_a)

    assert split_a.to_dict() == split_b.to_dict()

    artifact_path = tmp_path / "splits" / "student_chronological.json"
    save_split_artifacts(split_a, artifact_path)
    loaded = load_split_artifacts(artifact_path)
    assert loaded.to_dict() == split_a.to_dict()


def test_student_holdout_has_no_overlap() -> None:
    bundle = CanonicalPreprocessor(PreprocessingConfig(min_interactions_per_student=4)).run(_assistments_bundle())

    split = make_splits(bundle, SplitConfig(strategy="student_holdout", random_seed=11))
    check_no_student_overlap(split)

    assert set(split.train.student_ids).isdisjoint(split.val.student_ids)
    assert set(split.train.student_ids).isdisjoint(split.test.student_ids)
    assert set(split.val.student_ids).isdisjoint(split.test.student_ids)
