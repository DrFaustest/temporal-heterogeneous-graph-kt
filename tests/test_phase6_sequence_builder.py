from __future__ import annotations

from pathlib import Path

from thgkt.data.adapters.assistments import AssistmentsAdapter
from thgkt.data.adapters.synthetic import SyntheticToyAdapter
from thgkt.data.preprocessing import CanonicalPreprocessor, PreprocessingConfig
from thgkt.data.splitting import SplitConfig, make_splits
from thgkt.sequences import build_sequence_artifacts, collate_sequence_batch, load_sequence_artifacts, save_sequence_artifacts


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "assistments_sample.csv"


def _prepared_toy_bundle_and_split():
    bundle = CanonicalPreprocessor(PreprocessingConfig(min_interactions_per_student=4)).run(
        SyntheticToyAdapter().to_canonical()
    )
    split = make_splits(bundle, SplitConfig(strategy="student_chronological", random_seed=5))
    return bundle, split


def _prepared_assistments_bundle_and_split():
    bundle = CanonicalPreprocessor(PreprocessingConfig(min_interactions_per_student=4)).run(
        AssistmentsAdapter().to_canonical(AssistmentsAdapter().load_raw(FIXTURE_PATH))
    )
    split = make_splits(bundle, SplitConfig(strategy="student_chronological", random_seed=5))
    return bundle, split


def test_sequence_examples_are_correct_next_response_tuples() -> None:
    bundle, split = _prepared_toy_bundle_and_split()
    artifacts = build_sequence_artifacts(bundle, split)

    assert artifacts.split_counts() == {"train": 4, "val": 4, "test": 4}
    example = next(item for item in artifacts.val.examples if item.student_id == "s-0")

    assert example.target_interaction_id == "s-0-i-2"
    assert example.history_interaction_ids == ("s-0-i-0", "s-0-i-1")
    assert example.history_length == 2
    assert example.target_correct == 1


def test_sequence_builder_has_no_future_leakage() -> None:
    bundle, split = _prepared_assistments_bundle_and_split()
    artifacts = build_sequence_artifacts(bundle, split)
    seq_by_interaction = {
        str(row["interaction_id"]): int(row["seq_idx"])
        for row in bundle.interactions.rows
    }

    for sequence_split in (artifacts.train, artifacts.val, artifacts.test):
        for example in sequence_split.examples:
            target_seq = seq_by_interaction[example.target_interaction_id]
            assert all(seq_by_interaction[item] < target_seq for item in example.history_interaction_ids)


def test_variable_length_batching_and_sequence_artifact_roundtrip(tmp_path) -> None:
    bundle, split = _prepared_toy_bundle_and_split()
    artifacts = build_sequence_artifacts(bundle, split)

    batch = collate_sequence_batch([
        artifacts.train.examples[0],
        artifacts.val.examples[0],
        artifacts.test.examples[0],
    ])

    assert batch["history_lengths"] == [1, 2, 3]
    assert batch["history_masks"] == [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
    assert len(batch["history_question_indices"]) == 3

    artifact_path = tmp_path / "sequences" / "toy_sequences.json"
    save_sequence_artifacts(artifacts, artifact_path)
    loaded = load_sequence_artifacts(artifact_path)
    assert loaded.to_dict() == artifacts.to_dict()
