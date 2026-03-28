from __future__ import annotations

from thgkt.data.adapters.synthetic import SyntheticToyAdapter
from thgkt.data.io import load_canonical_bundle, save_canonical_bundle
from thgkt.schemas.validators import validate_bundle


def test_synthetic_toy_adapter_generates_valid_bundle(tmp_path) -> None:
    adapter = SyntheticToyAdapter()

    bundle = adapter.to_canonical(adapter.load_raw())
    report = validate_bundle(bundle)

    assert report.row_counts["interactions"] == 16
    assert report.row_counts["questions"] == 6
    assert report.row_counts["concepts"] == 3


def test_synthetic_toy_sequences_are_ordered_per_student() -> None:
    bundle = SyntheticToyAdapter().to_canonical()

    by_student: dict[str, list[int]] = {}
    for row in bundle.interactions.rows:
        by_student.setdefault(str(row["student_id"]), []).append(int(row["seq_idx"]))

    assert by_student
    for seq_indices in by_student.values():
        assert seq_indices == list(range(len(seq_indices)))


def test_synthetic_toy_bundle_roundtrips_to_artifact(tmp_path) -> None:
    bundle = SyntheticToyAdapter().to_canonical()
    artifact_path = tmp_path / "canonical_bundle.json"

    save_canonical_bundle(bundle, artifact_path)
    loaded = load_canonical_bundle(artifact_path)

    assert loaded.table_sizes() == bundle.table_sizes()
    assert loaded.metadata["dataset_name"] == "synthetic_toy"
