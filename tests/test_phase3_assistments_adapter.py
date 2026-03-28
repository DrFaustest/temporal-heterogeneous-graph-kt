from __future__ import annotations

from pathlib import Path

from thgkt.data.adapters.assistments import AssistmentsAdapter
from thgkt.data.io import load_canonical_bundle, save_canonical_bundle, save_json
from thgkt.schemas.validators import validate_bundle


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "assistments_sample.csv"


def test_assistments_subset_loads_and_validates(tmp_path) -> None:
    adapter = AssistmentsAdapter()

    raw = adapter.load_raw(FIXTURE_PATH)
    bundle = adapter.to_canonical(raw)
    report = validate_bundle(bundle)

    assert report.row_counts["interactions"] == 16
    assert report.row_counts["questions"] == 4
    assert report.row_counts["concepts"] == 4
    assert bundle.metadata["summary"]["num_students"] == 4

    bundle_path = tmp_path / "assistments_bundle.json"
    summary_path = tmp_path / "assistments_summary.json"
    save_canonical_bundle(bundle, bundle_path)
    save_json(bundle.metadata["summary"], summary_path)

    assert load_canonical_bundle(bundle_path).table_sizes() == bundle.table_sizes()
    assert summary_path.exists()


def test_assistments_sequences_are_monotonic_and_mapping_exists() -> None:
    adapter = AssistmentsAdapter()
    bundle = adapter.to_canonical(adapter.load_raw(FIXTURE_PATH))

    per_student: dict[str, list[int]] = {}
    for row in bundle.interactions.rows:
        per_student.setdefault(str(row["student_id"]), []).append(int(row["seq_idx"]))

    for seq_indices in per_student.values():
        assert seq_indices == list(range(len(seq_indices)))

    q3_concepts = {
        row["concept_id"]
        for row in bundle.question_concept_map.rows
        if row["question_id"] == "q-3"
    }
    assert q3_concepts == {"c-add", "c-mul"}


def test_assistments_adapter_accepts_directory_input_and_non_brittle_filename(tmp_path) -> None:
    nested = tmp_path / "raw_data"
    nested.mkdir()
    renamed = nested / "subset_like_export.csv"
    renamed.write_text(FIXTURE_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    adapter = AssistmentsAdapter()
    raw = adapter.load_raw(nested)
    bundle = adapter.to_canonical(raw)

    assert bundle.metadata["dataset_name"] == "subset_like_export"
    assert bundle.metadata["raw_path"].endswith("subset_like_export.csv")
