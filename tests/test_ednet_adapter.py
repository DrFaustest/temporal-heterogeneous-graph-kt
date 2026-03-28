from __future__ import annotations

from pathlib import Path

from thgkt.data import EdNetAdapter
from thgkt.data.adapters.ednet import EdNetAdapterConfig
from thgkt.schemas.validators import validate_bundle


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "ednet_small"


def test_ednet_fixture_loads_and_validates() -> None:
    adapter = EdNetAdapter()
    bundle = adapter.to_canonical(
        adapter.load_raw(FIXTURE_ROOT / "KT1", contents_path=FIXTURE_ROOT / "contents")
    )
    report = validate_bundle(bundle)

    assert report.row_counts["interactions"] == 8
    assert report.row_counts["questions"] == 4
    assert report.row_counts["concepts"] == 5
    assert bundle.metadata["num_user_files_loaded"] == 2
    assert any(row["correct"] == 0 for row in bundle.interactions.rows)
    assert any(row["correct"] == 1 for row in bundle.interactions.rows)


def test_ednet_concept_mapping_comes_from_questions_metadata() -> None:
    adapter = EdNetAdapter()
    bundle = adapter.to_canonical(
        adapter.load_raw(FIXTURE_ROOT / "KT1", contents_path=FIXTURE_ROOT / "contents")
    )

    q2844_concepts = {
        row["concept_id"]
        for row in bundle.question_concept_map.rows
        if row["question_id"] == "q2844"
    }
    assert q2844_concepts == {"skill_add", "skill_foundation"}


def test_ednet_adapter_supports_user_cap() -> None:
    adapter = EdNetAdapter(config=EdNetAdapterConfig(max_users=1))
    bundle = adapter.to_canonical(
        adapter.load_raw(FIXTURE_ROOT / "KT1", contents_path=FIXTURE_ROOT / "contents")
    )

    assert bundle.metadata["num_user_files_loaded"] == 1
    assert len({row["student_id"] for row in bundle.interactions.rows}) == 1


def test_ednet_adapter_supports_specific_user_ids() -> None:
    adapter = EdNetAdapter(config=EdNetAdapterConfig(user_ids=("u101",)))
    bundle = adapter.to_canonical(
        adapter.load_raw(FIXTURE_ROOT / "KT1", contents_path=FIXTURE_ROOT / "contents")
    )

    assert bundle.metadata["num_user_files_loaded"] == 1
    assert {row["student_id"] for row in bundle.interactions.rows} == {"u101"}
