from __future__ import annotations

from pathlib import Path

from thgkt.data import EdNetAdapter
from thgkt.data.adapters.ednet import EdNetAdapterConfig
from thgkt.explainability import UserTagMapConfig, build_user_tag_map, export_user_tag_map_artifacts


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "ednet_small"


def _load_fixture_bundle(user_id: str):
    adapter = EdNetAdapter(config=EdNetAdapterConfig(user_ids=(user_id,)))
    return adapter.to_canonical(
        adapter.load_raw(FIXTURE_ROOT / "KT1", contents_path=FIXTURE_ROOT / "contents")
    )


def test_user_tag_map_counts_fixture_user_history() -> None:
    bundle = _load_fixture_bundle("u101")

    tag_map = build_user_tag_map(bundle, "u101")

    nodes = {node.tag: node for node in tag_map.nodes}
    assert nodes["skill_add"].correct_count == 0
    assert nodes["skill_add"].incorrect_count == 1
    assert nodes["skill_add"].node_weight == -1
    assert nodes["skill_foundation"].correct_count == 1
    assert nodes["skill_foundation"].incorrect_count == 1
    assert nodes["skill_foundation"].mastery_score == 0.0
    assert nodes["skill_div"].accuracy == 1.0

    edges = {(edge.source_tag, edge.target_tag): edge for edge in tag_map.edges}
    assert edges[("skill_add", "skill_foundation")].question_count == 1
    assert edges[("skill_foundation", "skill_sub")].question_count == 1
    assert tag_map.metadata["num_interactions"] == 4
    assert tag_map.metadata["num_nodes"] == 5


def test_user_tag_map_exports_json_and_svg(tmp_path) -> None:
    bundle = _load_fixture_bundle("u100")
    tag_map = build_user_tag_map(bundle, "u100", config=UserTagMapConfig(edge_scope="seen_questions"))

    exported = export_user_tag_map_artifacts(tag_map, tmp_path / "u100")

    assert Path(exported["json_path"]).exists()
    assert Path(exported["svg_path"]).exists()

