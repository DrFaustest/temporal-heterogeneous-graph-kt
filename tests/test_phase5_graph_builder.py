from __future__ import annotations

from pathlib import Path

from thgkt.data.adapters.assistments import AssistmentsAdapter
from thgkt.data.adapters.synthetic import SyntheticToyAdapter
from thgkt.data.preprocessing import CanonicalPreprocessor, PreprocessingConfig
from thgkt.data.splitting import SplitConfig, make_splits
from thgkt.graph import build_hetero_graph, load_graph_artifacts, save_graph_artifacts


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "assistments_sample.csv"


def _prepared_toy_bundle_and_split():
    bundle = CanonicalPreprocessor(PreprocessingConfig(min_interactions_per_student=4)).run(
        SyntheticToyAdapter().to_canonical()
    )
    split = make_splits(bundle, SplitConfig(strategy="student_chronological", random_seed=3))
    return bundle, split


def _prepared_assistments_bundle_and_split():
    bundle = CanonicalPreprocessor(PreprocessingConfig(min_interactions_per_student=4)).run(
        AssistmentsAdapter().to_canonical(AssistmentsAdapter().load_raw(FIXTURE_PATH))
    )
    split = make_splits(bundle, SplitConfig(strategy="student_chronological", random_seed=3))
    return bundle, split


def test_graph_builds_from_toy_and_roundtrips(tmp_path) -> None:
    bundle, split = _prepared_toy_bundle_and_split()

    graph = build_hetero_graph(bundle, split)
    graph.validate_indices()

    assert graph.node_counts() == {"student": 4, "question": 6, "concept": 3}
    assert graph.edge_counts()["student|answered|question"] == len(split.train.interaction_ids)
    assert graph.edge_counts()["question|tests|concept"] == len(bundle.question_concept_map.rows)
    assert graph.edge_counts()["concept|prerequisite_of|concept"] == len(bundle.concept_relations.rows)
    assert graph.edge_counts()["student|exposure_to|concept"] > 0

    artifact_path = tmp_path / "graph" / "toy_graph.json"
    save_graph_artifacts(graph, artifact_path)
    loaded = load_graph_artifacts(artifact_path)
    assert loaded.to_dict() == graph.to_dict()


def test_graph_builds_from_assistments_subset_without_invalid_indices() -> None:
    bundle, split = _prepared_assistments_bundle_and_split()

    graph = build_hetero_graph(bundle, split)
    graph.validate_indices()

    assert graph.node_counts() == {"student": 4, "question": 4, "concept": 4}
    assert graph.edge_counts()["student|answered|question"] == len(split.train.interaction_ids)
    assert graph.edge_counts()["question|tests|concept"] == len(bundle.question_concept_map.rows)
    assert graph.edge_counts()["concept|prerequisite_of|concept"] == 0
    assert graph.metadata["interaction_edge_split"] == "train"
