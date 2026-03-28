from __future__ import annotations

from pathlib import Path

from thgkt.data import (
    AssistmentsAdapter,
    CanonicalPreprocessor,
    PreprocessingConfig,
    RelationConfig,
    apply_concept_relation_mode,
    generate_concept_relations,
)
from thgkt.graph import GraphBuilderConfig, build_hetero_graph
from thgkt.data import SplitConfig, make_splits


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "assistments_sample.csv"


def _base_bundle():
    return CanonicalPreprocessor(PreprocessingConfig(min_interactions_per_student=4)).run(
        AssistmentsAdapter().to_canonical(AssistmentsAdapter().load_raw(FIXTURE_PATH))
    )


def test_relation_generation_modes_produce_valid_variants() -> None:
    bundle = _base_bundle()

    none_relations = generate_concept_relations(bundle, RelationConfig(mode="none"))
    cooccurrence_relations = generate_concept_relations(bundle, RelationConfig(mode="cooccurrence"))
    transition_relations = generate_concept_relations(bundle, RelationConfig(mode="transition"))

    assert none_relations == []
    assert len(cooccurrence_relations) > 0
    assert len(transition_relations) > 0
    assert {row["relation_type"] for row in cooccurrence_relations} == {"cooccurs_with"}
    assert {row["relation_type"] for row in transition_relations} == {"transition_to"}


def test_graph_mode_switch_changes_graph_artifact_shape() -> None:
    bundle = _base_bundle()
    split = make_splits(bundle, SplitConfig(strategy="student_chronological", random_seed=29))

    none_bundle = apply_concept_relation_mode(bundle, RelationConfig(mode="none"))
    co_bundle = apply_concept_relation_mode(bundle, RelationConfig(mode="cooccurrence"))
    transition_bundle = apply_concept_relation_mode(bundle, RelationConfig(mode="transition"))

    none_graph = build_hetero_graph(none_bundle, split, GraphBuilderConfig(interaction_edge_split="train"))
    co_graph = build_hetero_graph(co_bundle, split, GraphBuilderConfig(interaction_edge_split="train"))
    transition_graph = build_hetero_graph(transition_bundle, split, GraphBuilderConfig(interaction_edge_split="train"))

    assert none_graph.edge_counts()["concept|prerequisite_of|concept"] == 0
    assert co_graph.edge_counts()["concept|prerequisite_of|concept"] > 0
    assert transition_graph.edge_counts()["concept|prerequisite_of|concept"] > 0
    assert co_bundle.metadata["relation_generation"]["mode"] == "cooccurrence"
    assert transition_bundle.metadata["relation_generation"]["mode"] == "transition"
    assert co_graph.edge_counts()["concept|prerequisite_of|concept"] != none_graph.edge_counts()["concept|prerequisite_of|concept"]
