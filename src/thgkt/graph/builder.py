"""Build heterogeneous graph artifacts from canonical bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from thgkt.data.artifacts import SplitArtifacts
from thgkt.graph.artifacts import EdgeIndexArtifact, EdgeRelation, HeteroGraphArtifacts
from thgkt.schemas.canonical import CanonicalBundle


@dataclass(frozen=True, slots=True)
class GraphBuilderConfig:
    interaction_edge_split: str = "train"
    include_student_exposure_edges: bool = True
    include_reverse_edges: bool = False


def build_hetero_graph(
    bundle: CanonicalBundle,
    split_artifacts: SplitArtifacts,
    config: GraphBuilderConfig | None = None,
) -> HeteroGraphArtifacts:
    graph_config = config or GraphBuilderConfig()
    node_maps = {
        "student": _make_node_map(row["student_id"] for row in bundle.interactions.rows),
        "question": _make_node_map(row["question_id"] for row in bundle.questions.rows),
        "concept": _make_node_map(row["concept_id"] for row in bundle.concepts.rows),
    }

    interaction_ids = _interaction_ids_for_split(split_artifacts, graph_config.interaction_edge_split)
    interactions = [
        dict(row)
        for row in bundle.interactions.rows
        if str(row["interaction_id"]) in interaction_ids
    ]

    edges: dict[str, EdgeIndexArtifact] = {}
    answered_relation = EdgeRelation("student", "answered", "question")
    student_answered_pairs = [
        (
            node_maps["student"][str(row["student_id"])],
            node_maps["question"][str(row["question_id"])],
        )
        for row in interactions
    ]
    edges[answered_relation.key] = _pairs_to_edge_index(student_answered_pairs)

    tests_relation = EdgeRelation("question", "tests", "concept")
    question_concept_pairs = [
        (
            node_maps["question"][str(row["question_id"])],
            node_maps["concept"][str(row["concept_id"])],
        )
        for row in bundle.question_concept_map.rows
    ]
    edges[tests_relation.key] = _pairs_to_edge_index(question_concept_pairs)

    prerequisite_relation = EdgeRelation("concept", "prerequisite_of", "concept")
    prerequisite_pairs = [
        (
            node_maps["concept"][str(row["source_concept_id"])],
            node_maps["concept"][str(row["target_concept_id"])],
        )
        for row in bundle.concept_relations.rows
    ]
    edges[prerequisite_relation.key] = _pairs_to_edge_index(prerequisite_pairs)

    if graph_config.include_student_exposure_edges:
        exposure_relation = EdgeRelation("student", "exposure_to", "concept")
        exposure_pairs = sorted(
            {
                (
                    node_maps["student"][str(row["student_id"])],
                    node_maps["concept"][str(concept_id)],
                )
                for row in interactions
                for concept_id in row["concept_ids"]
            }
        )
        edges[exposure_relation.key] = _pairs_to_edge_index(exposure_pairs)

    if graph_config.include_reverse_edges:
        for edge_key, edge_index in list(edges.items()):
            relation = EdgeRelation.from_key(edge_key)
            reverse = EdgeRelation(relation.dst_type, f"rev_{relation.relation}", relation.src_type)
            edges[reverse.key] = EdgeIndexArtifact(
                src_indices=edge_index.dst_indices,
                dst_indices=edge_index.src_indices,
            )

    graph = HeteroGraphArtifacts(
        node_maps=node_maps,
        edges=edges,
        metadata={
            "graph_builder": "build_hetero_graph",
            "interaction_edge_split": graph_config.interaction_edge_split,
            "include_student_exposure_edges": graph_config.include_student_exposure_edges,
            "include_reverse_edges": graph_config.include_reverse_edges,
            "split_strategy": split_artifacts.split_strategy,
            "student_interaction_counts": _count_student_interactions(interactions),
        },
    )
    graph.validate_indices()
    return graph


def _make_node_map(values: Any) -> dict[str, int]:
    unique_values = sorted({str(value) for value in values})
    return {value: index for index, value in enumerate(unique_values)}


def _pairs_to_edge_index(pairs: list[tuple[int, int]]) -> EdgeIndexArtifact:
    if not pairs:
        return EdgeIndexArtifact(src_indices=(), dst_indices=())
    src_indices, dst_indices = zip(*pairs)
    return EdgeIndexArtifact(src_indices=tuple(src_indices), dst_indices=tuple(dst_indices))


def _interaction_ids_for_split(split_artifacts: SplitArtifacts, split_name: str) -> set[str]:
    if split_name == "train":
        return set(split_artifacts.train.interaction_ids)
    if split_name == "val":
        return set(split_artifacts.val.interaction_ids)
    if split_name == "test":
        return set(split_artifacts.test.interaction_ids)
    if split_name == "all":
        return set(split_artifacts.train.interaction_ids) | set(split_artifacts.val.interaction_ids) | set(split_artifacts.test.interaction_ids)
    raise ValueError(f"Unsupported graph interaction split: {split_name}")


def _count_student_interactions(interactions: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in interactions:
        student_id = str(row["student_id"])
        counts[student_id] = counts.get(student_id, 0) + 1
    return dict(sorted(counts.items()))
