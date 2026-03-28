"""PyG conversion helpers for heterogeneous graph artifacts."""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from thgkt.graph.artifacts import EdgeRelation, HeteroGraphArtifacts


def to_pyg_heterodata(
    graph: HeteroGraphArtifacts,
    *,
    add_reverse_edges: bool = True,
    drop_prerequisite_edges: bool = False,
) -> HeteroData:
    data = HeteroData()
    for node_type, mapping in graph.node_maps.items():
        data[node_type].node_id = torch.arange(len(mapping), dtype=torch.long)

    for edge_key, edge_index in graph.edges.items():
        relation = EdgeRelation.from_key(edge_key)
        if drop_prerequisite_edges and relation.relation == "prerequisite_of":
            continue
        edge_tensor = torch.tensor([list(edge_index.src_indices), list(edge_index.dst_indices)], dtype=torch.long)
        data[(relation.src_type, relation.relation, relation.dst_type)].edge_index = edge_tensor
        if add_reverse_edges:
            reverse_key = (relation.dst_type, f"rev_{relation.relation}", relation.src_type)
            data[reverse_key].edge_index = torch.flip(edge_tensor, dims=[0])

    return data
