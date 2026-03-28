"""Serializable heterogeneous graph artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class EdgeRelation:
    src_type: str
    relation: str
    dst_type: str

    @property
    def key(self) -> str:
        return f"{self.src_type}|{self.relation}|{self.dst_type}"

    def to_dict(self) -> dict[str, str]:
        return {
            "src_type": self.src_type,
            "relation": self.relation,
            "dst_type": self.dst_type,
        }

    @classmethod
    def from_key(cls, key: str) -> "EdgeRelation":
        src_type, relation, dst_type = key.split("|", maxsplit=2)
        return cls(src_type=src_type, relation=relation, dst_type=dst_type)


@dataclass(frozen=True, slots=True)
class EdgeIndexArtifact:
    src_indices: tuple[int, ...]
    dst_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.src_indices) != len(self.dst_indices):
            raise ValueError("Edge index artifacts must have equal source and destination lengths.")

    @property
    def num_edges(self) -> int:
        return len(self.src_indices)

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "src_indices": list(self.src_indices),
            "dst_indices": list(self.dst_indices),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EdgeIndexArtifact":
        return cls(
            src_indices=tuple(int(item) for item in payload["src_indices"]),
            dst_indices=tuple(int(item) for item in payload["dst_indices"]),
        )


@dataclass(frozen=True, slots=True)
class HeteroGraphArtifacts:
    node_maps: dict[str, dict[str, int]]
    edges: dict[str, EdgeIndexArtifact]
    metadata: dict[str, Any] = field(default_factory=dict)

    def node_counts(self) -> dict[str, int]:
        return {node_type: len(mapping) for node_type, mapping in self.node_maps.items()}

    def edge_counts(self) -> dict[str, int]:
        return {edge_key: edge.num_edges for edge_key, edge in self.edges.items()}

    def validate_indices(self) -> None:
        node_counts = self.node_counts()
        for edge_key, edge in self.edges.items():
            relation = EdgeRelation.from_key(edge_key)
            max_src = node_counts.get(relation.src_type, 0)
            max_dst = node_counts.get(relation.dst_type, 0)
            for src in edge.src_indices:
                if src < 0 or src >= max_src:
                    raise ValueError(f"Invalid source index {src} for edge type {edge_key}")
            for dst in edge.dst_indices:
                if dst < 0 or dst >= max_dst:
                    raise ValueError(f"Invalid destination index {dst} for edge type {edge_key}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_maps": {node_type: dict(mapping) for node_type, mapping in self.node_maps.items()},
            "edges": {edge_key: edge.to_dict() for edge_key, edge in self.edges.items()},
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HeteroGraphArtifacts":
        return cls(
            node_maps={
                str(node_type): {str(node_id): int(index) for node_id, index in mapping.items()}
                for node_type, mapping in payload["node_maps"].items()
            },
            edges={
                str(edge_key): EdgeIndexArtifact.from_dict(edge_payload)
                for edge_key, edge_payload in payload["edges"].items()
            },
            metadata=dict(payload.get("metadata", {})),
        )
