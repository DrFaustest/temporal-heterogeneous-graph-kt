"""Temporal Heterogeneous Graph Knowledge Tracing model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv


@dataclass(frozen=True, slots=True)
class THGKTConfig:
    num_students: int
    num_questions: int
    num_concepts: int
    hidden_dim: int = 32
    temporal_hidden_dim: int = 32
    graph_num_layers: int = 2
    use_graph_encoder: bool = True
    use_temporal_encoder: bool = True
    use_time_features: bool = True
    use_prerequisite_edges: bool = True
    use_target_concept_attention: bool = False
    elapsed_time_scale: float = 5000.0
    attempt_count_scale: float = 5.0
    dropout: float = 0.1


class THGKTModel(nn.Module):
    def __init__(self, config: THGKTConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        temporal_hidden = config.temporal_hidden_dim

        self.student_embedding = nn.Embedding(config.num_students, hidden_dim)
        self.question_embedding = nn.Embedding(config.num_questions + 1, hidden_dim, padding_idx=config.num_questions)
        self.concept_embedding = nn.Embedding(config.num_concepts, hidden_dim)
        self.correctness_embedding = nn.Embedding(2, hidden_dim)
        self.time_projection = nn.Linear(2, hidden_dim)

        self.history_projection = nn.Linear(hidden_dim * 3, temporal_hidden)
        self.temporal_encoder = nn.GRU(
            input_size=temporal_hidden,
            hidden_size=temporal_hidden,
            batch_first=True,
        )

        self.graph_encoders = nn.ModuleList([
            self._build_graph_encoder(hidden_dim) for _ in range(max(1, config.graph_num_layers))
        ])
        self.graph_norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "student": nn.LayerNorm(hidden_dim),
                        "question": nn.LayerNorm(hidden_dim),
                        "concept": nn.LayerNorm(hidden_dim),
                    }
                )
                for _ in range(len(self.graph_encoders))
            ]
        )
        self.graph_dropout = nn.Dropout(config.dropout)
        self.target_concept_attention = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            if config.use_target_concept_attention
            else None
        )
        fusion_input_dim = hidden_dim * 3 + temporal_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        graph_data = batch["graph_data"]
        x_dict = {
            "student": self.student_embedding.weight,
            "question": self.question_embedding.weight[:-1],
            "concept": self.concept_embedding.weight,
        }
        if self.config.use_graph_encoder and graph_data is not None:
            edge_index_dict = {
                edge_type: edge_index
                for edge_type, edge_index in graph_data.edge_index_dict.items()
                if self.config.use_prerequisite_edges or edge_type[1] not in {"prerequisite_of", "rev_prerequisite_of"}
            }
            for layer_index, graph_encoder in enumerate(self.graph_encoders):
                encoded = graph_encoder(x_dict, edge_index_dict)
                next_x_dict: dict[str, torch.Tensor] = {}
                for node_type, residual in x_dict.items():
                    updated = encoded.get(node_type, residual)
                    updated = self.graph_norms[layer_index][node_type](updated + residual)
                    if layer_index < len(self.graph_encoders) - 1:
                        updated = torch.relu(updated)
                        updated = self.graph_dropout(updated)
                    next_x_dict[node_type] = updated
                x_dict = next_x_dict

        student_repr = x_dict["student"][batch["student_indices"]]
        target_question_repr = x_dict["question"][batch["target_question_index"]]
        target_concept_repr = self._pool_concepts(
            x_dict["concept"],
            batch["target_concept_indices"],
            batch["target_concept_mask"],
        )
        temporal_repr = self._encode_temporal(batch)
        fused = torch.cat([student_repr, target_question_repr, target_concept_repr, temporal_repr], dim=-1)
        logits = self.fusion(fused).squeeze(-1)
        probs = torch.sigmoid(logits)
        return {
            "logits": logits,
            "probs": probs,
            "aux_outputs": {
                "student_graph_repr": student_repr.detach(),
                "target_question_repr": target_question_repr.detach(),
                "target_concept_repr": target_concept_repr.detach(),
                "temporal_repr": temporal_repr.detach(),
            },
            "debug_info": {
                "use_graph_encoder": self.config.use_graph_encoder,
                "graph_num_layers": len(self.graph_encoders),
                "use_temporal_encoder": self.config.use_temporal_encoder,
                "use_time_features": self.config.use_time_features,
                "use_prerequisite_edges": self.config.use_prerequisite_edges,
                "use_target_concept_attention": self.config.use_target_concept_attention,
                "elapsed_time_scale": self.config.elapsed_time_scale,
                "attempt_count_scale": self.config.attempt_count_scale,
            },
        }

    def _build_graph_encoder(self, hidden_dim: int) -> HeteroConv:
        return HeteroConv(
            {
                ("student", "answered", "question"): SAGEConv((-1, -1), hidden_dim),
                ("question", "tests", "concept"): SAGEConv((-1, -1), hidden_dim),
                ("concept", "prerequisite_of", "concept"): SAGEConv((-1, -1), hidden_dim),
                ("student", "exposure_to", "concept"): SAGEConv((-1, -1), hidden_dim),
                ("question", "rev_answered", "student"): SAGEConv((-1, -1), hidden_dim),
                ("concept", "rev_tests", "question"): SAGEConv((-1, -1), hidden_dim),
                ("concept", "rev_exposure_to", "student"): SAGEConv((-1, -1), hidden_dim),
                ("concept", "rev_prerequisite_of", "concept"): SAGEConv((-1, -1), hidden_dim),
            },
            aggr="sum",
        )

    def _encode_temporal(self, batch: dict[str, Any]) -> torch.Tensor:
        batch_size = batch["history_question_indices"].shape[0]
        device = batch["history_question_indices"].device
        if not self.config.use_temporal_encoder:
            return torch.zeros(batch_size, self.config.temporal_hidden_dim, device=device)

        lengths = batch["history_lengths"].long()
        max_length = int(batch["history_question_indices"].shape[1])
        if max_length == 0 or int(lengths.max().item()) == 0:
            return torch.zeros(batch_size, self.config.temporal_hidden_dim, device=device)

        history_questions = batch["history_question_indices"].clone()
        history_questions[history_questions < 0] = self.config.num_questions
        question_repr = self.question_embedding(history_questions)
        correctness_repr = self.correctness_embedding(batch["history_correctness"])
        if self.config.use_time_features:
            time_features = self._normalize_time_features(batch)
            time_repr = self.time_projection(time_features)
        else:
            time_repr = torch.zeros_like(question_repr)
        history_repr = torch.cat([question_repr, correctness_repr, time_repr], dim=-1)
        history_repr = self.history_projection(history_repr)
        history_repr = history_repr * batch["history_masks"].unsqueeze(-1)

        packed = nn.utils.rnn.pack_padded_sequence(
            history_repr,
            lengths.clamp_min(1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.temporal_encoder(packed)
        unpacked_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=max_length,
        )
        gather_index = (lengths.clamp_min(1) - 1).view(-1, 1, 1).expand(-1, 1, self.config.temporal_hidden_dim)
        last_valid_state = unpacked_output.gather(1, gather_index).squeeze(1)
        last_valid_state = last_valid_state * (lengths > 0).unsqueeze(-1)
        return last_valid_state

    def _normalize_time_features(self, batch: dict[str, Any]) -> torch.Tensor:
        elapsed_scale = max(float(self.config.elapsed_time_scale), 1e-6)
        attempt_scale = max(float(self.config.attempt_count_scale), 1e-6)
        elapsed = torch.log1p(batch["history_elapsed_times"]) / math.log1p(elapsed_scale)
        attempts = torch.log1p(batch["history_attempt_counts"]) / math.log1p(attempt_scale)
        return torch.stack([elapsed, attempts], dim=-1)

    def _pool_concepts(
        self,
        concept_embeddings: torch.Tensor,
        concept_indices: torch.Tensor,
        concept_mask: torch.Tensor,
    ) -> torch.Tensor:
        embedded = concept_embeddings[concept_indices]
        if self.target_concept_attention is not None:
            scores = self.target_concept_attention(embedded).squeeze(-1)
            scores = scores.masked_fill(concept_mask <= 0, -1e9)
            weights = torch.softmax(scores, dim=1) * concept_mask
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            return (embedded * weights.unsqueeze(-1)).sum(dim=1)
        masked = embedded * concept_mask.unsqueeze(-1)
        denom = concept_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return masked.sum(dim=1) / denom
