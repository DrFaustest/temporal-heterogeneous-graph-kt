"""Baseline models with a shared interface."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from thgkt.graph.artifacts import EdgeRelation, HeteroGraphArtifacts
from thgkt.models.base import BaseModel
from thgkt.models.math_utils import bce_loss, dot, random_vector, sigmoid, zeros


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    random_seed: int = 7
    hidden_dim: int = 64
    question_embedding_dim: int = 32
    correctness_embedding_dim: int = 8
    attention_heads: int = 4
    dropout: float = 0.1
    gradient_clip_norm: float = 5.0
    elapsed_time_scale: float = 5000.0
    attempt_count_scale: float = 5.0



def _safe_mean(values: list[float] | list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0



def _recency_weighted_accuracy(values: list[int], decay: float = 0.7) -> float:
    if not values:
        return 0.0
    weighted_sum = 0.0
    total_weight = 0.0
    for offset, value in enumerate(reversed(values)):
        weight = decay**offset
        weighted_sum += weight * int(value)
        total_weight += weight
    return weighted_sum / max(total_weight, 1e-8)



def _recent_correct_streak(values: list[int]) -> float:
    if not values:
        return 0.0
    streak = 0
    for value in reversed(values):
        if int(value) != 1:
            break
        streak += 1
    return streak / max(1, len(values))



def _gap_score(match_positions: list[int], history_length: int) -> float:
    if not match_positions:
        return 0.0
    steps_since = max(0, history_length - 1 - match_positions[-1])
    return 1.0 / (1.0 + steps_since)



def _ema(values: list[int], smoothing: float = 0.6) -> float:
    if not values:
        return 0.0
    estimate = float(values[0])
    for value in values[1:]:
        estimate = smoothing * float(value) + (1.0 - smoothing) * estimate
    return estimate



def _concept_match_stats(
    history_concepts: list[list[int]],
    history_correct: list[int],
    target_concepts: list[int],
) -> dict[str, float]:
    target_set = {int(concept) for concept in target_concepts if int(concept) >= 0}
    if not target_set:
        return {"accuracy": 0.0, "recency_accuracy": 0.0, "recent_streak": 0.0, "gap_score": 0.0, "mastery": 0.0}

    matched_correct: list[int] = []
    matched_positions: list[int] = []
    for position, (concepts, correctness) in enumerate(zip(history_concepts, history_correct)):
        concepts_set = {int(concept) for concept in concepts if int(concept) >= 0}
        if target_set.intersection(concepts_set):
            matched_correct.append(int(correctness))
            matched_positions.append(position)

    if not matched_correct:
        return {"accuracy": 0.0, "recency_accuracy": 0.0, "recent_streak": 0.0, "gap_score": 0.0, "mastery": 0.0}

    return {
        "accuracy": _safe_mean(matched_correct),
        "recency_accuracy": _recency_weighted_accuracy(matched_correct),
        "recent_streak": _recent_correct_streak(matched_correct),
        "gap_score": _gap_score(matched_positions, len(history_correct)),
        "mastery": _ema(matched_correct),
    }



def _same_question_stats(history_questions: list[int], history_correct: list[int], target_question: int) -> dict[str, float]:
    matched_correct = [int(correctness) for question, correctness in zip(history_questions, history_correct) if int(question) == target_question]
    if not matched_correct:
        return {"repeat_rate": 0.0, "accuracy": 0.0}
    return {
        "repeat_rate": min(len(matched_correct) / 5.0, 1.0),
        "accuracy": _safe_mean(matched_correct),
    }



def _resolve_attention_heads(hidden_dim: int, requested_heads: int) -> int:
    upper = max(1, min(hidden_dim, requested_heads))
    for heads in range(upper, 0, -1):
        if hidden_dim % heads == 0:
            return heads
    return 1


class LogisticBaseline(BaseModel):
    """Feature-engineered logistic next-response baseline."""

    def __init__(self, num_questions: int, num_concepts: int, config: BaselineConfig | None = None) -> None:
        self.num_questions = max(1, num_questions)
        self.num_concepts = max(1, num_concepts)
        self.config = config or BaselineConfig()
        rng = random.Random(self.config.random_seed)
        self.weights = random_vector(12, rng)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        features = [self._features_for_example(batch, index) for index in range(len(batch["targets"]))]
        logits = [dot(self.weights, feature) for feature in features]
        probs = [sigmoid(logit) for logit in logits]
        return {
            "logits": logits,
            "probs": probs,
            "targets": list(batch["targets"]),
            "aux_outputs": {"features": features},
            "debug_info": {"model_name": "LogisticBaseline", "feature_count": len(self.weights)},
        }

    def train_step(self, batch: dict[str, Any], learning_rate: float = 0.05) -> dict[str, Any]:
        outputs = self.forward(batch)
        grads = zeros(len(self.weights))
        losses: list[float] = []
        for feature, probability, target in zip(outputs["aux_outputs"]["features"], outputs["probs"], outputs["targets"]):
            error = probability - int(target)
            losses.append(bce_loss(probability, int(target)))
            for index, value in enumerate(feature):
                grads[index] += error * value
        batch_size = max(1, len(outputs["targets"]))
        for index in range(len(self.weights)):
            self.weights[index] -= learning_rate * grads[index] / batch_size
        return {
            "loss": sum(losses) / batch_size,
            "outputs": outputs,
        }

    def state_dict(self) -> dict[str, Any]:
        return {"weights": list(self.weights)}

    def _features_for_example(self, batch: dict[str, Any], index: int) -> list[float]:
        history_length = int(batch["history_lengths"][index])
        history_correct = [int(item) for item in batch["history_correctness"][index][:history_length]]
        history_questions = [int(item) for item in batch["history_question_indices"][index][:history_length]]
        history_concepts = [
            [int(concept) for concept in concepts]
            for concepts in batch.get("history_concept_indices", [[] for _ in batch["targets"]])[index][:history_length]
        ]
        target_question = int(batch["target_question_index"][index])
        target_concepts = [int(item) for item in batch["target_concept_indices"][index]]

        same_question = _same_question_stats(history_questions, history_correct, target_question)
        concept_stats = _concept_match_stats(history_concepts, history_correct, target_concepts)
        mean_correct = _safe_mean(history_correct)
        last_correct = float(history_correct[-1]) if history_correct else 0.0
        target_question_norm = float(target_question) / max(1, self.num_questions - 1)

        return [
            1.0,
            min(history_length / 50.0, 1.0),
            mean_correct,
            _recency_weighted_accuracy(history_correct),
            last_correct,
            target_question_norm,
            same_question["repeat_rate"],
            same_question["accuracy"],
            concept_stats["accuracy"],
            concept_stats["recency_accuracy"],
            concept_stats["recent_streak"],
            concept_stats["gap_score"],
        ]


class GraphOnlyModel(BaseModel):
    """Graph-statistics baseline with correctness-aware sequence features."""

    def __init__(self, graph: HeteroGraphArtifacts, config: BaselineConfig | None = None) -> None:
        self.graph = graph
        self.config = config or BaselineConfig()
        rng = random.Random(self.config.random_seed)
        self.weights = random_vector(10, rng)
        self.student_counts = self._student_answer_counts()
        self.question_counts = self._question_answer_counts()
        self.student_concept_exposures = self._student_concept_exposure_counts()
        self.concept_outdegrees = self._concept_outdegrees()
        self.max_student_count = max(self.student_counts.values(), default=1)
        self.max_question_count = max(self.question_counts.values(), default=1)
        self.max_concept_outdegree = max(self.concept_outdegrees.values(), default=1)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        features = [self._features_for_example(batch, index) for index in range(len(batch["targets"]))]
        logits = [dot(self.weights, feature) for feature in features]
        probs = [sigmoid(logit) for logit in logits]
        return {
            "logits": logits,
            "probs": probs,
            "targets": list(batch["targets"]),
            "aux_outputs": {"features": features},
            "debug_info": {"model_name": "GraphOnlyModel", "feature_count": len(self.weights)},
        }

    def train_step(self, batch: dict[str, Any], learning_rate: float = 0.05) -> dict[str, Any]:
        outputs = self.forward(batch)
        grads = zeros(len(self.weights))
        losses: list[float] = []
        for feature, probability, target in zip(outputs["aux_outputs"]["features"], outputs["probs"], outputs["targets"]):
            error = probability - int(target)
            losses.append(bce_loss(probability, int(target)))
            for index, value in enumerate(feature):
                grads[index] += error * value
        batch_size = max(1, len(outputs["targets"]))
        for index in range(len(self.weights)):
            self.weights[index] -= learning_rate * grads[index] / batch_size
        return {"loss": sum(losses) / batch_size, "outputs": outputs}

    def state_dict(self) -> dict[str, Any]:
        return {"weights": list(self.weights)}

    def _features_for_example(self, batch: dict[str, Any], index: int) -> list[float]:
        student_id = str(batch["student_ids"][index])
        history_length = int(batch["history_lengths"][index])
        history_correct = [int(item) for item in batch["history_correctness"][index][:history_length]]
        history_questions = [int(item) for item in batch["history_question_indices"][index][:history_length]]
        history_concepts = [
            [int(concept) for concept in concepts]
            for concepts in batch.get("history_concept_indices", [[] for _ in batch["targets"]])[index][:history_length]
        ]
        target_question = int(batch["target_question_index"][index])
        target_concepts = [int(item) for item in batch["target_concept_indices"][index]]
        student_count = self.student_counts.get(student_id, 0) / max(1, self.max_student_count)
        question_count = self.question_counts.get(target_question, 0) / max(1, self.max_question_count)
        exposure_rate = 0.0
        prereq_mean = 0.0
        if target_concepts:
            exposure_rate = sum(
                self.student_concept_exposures.get((student_id, concept_index), 0) for concept_index in target_concepts
            ) / float(len(target_concepts))
            exposure_rate /= max(1, self.max_student_count)
            prereq_mean = sum(self.concept_outdegrees.get(concept_index, 0) for concept_index in target_concepts) / float(len(target_concepts))
            prereq_mean /= max(1, self.max_concept_outdegree)

        same_question = _same_question_stats(history_questions, history_correct, target_question)
        concept_stats = _concept_match_stats(history_concepts, history_correct, target_concepts)
        return [
            1.0,
            student_count,
            question_count,
            exposure_rate,
            prereq_mean,
            same_question["accuracy"],
            concept_stats["accuracy"],
            concept_stats["recency_accuracy"],
            concept_stats["mastery"],
            concept_stats["gap_score"],
        ]

    def _student_answer_counts(self) -> dict[str, int]:
        relation = EdgeRelation("student", "answered", "question").key
        reverse_students = {index: student_id for student_id, index in self.graph.node_maps["student"].items()}
        counts: dict[str, int] = {}
        for student_index in self.graph.edges[relation].src_indices:
            student_id = reverse_students[int(student_index)]
            counts[student_id] = counts.get(student_id, 0) + 1
        return counts

    def _question_answer_counts(self) -> dict[int, int]:
        relation = EdgeRelation("student", "answered", "question").key
        counts: dict[int, int] = {}
        for question_index in self.graph.edges[relation].dst_indices:
            counts[int(question_index)] = counts.get(int(question_index), 0) + 1
        return counts

    def _student_concept_exposure_counts(self) -> dict[tuple[str, int], int]:
        relation = EdgeRelation("student", "exposure_to", "concept").key
        if relation not in self.graph.edges:
            return {}
        reverse_students = {index: student_id for student_id, index in self.graph.node_maps["student"].items()}
        counts: dict[tuple[str, int], int] = {}
        for student_index, concept_index in zip(
            self.graph.edges[relation].src_indices,
            self.graph.edges[relation].dst_indices,
        ):
            key = (reverse_students[int(student_index)], int(concept_index))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _concept_outdegrees(self) -> dict[int, int]:
        relation = EdgeRelation("concept", "prerequisite_of", "concept").key
        counts: dict[int, int] = {}
        if relation not in self.graph.edges:
            return counts
        for concept_index in self.graph.edges[relation].src_indices:
            counts[int(concept_index)] = counts.get(int(concept_index), 0) + 1
        return counts


class _SequenceTorchBaseline(nn.Module):
    def __init__(self, num_questions: int, config: BaselineConfig | None = None) -> None:
        super().__init__()
        self.num_questions = max(1, num_questions)
        self.config = config or BaselineConfig()
        self.gradient_clip_norm = float(self.config.gradient_clip_norm)
        self._optimizer: torch.optim.Optimizer | None = None

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        if torch.is_tensor(batch["history_question_indices"]):
            return self._forward_tensors(batch)
        tensor_batch = self._tensorize_batch(batch)
        outputs = self._forward_tensors(tensor_batch)
        return self._detached_outputs(outputs, targets=list(batch["targets"]))

    def train_step(self, batch: dict[str, Any], learning_rate: float = 0.05) -> dict[str, Any]:
        tensor_batch = self._tensorize_batch(batch)
        optimizer = self._optimizer_for(learning_rate)
        self.train()
        optimizer.zero_grad()
        outputs = self._forward_tensors(tensor_batch)
        loss = nn.functional.binary_cross_entropy_with_logits(outputs["logits"], tensor_batch["targets"].float())
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_norm)
        optimizer.step()
        return {
            "loss": float(loss.detach().cpu().item()),
            "outputs": self._detached_outputs(outputs, targets=list(batch["targets"])),
        }

    def _optimizer_for(self, learning_rate: float) -> torch.optim.Optimizer:
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        else:
            for group in self._optimizer.param_groups:
                group["lr"] = learning_rate
        return self._optimizer

    def _tensorize_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        return {
            "history_question_indices": torch.tensor(batch["history_question_indices"], dtype=torch.long, device=device),
            "history_correctness": torch.tensor(batch["history_correctness"], dtype=torch.long, device=device),
            "history_elapsed_times": torch.tensor(batch["history_elapsed_times"], dtype=torch.float32, device=device),
            "history_attempt_counts": torch.tensor(batch["history_attempt_counts"], dtype=torch.float32, device=device),
            "history_masks": torch.tensor(batch["history_masks"], dtype=torch.float32, device=device),
            "history_lengths": torch.tensor(batch["history_lengths"], dtype=torch.long, device=device),
            "target_question_index": torch.tensor(batch["target_question_index"], dtype=torch.long, device=device),
            "targets": torch.tensor(batch["targets"], dtype=torch.float32, device=device),
        }

    def _detached_outputs(self, outputs: dict[str, Any], *, targets: list[int]) -> dict[str, Any]:
        return {
            "logits": outputs["logits"].detach().cpu().view(-1).tolist(),
            "probs": outputs["probs"].detach().cpu().view(-1).tolist(),
            "targets": targets,
            "aux_outputs": {},
            "debug_info": dict(outputs["debug_info"]),
        }

    def _forward_tensors(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        raise NotImplementedError


class DKTBaseline(_SequenceTorchBaseline):
    """Question-response GRU baseline with side-channel timing features."""

    def __init__(self, num_questions: int, config: BaselineConfig | None = None) -> None:
        super().__init__(num_questions, config=config)
        self.hidden_dim = max(8, int(self.config.hidden_dim))
        self.question_embedding = nn.Embedding(self.num_questions + 1, self.config.question_embedding_dim, padding_idx=0)
        self.correctness_embedding = nn.Embedding(3, self.config.correctness_embedding_dim, padding_idx=2)
        self.dropout = nn.Dropout(self.config.dropout)
        self.gru = nn.GRU(self.config.question_embedding_dim + self.config.correctness_embedding_dim + 2, self.hidden_dim, batch_first=True)
        self.output = nn.Linear(self.hidden_dim + self.config.question_embedding_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        generator = torch.Generator(device=self.question_embedding.weight.device)
        generator.manual_seed(self.config.random_seed)
        for name, parameter in self.named_parameters():
            if parameter.ndim > 1:
                parameter.data.uniform_(-0.1, 0.1, generator=generator)
            elif name.endswith("bias"):
                parameter.data.zero_()
            else:
                parameter.data.uniform_(-0.05, 0.05, generator=generator)

    def _forward_tensors(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        device = self.question_embedding.weight.device
        history_questions = batch["history_question_indices"].to(device)
        history_correctness = batch["history_correctness"].to(device)
        history_elapsed = batch["history_elapsed_times"].to(device)
        history_attempts = batch["history_attempt_counts"].to(device)
        history_masks = batch["history_masks"].to(device)
        history_lengths = batch["history_lengths"].to(device)
        target_questions = batch["target_question_index"].to(device)

        question_tokens = torch.where(history_questions >= 0, history_questions + 1, torch.zeros_like(history_questions))
        correctness_pad = torch.full_like(history_correctness, 2)
        correctness_tokens = torch.where(history_masks > 0, history_correctness.long(), correctness_pad)
        question_repr = self.question_embedding(question_tokens)
        correctness_repr = self.correctness_embedding(correctness_tokens)
        elapsed_feature = torch.clamp(history_elapsed / float(self.config.elapsed_time_scale), 0.0, 1.0).unsqueeze(-1)
        attempt_feature = torch.clamp(history_attempts / float(self.config.attempt_count_scale), 0.0, 1.0).unsqueeze(-1)
        step_inputs = torch.cat([question_repr, correctness_repr, elapsed_feature, attempt_feature], dim=-1)
        if self.training and self.config.dropout > 0.0:
            step_inputs = self.dropout(step_inputs)

        batch_size = history_questions.shape[0]
        final_hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        non_empty = history_lengths > 0
        if torch.any(non_empty):
            packed = nn.utils.rnn.pack_padded_sequence(
                step_inputs[non_empty],
                history_lengths[non_empty].detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.gru(packed)
            final_hidden[non_empty] = hidden[-1]

        target_tokens = torch.clamp(target_questions.long() + 1, min=0)
        target_repr = self.question_embedding(target_tokens)
        if self.training and self.config.dropout > 0.0:
            final_hidden = self.dropout(final_hidden)
        logits = self.output(torch.cat([final_hidden, target_repr], dim=-1)).squeeze(-1)
        probs = torch.sigmoid(logits)
        return {
            "logits": logits,
            "probs": probs,
            "targets": batch["targets"].float().to(device),
            "aux_outputs": {
                "final_hidden": final_hidden,
                "target_question_repr": target_repr,
            },
            "debug_info": {
                "model_name": "DKTBaseline",
                "hidden_dim": self.hidden_dim,
                "question_embedding_dim": self.config.question_embedding_dim,
                "gradient_clip_norm": self.gradient_clip_norm,
            },
        }


class SAKTBaseline(_SequenceTorchBaseline):
    """Self-attentive knowledge tracing baseline."""

    def __init__(self, num_questions: int, config: BaselineConfig | None = None) -> None:
        super().__init__(num_questions, config=config)
        self.hidden_dim = max(8, int(self.config.hidden_dim))
        self.attention_heads = _resolve_attention_heads(self.hidden_dim, int(self.config.attention_heads))
        self.question_embedding = nn.Embedding(self.num_questions + 1, self.hidden_dim, padding_idx=0)
        self.interaction_embedding = nn.Embedding((self.num_questions + 1) * 2 + 1, self.hidden_dim, padding_idx=0)
        self.side_projection = nn.Linear(2, self.hidden_dim)
        self.input_dropout = nn.Dropout(self.config.dropout)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.attention_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(self.hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim * 2, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        generator = torch.Generator(device=self.question_embedding.weight.device)
        generator.manual_seed(self.config.random_seed)
        for name, parameter in self.named_parameters():
            if parameter.ndim > 1:
                parameter.data.uniform_(-0.1, 0.1, generator=generator)
            elif name.endswith("bias"):
                parameter.data.zero_()
            else:
                parameter.data.uniform_(-0.05, 0.05, generator=generator)

    def _forward_tensors(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        device = self.question_embedding.weight.device
        history_questions = batch["history_question_indices"].to(device)
        history_correctness = batch["history_correctness"].to(device)
        history_elapsed = batch["history_elapsed_times"].to(device)
        history_attempts = batch["history_attempt_counts"].to(device)
        history_masks = batch["history_masks"].to(device)
        history_lengths = batch["history_lengths"].to(device)
        target_questions = batch["target_question_index"].to(device)

        history_question_tokens = torch.where(history_questions >= 0, history_questions + 1, torch.zeros_like(history_questions))
        interaction_tokens = history_question_tokens + history_correctness.long().clamp(0, 1) * (self.num_questions + 1)
        interaction_tokens = torch.where(history_masks > 0, interaction_tokens, torch.zeros_like(interaction_tokens))
        interaction_repr = self.interaction_embedding(interaction_tokens)
        elapsed_feature = torch.clamp(history_elapsed / float(self.config.elapsed_time_scale), 0.0, 1.0)
        attempt_feature = torch.clamp(history_attempts / float(self.config.attempt_count_scale), 0.0, 1.0)
        side_repr = self.side_projection(torch.stack([elapsed_feature, attempt_feature], dim=-1))
        history_repr = interaction_repr + side_repr
        if self.training and self.config.dropout > 0.0:
            history_repr = self.input_dropout(history_repr)

        target_tokens = torch.clamp(target_questions.long() + 1, min=0)
        target_repr = self.question_embedding(target_tokens)
        attended = torch.zeros(target_repr.shape[0], self.hidden_dim, device=device)
        attention_weights = torch.zeros_like(history_masks)
        non_empty = history_lengths > 0
        if torch.any(non_empty):
            query = target_repr[non_empty].unsqueeze(1)
            key_value = history_repr[non_empty]
            key_padding_mask = history_masks[non_empty] <= 0
            attn_output, attn_weights = self.attention(
                query=query,
                key=key_value,
                value=key_value,
                key_padding_mask=key_padding_mask,
                need_weights=True,
            )
            attn_output = self.attention_norm(attn_output + query)
            ff_output = self.feed_forward(attn_output)
            attn_output = self.output_norm(attn_output + ff_output)
            attended[non_empty] = attn_output.squeeze(1)
            attention_weights[non_empty] = attn_weights.squeeze(1)

        combined = torch.cat([attended, target_repr], dim=-1)
        if self.training and self.config.dropout > 0.0:
            combined = self.input_dropout(combined)
        logits = self.output(combined).squeeze(-1)
        probs = torch.sigmoid(logits)
        return {
            "logits": logits,
            "probs": probs,
            "targets": batch["targets"].float().to(device),
            "aux_outputs": {
                "attention_weights": attention_weights,
                "attended_history": attended,
                "target_question_repr": target_repr,
            },
            "debug_info": {
                "model_name": "SAKTBaseline",
                "hidden_dim": self.hidden_dim,
                "attention_heads": self.attention_heads,
                "gradient_clip_norm": self.gradient_clip_norm,
            },
        }
