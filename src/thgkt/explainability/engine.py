"""Explainability engine for THGKT outputs."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch

from thgkt.data.io import save_json
from thgkt.explainability.artifacts import ExplainabilityArtifacts
from thgkt.explainability.plotting import save_bar_chart_svg
from thgkt.training.runner import tensorize_batch


class ExplainabilityEngine:
    def concept_importance(
        self,
        model: Any,
        batch: dict[str, Any],
        *,
        context: dict[str, Any],
        concept_label_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        model.eval()
        tensor_batch = tensorize_batch(batch, context=context)
        with torch.no_grad():
            base_outputs = model(tensor_batch)
            base_probs = torch.sigmoid(base_outputs["logits"]).detach().cpu().tolist()

        examples: list[dict[str, Any]] = []
        for example_index, base_prob in enumerate(base_probs):
            concept_indices = _active_concepts(
                tensor_batch["target_concept_indices"][example_index],
                tensor_batch["target_concept_mask"][example_index],
            )
            concept_scores: list[dict[str, Any]] = []
            for concept_position, concept_index in enumerate(concept_indices):
                ablated_batch = _clone_tensor_batch(tensor_batch)
                ablated_batch["target_concept_mask"][example_index, concept_position] = 0.0
                with torch.no_grad():
                    ablated_outputs = model(ablated_batch)
                    ablated_prob = float(torch.sigmoid(ablated_outputs["logits"])[example_index].cpu().item())
                label = concept_label_map.get(concept_index, str(concept_index)) if concept_label_map else str(concept_index)
                concept_scores.append(
                    {
                        "concept_index": int(concept_index),
                        "concept_label": label,
                        "base_probability": float(base_prob),
                        "ablated_probability": ablated_prob,
                        "importance": float(base_prob - ablated_prob),
                    }
                )
            concept_scores.sort(key=lambda item: abs(item["importance"]), reverse=True)
            examples.append(
                {
                    "student_id": str(batch["student_ids"][example_index]),
                    "target_interaction_id": str(batch["target_interaction_ids"][example_index]),
                    "target_probability": float(base_prob),
                    "target_correct": int(batch["targets"][example_index]),
                    "concept_scores": concept_scores,
                }
            )

        return {
            "method": "leave_one_target_concept_out",
            "examples": examples,
        }

    def prerequisite_influence(
        self,
        model: Any,
        batch: dict[str, Any],
        *,
        context: dict[str, Any],
        concept_label_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        model.eval()
        tensor_batch = tensorize_batch(batch, context=context)
        with torch.no_grad():
            base_probs = torch.sigmoid(model(tensor_batch)["logits"]).detach().cpu().tolist()

        no_prereq_context = dict(context)
        graph_without_prereq = copy.deepcopy(context["graph_data"])
        if ("concept", "prerequisite_of", "concept") in graph_without_prereq.edge_index_dict:
            del graph_without_prereq[("concept", "prerequisite_of", "concept")]
        if ("concept", "rev_prerequisite_of", "concept") in graph_without_prereq.edge_index_dict:
            del graph_without_prereq[("concept", "rev_prerequisite_of", "concept")]
        no_prereq_context["graph_data"] = graph_without_prereq
        ablated_batch = tensorize_batch(batch, context=no_prereq_context)
        with torch.no_grad():
            no_prereq_probs = torch.sigmoid(model(ablated_batch)["logits"]).detach().cpu().tolist()

        predecessors = _concept_predecessors(context["graph_data"])
        examples: list[dict[str, Any]] = []
        for example_index, base_prob in enumerate(base_probs):
            target_concepts = _active_concepts(
                tensor_batch["target_concept_indices"][example_index],
                tensor_batch["target_concept_mask"][example_index],
            )
            prerequisite_labels = sorted(
                {
                    concept_label_map.get(source, str(source)) if concept_label_map else str(source)
                    for target_concept in target_concepts
                    for source in predecessors.get(int(target_concept), set())
                }
            )
            examples.append(
                {
                    "student_id": str(batch["student_ids"][example_index]),
                    "target_interaction_id": str(batch["target_interaction_ids"][example_index]),
                    "base_probability": float(base_prob),
                    "no_prerequisite_probability": float(no_prereq_probs[example_index]),
                    "prerequisite_influence": float(base_prob - no_prereq_probs[example_index]),
                    "prerequisite_concepts": prerequisite_labels,
                }
            )
        return {
            "method": "drop_prerequisite_edges",
            "examples": examples,
        }

    def export_concept_importance_artifacts(
        self,
        report: dict[str, Any],
        output_dir: str | Path,
    ) -> ExplainabilityArtifacts:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        report_path = output_path / "concept_importance.json"
        plot_path = output_path / "concept_importance.svg"
        save_json(report, report_path)

        top_example = report["examples"][0]
        labels = [item["concept_label"] for item in top_example["concept_scores"]]
        values = [float(item["importance"]) for item in top_example["concept_scores"]]
        save_bar_chart_svg(
            labels,
            values,
            plot_path,
            title=f"Concept Importance: {top_example['target_interaction_id']}",
        )
        return ExplainabilityArtifacts(
            report_path=str(report_path),
            plot_path=str(plot_path),
            method=str(report["method"]),
            metadata={
                "num_examples": len(report["examples"]),
                "top_target_interaction_id": top_example["target_interaction_id"],
            },
        )


def _clone_tensor_batch(batch: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def _active_concepts(concept_indices: torch.Tensor, concept_mask: torch.Tensor) -> list[int]:
    active = []
    for index, mask in zip(concept_indices.detach().cpu().tolist(), concept_mask.detach().cpu().tolist()):
        if float(mask) > 0.0:
            active.append(int(index))
    return active


def _concept_predecessors(graph_data: Any) -> dict[int, set[int]]:
    predecessors: dict[int, set[int]] = {}
    edge_key = ("concept", "prerequisite_of", "concept")
    if edge_key not in graph_data.edge_index_dict:
        return predecessors
    edge_index = graph_data[edge_key].edge_index.detach().cpu().tolist()
    for source, target in zip(edge_index[0], edge_index[1]):
        predecessors.setdefault(int(target), set()).add(int(source))
    return predecessors
