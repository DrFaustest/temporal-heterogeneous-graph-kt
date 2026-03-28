"""Training, evaluation, checkpointing, and sequence loaders."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from thgkt.data.io import load_json, save_json
from thgkt.models.base import BaseModel
from thgkt.sequences import SequenceExample, collate_sequence_batch
from thgkt.training.artifacts import TrainingRunArtifacts
from thgkt.training.metrics import compute_classification_metrics


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    run_name: str
    run_dir: str
    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-3
    device: str = "cpu"
    shuffle_train: bool = True
    random_seed: int = 42
    early_stopping_patience: int | None = 3


class Evaluator:
    def evaluate(self, model: Any, loader: Iterable[dict[str, Any]], context: dict[str, Any] | None = None) -> dict[str, float]:
        probs: list[float] = []
        targets: list[int] = []
        for batch in loader:
            outputs = _forward_model(model, batch, context=context, train_mode=False)
            probs.extend(float(item) for item in outputs["probs"])
            targets.extend(int(item) for item in outputs["targets"])
        return compute_classification_metrics(probs, targets)


class Trainer:
    def __init__(self, evaluator: Evaluator | None = None) -> None:
        self.evaluator = evaluator or Evaluator()

    def fit(
        self,
        model: Any,
        train_examples: Iterable[SequenceExample],
        val_examples: Iterable[SequenceExample],
        config: TrainingConfig,
        *,
        context: dict[str, Any] | None = None,
        progress_callback: Any | None = None,
    ) -> TrainingRunArtifacts:
        run_dir = Path(config.run_dir)
        checkpoint_dir = run_dir / "checkpoints"
        metrics_dir = run_dir / "metrics"
        config_dir = run_dir / "config"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)

        optimizer = _make_optimizer(model, config.learning_rate)
        train_history: list[dict[str, float]] = []
        val_history: list[dict[str, float]] = []
        best_val_metrics: dict[str, float] | None = None
        best_epoch = 0
        completed_epochs = 0
        epochs_without_improvement = 0
        stopped_early = False
        best_checkpoint_path = checkpoint_dir / "best_checkpoint.pt"

        train_examples_list = list(train_examples)
        val_examples_list = list(val_examples)
        run_context = {**(context or {}), "classic_learning_rate": config.learning_rate}
        fit_started_at = time.perf_counter()
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "training_started",
                    "current_epoch": 0,
                    "total_epochs": config.epochs,
                    "elapsed_seconds": 0.0,
                }
            )

        for epoch in range(config.epochs):
            epoch_started_at = time.perf_counter()
            train_loader = build_sequence_loader(
                train_examples_list,
                batch_size=config.batch_size,
                shuffle=config.shuffle_train,
                random_seed=config.random_seed + epoch,
            )
            train_metrics = self.train_epoch(model, train_loader, optimizer=optimizer, context=run_context)
            val_loader = build_sequence_loader(
                val_examples_list,
                batch_size=config.batch_size,
                shuffle=False,
                random_seed=config.random_seed,
            )
            val_metrics = self.evaluator.evaluate(model, val_loader, context=run_context)
            train_history.append({"epoch": float(epoch + 1), **train_metrics})
            val_history.append({"epoch": float(epoch + 1), **val_metrics})
            completed_epochs = epoch + 1
            if best_val_metrics is None or val_metrics["auc"] > best_val_metrics["auc"]:
                best_val_metrics = dict(val_metrics)
                best_epoch = completed_epochs
                epochs_without_improvement = 0
                save_checkpoint(model, best_checkpoint_path, extra={"epoch": completed_epochs, "val_metrics": val_metrics})
            else:
                epochs_without_improvement += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "training_epoch_completed",
                        "current_epoch": completed_epochs,
                        "total_epochs": config.epochs,
                        "elapsed_seconds": time.perf_counter() - fit_started_at,
                        "epoch_seconds": time.perf_counter() - epoch_started_at,
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                    }
                )
            if config.early_stopping_patience is not None and epochs_without_improvement >= config.early_stopping_patience:
                stopped_early = completed_epochs < config.epochs
                break

        assert best_val_metrics is not None
        load_checkpoint(model, best_checkpoint_path)
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "training_completed",
                    "current_epoch": completed_epochs,
                    "total_epochs": config.epochs,
                    "elapsed_seconds": time.perf_counter() - fit_started_at,
                    "best_val_metrics": best_val_metrics,
                    "best_epoch": best_epoch,
                    "stopped_early": stopped_early,
                }
            )
        metrics_path = metrics_dir / "metrics.json"
        config_snapshot_path = config_dir / "training_config.json"
        save_json(
            {
                "train_history": train_history,
                "val_history": val_history,
                "best_val_metrics": best_val_metrics,
                "best_epoch": best_epoch,
                "completed_epochs": completed_epochs,
                "stopped_early": stopped_early,
            },
            metrics_path,
        )
        save_json(
            {
                "run_name": config.run_name,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "device": config.device,
                "random_seed": config.random_seed,
                "early_stopping_patience": config.early_stopping_patience,
                "early_stopping_metric": "auc",
            },
            config_snapshot_path,
        )
        artifacts = TrainingRunArtifacts(
            run_name=config.run_name,
            run_dir=str(run_dir),
            model_name=model.__class__.__name__,
            train_history=tuple(train_history),
            val_history=tuple(val_history),
            best_val_metrics=dict(best_val_metrics),
            checkpoint_path=str(best_checkpoint_path),
            metrics_path=str(metrics_path),
            config_snapshot_path=str(config_snapshot_path),
            extra={
                "best_epoch": best_epoch,
                "completed_epochs": completed_epochs,
                "stopped_early": stopped_early,
            },
        )
        save_json(artifacts.to_dict(), run_dir / "run_artifacts.json")
        return artifacts

    def train_epoch(
        self,
        model: Any,
        loader: Iterable[dict[str, Any]],
        *,
        optimizer: torch.optim.Optimizer | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        losses: list[float] = []
        probs: list[float] = []
        targets: list[int] = []
        for batch in loader:
            if isinstance(model, BaseModel):
                result = model.train_step(batch, learning_rate=float((context or {}).get("classic_learning_rate", 0.05)))
                outputs = result["outputs"]
                losses.append(float(result["loss"]))
            else:
                assert optimizer is not None
                model.train()
                optimizer.zero_grad()
                outputs = _forward_model(model, batch, context=context, train_mode=True)
                loss = outputs["loss_tensor"]
                loss.backward()
                clip_norm = getattr(model, "gradient_clip_norm", None)
                if clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
            probs.extend(float(item) for item in outputs["probs"])
            targets.extend(int(item) for item in outputs["targets"])
        metrics = compute_classification_metrics(probs, targets)
        metrics["loss"] = sum(losses) / max(1, len(losses))
        return metrics


def build_sequence_loader(
    examples: Iterable[SequenceExample],
    *,
    batch_size: int,
    shuffle: bool,
    random_seed: int,
) -> list[dict[str, Any]]:
    example_list = list(examples)
    if shuffle:
        rng = random.Random(random_seed)
        rng.shuffle(example_list)
    batches: list[dict[str, Any]] = []
    for start in range(0, len(example_list), batch_size):
        batches.append(collate_sequence_batch(example_list[start : start + batch_size]))
    return batches


def save_checkpoint(model: Any, path: str | Path, extra: dict[str, Any] | None = None) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_name": model.__class__.__name__, "extra": dict(extra or {})}
    if isinstance(model, nn.Module):
        payload["kind"] = "torch"
        payload["state_dict"] = model.state_dict()
        torch.save(payload, checkpoint_path)
    else:
        payload["kind"] = "classic"
        payload["state_dict"] = model.state_dict()
        checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return checkpoint_path


def load_checkpoint(model: Any, path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if isinstance(model, nn.Module):
        # Project checkpoints are locally generated and may include lazy PyG
        # parameters for edge types that were disabled during training.
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"])
        return payload
    payload = load_json(checkpoint_path)
    if hasattr(model, "load_state_dict"):
        model.load_state_dict(payload["state_dict"])
    else:
        for key, value in payload["state_dict"].items():
            setattr(model, key, value)
    return payload


def _make_optimizer(model: Any, learning_rate: float) -> torch.optim.Optimizer | None:
    if isinstance(model, nn.Module):
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    return None


def _forward_model(
    model: Any,
    batch: dict[str, Any],
    *,
    context: dict[str, Any] | None,
    train_mode: bool,
) -> dict[str, Any]:
    if isinstance(model, BaseModel):
        return model.forward(batch)

    tensor_batch = tensorize_batch(batch, context=context)
    previous_mode = model.training if isinstance(model, nn.Module) else None
    if isinstance(model, nn.Module):
        model.train(mode=train_mode)
    with torch.set_grad_enabled(train_mode):
        outputs = model(tensor_batch)
    if previous_mode is not None:
        model.train(mode=bool(previous_mode))
    logits_tensor = outputs["logits"]
    probs_tensor = outputs.get("probs")
    if probs_tensor is None:
        probs_tensor = torch.sigmoid(logits_tensor)
    elif not torch.is_tensor(probs_tensor):
        probs_tensor = torch.as_tensor(probs_tensor, dtype=logits_tensor.dtype, device=logits_tensor.device)
    targets_tensor = tensor_batch["targets"].float()
    loss_tensor = nn.functional.binary_cross_entropy_with_logits(logits_tensor, targets_tensor)
    return {
        "logits": logits_tensor.detach().cpu().view(-1).tolist(),
        "probs": probs_tensor.detach().cpu().view(-1).tolist(),
        "targets": targets_tensor.detach().cpu().view(-1).int().tolist(),
        "aux_outputs": outputs.get("aux_outputs", {}),
        "debug_info": outputs.get("debug_info", {}),
        "loss_tensor": loss_tensor,
    }


def tensorize_batch(batch: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    device = torch.device((context or {}).get("device", "cpu"))
    student_id_map = (context or {}).get("student_id_map")
    tensor_batch = {
        "history_question_indices": torch.tensor(batch["history_question_indices"], dtype=torch.long, device=device),
        "history_correctness": torch.tensor(batch["history_correctness"], dtype=torch.long, device=device),
        "history_elapsed_times": torch.tensor(batch["history_elapsed_times"], dtype=torch.float32, device=device),
        "history_attempt_counts": torch.tensor(batch["history_attempt_counts"], dtype=torch.float32, device=device),
        "history_masks": torch.tensor(batch["history_masks"], dtype=torch.float32, device=device),
        "history_lengths": torch.tensor(batch["history_lengths"], dtype=torch.long, device=device),
        "target_question_index": torch.tensor(batch["target_question_index"], dtype=torch.long, device=device),
        "targets": torch.tensor(batch["targets"], dtype=torch.float32, device=device),
        "student_indices": torch.tensor(
            [student_id_map[student_id] for student_id in batch["student_ids"]] if student_id_map is not None else [0] * len(batch["student_ids"]),
            dtype=torch.long,
            device=device,
        ),
        "graph_data": context.get("graph_data") if context else None,
    }

    max_target_concepts = max(len(item) for item in batch["target_concept_indices"])
    padded_target_concepts: list[list[int]] = []
    target_masks: list[list[float]] = []
    for concept_indices in batch["target_concept_indices"]:
        pad = max_target_concepts - len(concept_indices)
        padded_target_concepts.append(list(concept_indices) + [0] * pad)
        target_masks.append([1.0] * len(concept_indices) + [0.0] * pad)
    tensor_batch["target_concept_indices"] = torch.tensor(padded_target_concepts, dtype=torch.long, device=device)
    tensor_batch["target_concept_mask"] = torch.tensor(target_masks, dtype=torch.float32, device=device)
    return tensor_batch
