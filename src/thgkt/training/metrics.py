"""Metric computation utilities."""

from __future__ import annotations

import math
from typing import Iterable


def compute_classification_metrics(
    probs: Iterable[float],
    targets: Iterable[int],
    threshold: float = 0.5,
) -> dict[str, float]:
    prob_list = [float(item) for item in probs]
    target_list = [int(item) for item in targets]
    if len(prob_list) != len(target_list):
        raise ValueError("Probability and target lengths must match.")
    if not prob_list:
        raise ValueError("Cannot compute metrics for an empty prediction set.")

    predictions = [1 if prob >= threshold else 0 for prob in prob_list]
    correct = sum(int(pred == target) for pred, target in zip(predictions, target_list))
    tp = sum(1 for pred, target in zip(predictions, target_list) if pred == 1 and target == 1)
    fp = sum(1 for pred, target in zip(predictions, target_list) if pred == 1 and target == 0)
    fn = sum(1 for pred, target in zip(predictions, target_list) if pred == 0 and target == 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    bce = -sum(
        target * math.log(min(max(prob, 1e-7), 1 - 1e-7))
        + (1 - target) * math.log(min(max(1 - prob, 1e-7), 1 - 1e-7))
        for prob, target in zip(prob_list, target_list)
    ) / len(prob_list)

    return {
        "auc": _binary_auc(prob_list, target_list),
        "accuracy": correct / len(prob_list),
        "f1": f1,
        "bce_loss": bce,
    }


def _binary_auc(probs: list[float], targets: list[int]) -> float:
    positives = sum(targets)
    negatives = len(targets) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    ranked = sorted(enumerate(probs), key=lambda item: item[1])
    rank_sum = 0.0
    for rank, (index, _) in enumerate(ranked, start=1):
        if targets[index] == 1:
            rank_sum += rank
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
