"""Small math helpers for fallback baseline implementations."""

from __future__ import annotations

import math
import random
from typing import Iterable


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def bce_loss(probability: float, target: int) -> float:
    clipped = min(max(probability, 1e-7), 1.0 - 1e-7)
    return -(target * math.log(clipped) + (1 - target) * math.log(1.0 - clipped))


def dot(lhs: Iterable[float], rhs: Iterable[float]) -> float:
    return sum(left * right for left, right in zip(lhs, rhs))


def zeros(size: int) -> list[float]:
    return [0.0 for _ in range(size)]


def zeros_matrix(rows: int, cols: int) -> list[list[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def random_vector(size: int, rng: random.Random, scale: float = 0.1) -> list[float]:
    return [rng.uniform(-scale, scale) for _ in range(size)]


def random_matrix(rows: int, cols: int, rng: random.Random, scale: float = 0.1) -> list[list[float]]:
    return [random_vector(cols, rng, scale=scale) for _ in range(rows)]


def tanh_vector(values: list[float]) -> list[float]:
    return [math.tanh(value) for value in values]
