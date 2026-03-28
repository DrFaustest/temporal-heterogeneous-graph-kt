"""Shared model interface for baseline models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """Minimal shared interface for baseline models."""

    @abstractmethod
    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Run a forward pass and return standardized outputs."""

    @abstractmethod
    def train_step(self, batch: dict[str, Any], learning_rate: float = 0.05) -> dict[str, Any]:
        """Run one optimization step and return metrics."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return serializable parameters."""
