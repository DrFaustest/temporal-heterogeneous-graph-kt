"""Training artifact types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class TrainingRunArtifacts:
    run_name: str
    run_dir: str
    model_name: str
    train_history: tuple[dict[str, float], ...]
    val_history: tuple[dict[str, float], ...]
    best_val_metrics: dict[str, float]
    checkpoint_path: str
    metrics_path: str
    config_snapshot_path: str
    created_at: str = field(default_factory=utc_now_iso)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "run_dir": self.run_dir,
            "model_name": self.model_name,
            "train_history": [dict(item) for item in self.train_history],
            "val_history": [dict(item) for item in self.val_history],
            "best_val_metrics": dict(self.best_val_metrics),
            "checkpoint_path": self.checkpoint_path,
            "metrics_path": self.metrics_path,
            "config_snapshot_path": self.config_snapshot_path,
            "created_at": self.created_at,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainingRunArtifacts":
        return cls(
            run_name=str(payload["run_name"]),
            run_dir=str(payload["run_dir"]),
            model_name=str(payload["model_name"]),
            train_history=tuple(dict(item) for item in payload["train_history"]),
            val_history=tuple(dict(item) for item in payload["val_history"]),
            best_val_metrics=dict(payload["best_val_metrics"]),
            checkpoint_path=str(payload["checkpoint_path"]),
            metrics_path=str(payload["metrics_path"]),
            config_snapshot_path=str(payload["config_snapshot_path"]),
            created_at=str(payload.get("created_at", utc_now_iso())),
            extra=dict(payload.get("extra", {})),
        )
