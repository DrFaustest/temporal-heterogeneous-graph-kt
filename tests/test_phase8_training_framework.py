from __future__ import annotations

import json
from pathlib import Path

from thgkt.data import CanonicalPreprocessor, PreprocessingConfig, SplitConfig, SyntheticToyAdapter, make_splits
from thgkt.models import LogisticBaseline
from thgkt.sequences import build_sequence_artifacts
from thgkt.training import Evaluator, Trainer, TrainingConfig, build_sequence_loader, load_checkpoint


def test_training_framework_runs_end_to_end_and_reloads_checkpoint(tmp_path) -> None:
    bundle = CanonicalPreprocessor(PreprocessingConfig(min_interactions_per_student=4)).run(
        SyntheticToyAdapter().to_canonical()
    )
    split = make_splits(bundle, SplitConfig(strategy="student_chronological", random_seed=13))
    sequences = build_sequence_artifacts(bundle, split)

    model = LogisticBaseline(len(sequences.question_id_map), len(sequences.concept_id_map))
    trainer = Trainer()
    config = TrainingConfig(
        run_name="baseline_phase8",
        run_dir=str(tmp_path / "baseline_run"),
        epochs=2,
        batch_size=2,
        learning_rate=0.1,
        random_seed=13,
    )
    artifacts = trainer.fit(model, sequences.train.examples, sequences.val.examples, config)

    assert Path(artifacts.checkpoint_path).exists()
    assert Path(artifacts.metrics_path).exists()
    assert Path(artifacts.config_snapshot_path).exists()
    assert set(artifacts.best_val_metrics) >= {"auc", "accuracy", "f1", "bce_loss"}

    clone = LogisticBaseline(len(sequences.question_id_map), len(sequences.concept_id_map))
    load_checkpoint(clone, artifacts.checkpoint_path)
    evaluator = Evaluator()
    val_loader = build_sequence_loader(sequences.val.examples, batch_size=2, shuffle=False, random_seed=13)
    reloaded_metrics = evaluator.evaluate(clone, val_loader)

    assert reloaded_metrics == artifacts.best_val_metrics
    saved_metrics = json.loads(Path(artifacts.metrics_path).read_text(encoding="utf-8"))
    assert "train_history" in saved_metrics and "val_history" in saved_metrics



class StubEvaluator:
    def __init__(self, auc_values: list[float]) -> None:
        self._auc_values = list(auc_values)
        self._index = 0

    def evaluate(self, model, loader, context=None) -> dict[str, float]:
        auc = self._auc_values[self._index]
        self._index += 1
        return {"auc": auc, "accuracy": 0.5, "f1": 0.5, "bce_loss": 0.7}


class DeterministicTrainer(Trainer):
    def train_epoch(self, model, loader, *, optimizer=None, context=None) -> dict[str, float]:
        return {"auc": 0.5, "accuracy": 0.5, "f1": 0.5, "bce_loss": 0.7, "loss": 0.7}


def test_training_stops_early_when_val_auc_plateaus(tmp_path) -> None:
    bundle = CanonicalPreprocessor(PreprocessingConfig(min_interactions_per_student=4)).run(
        SyntheticToyAdapter().to_canonical()
    )
    split = make_splits(bundle, SplitConfig(strategy="student_chronological", random_seed=13))
    sequences = build_sequence_artifacts(bundle, split)

    model = LogisticBaseline(len(sequences.question_id_map), len(sequences.concept_id_map))
    trainer = DeterministicTrainer(evaluator=StubEvaluator([0.8, 0.7, 0.6, 0.5]))
    config = TrainingConfig(
        run_name="baseline_early_stop",
        run_dir=str(tmp_path / "baseline_early_stop"),
        epochs=20,
        batch_size=2,
        learning_rate=0.1,
        random_seed=13,
        early_stopping_patience=2,
    )
    artifacts = trainer.fit(model, sequences.train.examples, sequences.val.examples, config)

    assert len(artifacts.train_history) == 3
    assert len(artifacts.val_history) == 3
    assert artifacts.best_val_metrics["auc"] == 0.8
    assert artifacts.extra["best_epoch"] == 1
    assert artifacts.extra["completed_epochs"] == 3
    assert artifacts.extra["stopped_early"] is True

    saved_metrics = json.loads(Path(artifacts.metrics_path).read_text(encoding="utf-8"))
    assert saved_metrics["best_epoch"] == 1
    assert saved_metrics["completed_epochs"] == 3
    assert saved_metrics["stopped_early"] is True
