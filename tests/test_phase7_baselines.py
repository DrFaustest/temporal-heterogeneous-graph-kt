from __future__ import annotations

import copy
from pathlib import Path

import torch

from thgkt.data.adapters.synthetic import SyntheticToyAdapter
from thgkt.data.preprocessing import CanonicalPreprocessor, PreprocessingConfig
from thgkt.data.splitting import SplitConfig, make_splits
from thgkt.graph import build_hetero_graph
from thgkt.models import DKTBaseline, GraphOnlyModel, LogisticBaseline, SAKTBaseline
from thgkt.sequences import build_sequence_artifacts, collate_sequence_batch
from thgkt.training import Evaluator, Trainer, TrainingConfig, build_sequence_loader, load_checkpoint


def _toy_graph_and_batch():
    bundle = CanonicalPreprocessor(PreprocessingConfig(min_interactions_per_student=4)).run(
        SyntheticToyAdapter().to_canonical()
    )
    split = make_splits(bundle, SplitConfig(strategy="student_chronological", random_seed=9))
    graph = build_hetero_graph(bundle, split)
    sequences = build_sequence_artifacts(bundle, split)
    batch = collate_sequence_batch(list(sequences.train.examples))
    return graph, sequences, batch


def _state_changed(before, after) -> bool:
    if isinstance(before, torch.Tensor):
        return not torch.equal(before, after)
    if isinstance(before, dict):
        return any(_state_changed(before[key], after[key]) for key in before)
    if isinstance(before, (list, tuple)):
        return any(_state_changed(left, right) for left, right in zip(before, after))
    return before != after


def test_all_baselines_share_interface_and_forward() -> None:
    graph, sequences, batch = _toy_graph_and_batch()
    models = [
        LogisticBaseline(len(sequences.question_id_map), len(sequences.concept_id_map)),
        DKTBaseline(len(sequences.question_id_map)),
        SAKTBaseline(len(sequences.question_id_map)),
        GraphOnlyModel(graph),
    ]

    for model in models:
        outputs = model.forward(batch)
        assert set(outputs.keys()) == {"logits", "probs", "targets", "aux_outputs", "debug_info"}
        assert len(outputs["logits"]) == len(batch["targets"])
        assert len(outputs["probs"]) == len(batch["targets"])
        assert outputs["targets"] == batch["targets"]
        assert all(0.0 < float(probability) < 1.0 for probability in outputs["probs"])


def test_all_baselines_complete_one_train_step() -> None:
    graph, sequences, batch = _toy_graph_and_batch()
    models = [
        LogisticBaseline(len(sequences.question_id_map), len(sequences.concept_id_map)),
        DKTBaseline(len(sequences.question_id_map)),
        SAKTBaseline(len(sequences.question_id_map)),
        GraphOnlyModel(graph),
    ]

    for model in models:
        before = copy.deepcopy(model.state_dict())
        result = model.train_step(batch, learning_rate=0.1)
        after = model.state_dict()
        assert result["loss"] > 0.0
        assert _state_changed(before, after)
        assert len(result["outputs"]["logits"]) == len(batch["targets"])


def test_sequence_torch_baselines_train_and_reload_with_shared_trainer(tmp_path) -> None:
    _, sequences, _ = _toy_graph_and_batch()
    constructors = {
        "dkt_baseline_toy": DKTBaseline,
        "sakt_baseline_toy": SAKTBaseline,
    }

    for run_name, constructor in constructors.items():
        model = constructor(len(sequences.question_id_map))
        trainer = Trainer()
        config = TrainingConfig(
            run_name=run_name,
            run_dir=str(tmp_path / run_name),
            epochs=2,
            batch_size=2,
            learning_rate=1e-2,
            random_seed=13,
        )
        artifacts = trainer.fit(model, sequences.train.examples, sequences.val.examples, config)

        assert Path(artifacts.checkpoint_path).exists()
        clone = constructor(len(sequences.question_id_map))
        load_checkpoint(clone, artifacts.checkpoint_path)
        evaluator = Evaluator()
        val_loader = build_sequence_loader(sequences.val.examples, batch_size=2, shuffle=False, random_seed=13)
        metrics = evaluator.evaluate(clone, val_loader)

        assert set(metrics) == {"auc", "accuracy", "f1", "bce_loss"}
        assert artifacts.best_val_metrics["bce_loss"] >= 0.0
