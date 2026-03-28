from __future__ import annotations

from pathlib import Path

import torch

from thgkt.data import (
    AssistmentsAdapter,
    CanonicalPreprocessor,
    PreprocessingConfig,
    SplitConfig,
    SyntheticToyAdapter,
    make_splits,
)
from thgkt.graph import build_hetero_graph, to_pyg_heterodata
from thgkt.models import THGKTConfig, THGKTModel
from thgkt.sequences import build_sequence_artifacts, collate_sequence_batch
from thgkt.training import Evaluator, Trainer, TrainingConfig, build_sequence_loader, load_checkpoint
from thgkt.training.runner import tensorize_batch


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "assistments_sample.csv"


def _toy_context_and_sequences():
    bundle = CanonicalPreprocessor(PreprocessingConfig(min_interactions_per_student=4)).run(
        SyntheticToyAdapter().to_canonical()
    )
    split = make_splits(bundle, SplitConfig(strategy="student_chronological", random_seed=17))
    graph = build_hetero_graph(bundle, split)
    sequences = build_sequence_artifacts(bundle, split)
    context = {
        "device": "cpu",
        "graph_data": to_pyg_heterodata(graph, add_reverse_edges=True),
        "student_id_map": graph.node_maps["student"],
    }
    return graph, sequences, context


def _assistments_context_and_sequences():
    bundle = CanonicalPreprocessor(PreprocessingConfig(min_interactions_per_student=4)).run(
        AssistmentsAdapter().to_canonical(AssistmentsAdapter().load_raw(FIXTURE_PATH))
    )
    split = make_splits(bundle, SplitConfig(strategy="student_chronological", random_seed=19))
    graph = build_hetero_graph(bundle, split)
    sequences = build_sequence_artifacts(bundle, split)
    context = {
        "device": "cpu",
        "graph_data": to_pyg_heterodata(graph, add_reverse_edges=True),
        "student_id_map": graph.node_maps["student"],
    }
    return graph, sequences, context


def test_thgkt_forward_and_ablation_flags_work() -> None:
    torch.manual_seed(7)
    graph, sequences, context = _toy_context_and_sequences()
    batch = collate_sequence_batch(list(sequences.train.examples)[:2])

    configs = [
        THGKTConfig(num_students=4, num_questions=6, num_concepts=3, hidden_dim=16, temporal_hidden_dim=12),
        THGKTConfig(num_students=4, num_questions=6, num_concepts=3, hidden_dim=16, temporal_hidden_dim=12, use_graph_encoder=False),
        THGKTConfig(num_students=4, num_questions=6, num_concepts=3, hidden_dim=16, temporal_hidden_dim=12, use_temporal_encoder=False),
        THGKTConfig(num_students=4, num_questions=6, num_concepts=3, hidden_dim=16, temporal_hidden_dim=12, use_prerequisite_edges=False, use_time_features=False),
    ]

    for config in configs:
        model = THGKTModel(config)
        outputs = model(tensorize_batch(batch, context=context))
        assert outputs["logits"].shape[0] == len(batch["targets"])
        assert outputs["probs"].shape[0] == len(batch["targets"])
        assert torch.allclose(outputs["probs"], torch.sigmoid(outputs["logits"]))
        assert outputs["debug_info"]["use_graph_encoder"] == config.use_graph_encoder
        assert outputs["debug_info"]["graph_num_layers"] == 2


def test_thgkt_trains_on_toy_and_reloads_checkpoint(tmp_path) -> None:
    torch.manual_seed(11)
    graph, sequences, context = _toy_context_and_sequences()
    model = THGKTModel(
        THGKTConfig(num_students=4, num_questions=6, num_concepts=3, hidden_dim=16, temporal_hidden_dim=12)
    )
    trainer = Trainer()
    config = TrainingConfig(
        run_name="thgkt_toy",
        run_dir=str(tmp_path / "thgkt_toy_run"),
        epochs=2,
        batch_size=2,
        learning_rate=1e-2,
        random_seed=11,
    )
    artifacts = trainer.fit(model, sequences.train.examples, sequences.val.examples, config, context=context)

    assert Path(artifacts.checkpoint_path).exists()
    clone = THGKTModel(
        THGKTConfig(num_students=4, num_questions=6, num_concepts=3, hidden_dim=16, temporal_hidden_dim=12)
    )
    load_checkpoint(clone, artifacts.checkpoint_path)
    evaluator = Evaluator()
    val_loader = build_sequence_loader(sequences.val.examples, batch_size=2, shuffle=False, random_seed=11)
    metrics = evaluator.evaluate(clone, val_loader, context=context)
    assert set(metrics) == {"auc", "accuracy", "f1", "bce_loss"}


def test_thgkt_trains_on_small_assistments_subset(tmp_path) -> None:
    torch.manual_seed(23)
    graph, sequences, context = _assistments_context_and_sequences()
    model = THGKTModel(
        THGKTConfig(num_students=4, num_questions=4, num_concepts=4, hidden_dim=16, temporal_hidden_dim=12)
    )
    trainer = Trainer()
    config = TrainingConfig(
        run_name="thgkt_assistments",
        run_dir=str(tmp_path / "thgkt_assistments_run"),
        epochs=1,
        batch_size=2,
        learning_rate=1e-2,
        random_seed=23,
    )
    artifacts = trainer.fit(model, sequences.train.examples, sequences.val.examples, config, context=context)

    assert Path(artifacts.metrics_path).exists()
    assert artifacts.best_val_metrics["bce_loss"] >= 0.0



def test_thgkt_temporal_encoder_uses_last_valid_state() -> None:
    torch.manual_seed(29)
    _, sequences, context = _toy_context_and_sequences()
    model = THGKTModel(
        THGKTConfig(num_students=4, num_questions=6, num_concepts=3, hidden_dim=16, temporal_hidden_dim=12)
    )
    batch = collate_sequence_batch(list(sequences.train.examples)[:2])
    batch["history_lengths"][0] = 0
    batch["history_masks"][0] = [0.0 for _ in batch["history_masks"][0]]
    batch["history_question_indices"][0] = [-1 for _ in batch["history_question_indices"][0]]
    batch["history_elapsed_times"][0] = [0.0 for _ in batch["history_elapsed_times"][0]]
    batch["history_attempt_counts"][0] = [0 for _ in batch["history_attempt_counts"][0]]

    encoded = model._encode_temporal(tensorize_batch(batch, context=context))

    assert torch.allclose(encoded[0], torch.zeros_like(encoded[0]))
    assert torch.isfinite(encoded[1]).all()


def test_thgkt_target_concept_attention_pooling_is_optional() -> None:
    torch.manual_seed(37)
    _, sequences, context = _toy_context_and_sequences()
    batch = collate_sequence_batch(list(sequences.train.examples)[:2])
    tensor_batch = tensorize_batch(batch, context=context)
    model = THGKTModel(
        THGKTConfig(
            num_students=4,
            num_questions=6,
            num_concepts=3,
            hidden_dim=16,
            temporal_hidden_dim=12,
            use_target_concept_attention=True,
        )
    )

    outputs = model(tensor_batch)

    assert outputs["debug_info"]["use_target_concept_attention"] is True
    assert outputs["aux_outputs"]["target_concept_repr"].shape == (len(batch["targets"]), 16)



def test_thgkt_no_prereq_checkpoint_reloads_with_lazy_relation_weights(tmp_path) -> None:
    torch.manual_seed(41)
    _, sequences, context = _toy_context_and_sequences()
    model = THGKTModel(
        THGKTConfig(
            num_students=4,
            num_questions=6,
            num_concepts=3,
            hidden_dim=16,
            temporal_hidden_dim=12,
            use_prerequisite_edges=False,
        )
    )
    trainer = Trainer()
    config = TrainingConfig(
        run_name="thgkt_no_prereq_reload",
        run_dir=str(tmp_path / "thgkt_no_prereq_reload"),
        epochs=1,
        batch_size=2,
        learning_rate=1e-2,
        random_seed=41,
    )
    artifacts = trainer.fit(model, sequences.train.examples, sequences.val.examples, config, context=context)

    clone = THGKTModel(
        THGKTConfig(
            num_students=4,
            num_questions=6,
            num_concepts=3,
            hidden_dim=16,
            temporal_hidden_dim=12,
            use_prerequisite_edges=False,
        )
    )
    load_checkpoint(clone, artifacts.checkpoint_path)
    val_loader = build_sequence_loader(sequences.val.examples, batch_size=2, shuffle=False, random_seed=41)
    metrics = Evaluator().evaluate(clone, val_loader, context=context)

    assert set(metrics) == {"auc", "accuracy", "f1", "bce_loss"}
