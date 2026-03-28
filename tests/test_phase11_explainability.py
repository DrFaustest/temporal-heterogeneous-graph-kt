from __future__ import annotations

from pathlib import Path

import torch

from thgkt.data import CanonicalPreprocessor, PreprocessingConfig, SplitConfig, SyntheticToyAdapter, make_splits
from thgkt.explainability import ExplainabilityEngine
from thgkt.graph import build_hetero_graph, to_pyg_heterodata
from thgkt.models import THGKTConfig, THGKTModel
from thgkt.sequences import build_sequence_artifacts, collate_sequence_batch
from thgkt.training import Trainer, TrainingConfig


def test_trained_thgkt_produces_explainability_artifacts_and_plot(tmp_path) -> None:
    torch.manual_seed(31)
    bundle = CanonicalPreprocessor(PreprocessingConfig(min_interactions_per_student=4)).run(
        SyntheticToyAdapter().to_canonical()
    )
    split = make_splits(bundle, SplitConfig(strategy="student_chronological", random_seed=31))
    graph = build_hetero_graph(bundle, split)
    sequences = build_sequence_artifacts(bundle, split)
    concept_label_map = {index: concept_id for concept_id, index in sequences.concept_id_map.items()}
    context = {
        "device": "cpu",
        "graph_data": to_pyg_heterodata(graph, add_reverse_edges=True),
        "student_id_map": graph.node_maps["student"],
    }

    model = THGKTModel(
        THGKTConfig(num_students=4, num_questions=6, num_concepts=3, hidden_dim=16, temporal_hidden_dim=12)
    )
    trainer = Trainer()
    trainer.fit(
        model,
        sequences.train.examples,
        sequences.val.examples,
        TrainingConfig(
            run_name="phase11_thgkt_explainability",
            run_dir=str(tmp_path / "phase11_training_run"),
            epochs=1,
            batch_size=2,
            learning_rate=1e-2,
            random_seed=31,
        ),
        context=context,
    )

    batch = collate_sequence_batch(list(sequences.test.examples)[:2])
    engine = ExplainabilityEngine()
    concept_report = engine.concept_importance(model, batch, context=context, concept_label_map=concept_label_map)
    prereq_report = engine.prerequisite_influence(model, batch, context=context, concept_label_map=concept_label_map)
    artifacts = engine.export_concept_importance_artifacts(concept_report, tmp_path / "explainability")

    assert concept_report["method"] == "leave_one_target_concept_out"
    assert prereq_report["method"] == "drop_prerequisite_edges"
    assert len(concept_report["examples"]) == 2
    assert len(prereq_report["examples"]) == 2
    assert concept_report["examples"][0]["concept_scores"]
    assert Path(artifacts.report_path).exists()
    assert Path(artifacts.plot_path).exists()
    assert Path(artifacts.plot_path).read_text(encoding="utf-8").startswith("<svg")
