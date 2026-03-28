"""Config-driven experiment execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from thgkt.config import load_config
from thgkt.data.adapters.ednet import EdNetAdapterConfig
from thgkt.data import (
    AssistmentsAdapter,
    EdNetAdapter,
    CanonicalPreprocessor,
    PreprocessingConfig,
    RelationConfig,
    SyntheticToyAdapter,
    apply_concept_relation_mode,
    save_canonical_bundle,
    save_json,
    save_split_artifacts,
    summarize_bundle,
    make_splits,
)
from thgkt.explainability import ExplainabilityEngine
from thgkt.graph import GraphBuilderConfig, build_hetero_graph, save_graph_artifacts, to_pyg_heterodata
from thgkt.models import BaselineConfig, DKTBaseline, GraphOnlyModel, LogisticBaseline, SAKTBaseline, THGKTConfig, THGKTModel
from thgkt.reporting import save_ablation_bar_chart_svg, save_roc_curve_svg, save_training_curves_svg
from thgkt.sequences import SequenceBuilderConfig, build_sequence_artifacts, save_sequence_artifacts
from thgkt.training import Evaluator, Trainer, TrainingConfig, build_sequence_loader, load_checkpoint
from thgkt.training.runner import _forward_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "assistments_sample.csv"


ProgressCallback = Callable[[dict[str, Any]], None]


def _build_runtime_context_and_model(config: dict[str, Any], sequences, graph) -> tuple[Any, dict[str, Any]]:
    device = torch.device(str(config["training"].get("device", "cpu")))
    graph_data = to_pyg_heterodata(
        graph,
        add_reverse_edges=True,
        drop_prerequisite_edges=not bool(config["model"].get("use_prerequisite_edges", True)),
    )
    if hasattr(graph_data, "to"):
        graph_data = graph_data.to(device)

    context = {
        "device": str(device),
        "student_id_map": graph.node_maps["student"],
        "graph_data": graph_data,
    }
    model = _build_model(config, sequences, graph)
    if isinstance(model, nn.Module):
        model = model.to(device)
    return model, context


def run_experiment_from_config(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    run_name = str(config["run"]["name"])
    output_base = Path(output_root).resolve() if output_root is not None else (PROJECT_ROOT / "artifacts" / "runs").resolve()
    run_dir = output_base / run_name
    data_dir = run_dir / "data"
    plots_dir = run_dir / "plots"
    explain_dir = run_dir / "explainability"
    config_dir = run_dir / "config"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    experiment_config_path = save_json(config, config_dir / "experiment_config.json")

    _emit_progress(progress_callback, stage="run_started", run_name=run_name, config_path=str(config_path), run_dir=str(run_dir))
    _emit_progress(progress_callback, stage="bundle_loading_started", run_name=run_name)

    bundle = _load_and_prepare_bundle(config, progress_callback=progress_callback, run_name=run_name)
    summary = summarize_bundle(bundle)
    canonical_path = save_canonical_bundle(bundle, data_dir / "canonical_bundle.json")
    save_json(summary.to_dict(), data_dir / "canonical_summary.json")
    _emit_progress(
        progress_callback,
        stage="data_prepared",
        run_name=run_name,
        interactions=len(bundle.interactions.rows),
        questions=len(bundle.questions.rows),
        concepts=len(bundle.concepts.rows),
        canonical_path=str(canonical_path),
    )

    _emit_progress(progress_callback, stage="split_started", run_name=run_name)
    split = make_splits(bundle, _split_config_from_dict(config["split"]))
    split_path = save_split_artifacts(split, data_dir / "splits.json")
    _emit_progress(progress_callback, stage="split_ready", run_name=run_name, split_path=str(split_path))

    _emit_progress(progress_callback, stage="graph_building_started", run_name=run_name)
    graph = build_hetero_graph(bundle, split, _graph_config_from_dict(config["graph"]))
    graph_path = save_graph_artifacts(graph, data_dir / "graph.json")
    _emit_progress(progress_callback, stage="graph_ready", run_name=run_name, graph_path=str(graph_path))

    _emit_progress(progress_callback, stage="sequence_building_started", run_name=run_name)
    sequence_cfg = config.get("sequence", {})
    sequences = build_sequence_artifacts(
        bundle,
        split,
        SequenceBuilderConfig(
            allow_empty_history=bool(sequence_cfg.get("allow_empty_history", False)),
            max_history_length=(
                int(sequence_cfg["max_history_length"])
                if sequence_cfg.get("max_history_length") is not None
                else None
            ),
        ),
    )
    sequence_path = save_sequence_artifacts(sequences, data_dir / "sequences.json")
    _emit_progress(
        progress_callback,
        stage="sequences_ready",
        run_name=run_name,
        train_examples=len(sequences.train.examples),
        val_examples=len(sequences.val.examples),
        test_examples=len(sequences.test.examples),
        sequence_path=str(sequence_path),
    )

    model, context = _build_runtime_context_and_model(config, sequences, graph)
    device_name = str(context["device"])
    trainer = Trainer()
    training_config = TrainingConfig(
        run_name=run_name,
        run_dir=str(run_dir),
        epochs=int(config["training"]["epochs"]),
        batch_size=int(config["training"]["batch_size"]),
        learning_rate=float(config["training"]["learning_rate"]),
        device=device_name,
        random_seed=int(config["training"].get("random_seed", 42)),
        early_stopping_patience=(
            int(config["training"]["early_stopping_patience"])
            if config["training"].get("early_stopping_patience") is not None
            else None
        ),
    )
    _emit_progress(progress_callback, stage="training_configured", run_name=run_name, device=device_name, epochs=training_config.epochs)
    run_artifacts = trainer.fit(
        model,
        sequences.train.examples,
        sequences.val.examples,
        training_config,
        context=context,
        progress_callback=lambda event: _emit_progress(progress_callback, run_name=run_name, **event),
    )

    load_checkpoint(model, run_artifacts.checkpoint_path)

    test_loader = build_sequence_loader(
        sequences.test.examples,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        random_seed=int(config["training"].get("random_seed", 42)),
    )
    evaluator = Evaluator()
    _emit_progress(progress_callback, stage="evaluation_started", run_name=run_name)
    test_metrics = evaluator.evaluate(model, test_loader, context=context)
    predictions = collect_predictions(model, test_loader, context=context)
    save_json(test_metrics, run_dir / "metrics" / "test_metrics.json")
    save_json(predictions, run_dir / "metrics" / "test_predictions.json")
    _emit_progress(progress_callback, stage="evaluation_ready", run_name=run_name, test_metrics=test_metrics)

    _emit_progress(progress_callback, stage="plotting_started", run_name=run_name)
    metrics_payload = load_config(run_artifacts.metrics_path)
    training_curve_path = save_training_curves_svg(
        metrics_payload["train_history"],
        metrics_payload["val_history"],
        plots_dir / "training_curves.svg",
    )
    roc_path = save_roc_curve_svg(predictions["probs"], predictions["targets"], plots_dir / "roc_curve.svg")

    explainability_paths: dict[str, str] = {}
    explainability_enabled = bool(config.get("explainability", {}).get("enabled", True))
    if isinstance(model, THGKTModel) and sequences.test.examples and explainability_enabled:
        _emit_progress(progress_callback, stage="explainability_started", run_name=run_name)
        engine = ExplainabilityEngine()
        explain_batch = {
            **build_sequence_loader(
                sequences.test.examples,
                batch_size=min(2, len(sequences.test.examples)),
                shuffle=False,
                random_seed=0,
            )[0]
        }
        concept_label_map = {index: concept_id for concept_id, index in sequences.concept_id_map.items()}
        concept_report = engine.concept_importance(model, explain_batch, context=context, concept_label_map=concept_label_map)
        prereq_report = engine.prerequisite_influence(model, explain_batch, context=context, concept_label_map=concept_label_map)
        concept_artifacts = engine.export_concept_importance_artifacts(concept_report, explain_dir)
        save_json(prereq_report, explain_dir / "prerequisite_influence.json")
        explainability_paths = {
            "concept_report": concept_artifacts.report_path,
            "concept_plot": concept_artifacts.plot_path,
            "prerequisite_report": str(explain_dir / "prerequisite_influence.json"),
        }
        _emit_progress(progress_callback, stage="explainability_ready", run_name=run_name, explainability_paths=explainability_paths)

    run_summary = {
        "run_name": run_name,
        "config_path": str(Path(config_path)),
        "run_dir": str(run_dir),
        "artifacts": {
            "canonical_bundle": str(canonical_path),
            "splits": str(split_path),
            "graph": str(graph_path),
            "sequences": str(sequence_path),
            "checkpoint": run_artifacts.checkpoint_path,
            "metrics": run_artifacts.metrics_path,
            "test_metrics": str(run_dir / "metrics" / "test_metrics.json"),
            "test_predictions": str(run_dir / "metrics" / "test_predictions.json"),
            "training_plot": str(training_curve_path),
            "roc_plot": str(roc_path),
            "experiment_config": str(experiment_config_path),
            **explainability_paths,
        },
        "best_val_metrics": run_artifacts.best_val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(run_summary, run_dir / "run_summary.json")
    _emit_progress(progress_callback, stage="run_completed", run_name=run_name, run_summary=run_summary)
    return run_summary


def collect_predictions(model: Any, loader: list[dict[str, Any]], *, context: dict[str, Any] | None = None) -> dict[str, Any]:
    probs: list[float] = []
    targets: list[int] = []
    for batch in loader:
        outputs = _forward_model(model, batch, context=context, train_mode=False)
        probs.extend(float(item) for item in outputs["probs"])
        targets.extend(int(item) for item in outputs["targets"])
    return {"probs": probs, "targets": targets}


def make_ablation_plot(run_summaries: list[dict[str, Any]], path: str | Path, metric_name: str = "auc") -> Path:
    labels = [summary["run_name"] for summary in run_summaries]
    values = [float(summary["test_metrics"][metric_name]) for summary in run_summaries]
    return save_ablation_bar_chart_svg(labels, values, path, title="Ablation Comparison", metric_name=metric_name)


def _emit_progress(progress_callback: ProgressCallback | None, **payload: Any) -> None:
    if progress_callback is not None:
        progress_callback(dict(payload))


def _load_and_prepare_bundle(
    config: dict[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
    run_name: str | None = None,
):
    dataset_name = str(config["dataset"]["name"])
    if dataset_name == "synthetic_toy":
        bundle = SyntheticToyAdapter().to_canonical()
    elif dataset_name == "assistments_fixture":
        adapter = AssistmentsAdapter()
        bundle = adapter.to_canonical(adapter.load_raw(FIXTURE_PATH))
    elif dataset_name == "assistments_path":
        adapter = AssistmentsAdapter()
        raw_path = Path(config["dataset"]["path"])
        bundle = adapter.to_canonical(adapter.load_raw(raw_path))
    elif dataset_name == "ednet_fixture":
        adapter = EdNetAdapter()
        raw_path = PROJECT_ROOT / "tests" / "fixtures" / "ednet_small" / "KT1"
        contents_path = PROJECT_ROOT / "tests" / "fixtures" / "ednet_small" / "contents"
        bundle = adapter.to_canonical(adapter.load_raw(raw_path, contents_path=contents_path))
    elif dataset_name == "ednet_path":
        dataset_cfg = config["dataset"]
        adapter = EdNetAdapter(
            config=EdNetAdapterConfig(max_users=int(dataset_cfg.get("max_users", 0)))
        )
        raw_path = Path(dataset_cfg["path"])
        contents_path = Path(dataset_cfg.get("contents_path", raw_path))
        bundle = adapter.to_canonical(adapter.load_raw(raw_path, contents_path=contents_path))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    _emit_progress(progress_callback, stage="bundle_loading_completed", run_name=run_name or dataset_name, dataset_name=dataset_name)
    relation_mode = str(config["dataset"].get("relation_mode", "none"))
    _emit_progress(progress_callback, stage="relation_building_started", run_name=run_name or dataset_name, relation_mode=relation_mode)
    bundle = apply_concept_relation_mode(bundle, RelationConfig(mode=relation_mode))
    _emit_progress(progress_callback, stage="preprocessing_started", run_name=run_name or dataset_name)
    preprocessor = CanonicalPreprocessor(
        PreprocessingConfig(
            min_interactions_per_student=int(config["preprocessing"].get("min_interactions_per_student", 3)),
            drop_interactions_without_concepts=bool(config["preprocessing"].get("drop_interactions_without_concepts", True)),
        )
    )
    return preprocessor.run(bundle)


def _split_config_from_dict(split_config: dict[str, Any]):
    from thgkt.data.splitting import SplitConfig

    return SplitConfig(
        strategy=str(split_config["strategy"]),
        train_ratio=float(split_config.get("train_ratio", 0.7)),
        val_ratio=float(split_config.get("val_ratio", 0.15)),
        test_ratio=float(split_config.get("test_ratio", 0.15)),
        random_seed=int(split_config.get("random_seed", 42)),
    )


def _graph_config_from_dict(graph_config: dict[str, Any]):
    return GraphBuilderConfig(
        interaction_edge_split=str(graph_config.get("interaction_edge_split", "train")),
        include_student_exposure_edges=bool(graph_config.get("include_student_exposure_edges", True)),
        include_reverse_edges=bool(graph_config.get("include_reverse_edges", False)),
    )


def _build_model(config: dict[str, Any], sequences, graph):
    model_type = str(config["model"]["type"])
    baseline_config = BaselineConfig(
        random_seed=int(config["training"].get("random_seed", 42)),
        hidden_dim=int(config["model"].get("hidden_dim", 64)),
        question_embedding_dim=int(config["model"].get("question_embedding_dim", 32)),
        correctness_embedding_dim=int(config["model"].get("correctness_embedding_dim", 8)),
        dropout=float(config["model"].get("dropout", 0.1)),
        gradient_clip_norm=float(config["model"].get("gradient_clip_norm", 5.0)),
        elapsed_time_scale=float(config["model"].get("elapsed_time_scale", 5000.0)),
        attempt_count_scale=float(config["model"].get("attempt_count_scale", 5.0)),
        attention_heads=int(config["model"].get("attention_heads", 4)),
    )
    if model_type == "logistic_baseline":
        return LogisticBaseline(len(sequences.question_id_map), len(sequences.concept_id_map), config=baseline_config)
    if model_type == "dkt_baseline":
        return DKTBaseline(len(sequences.question_id_map), config=baseline_config)
    if model_type == "graph_only":
        return GraphOnlyModel(graph, config=baseline_config)
    if model_type == "sakt_baseline":
        return SAKTBaseline(len(sequences.question_id_map), config=baseline_config)
    if model_type == "thgkt":
        return THGKTModel(
            THGKTConfig(
                num_students=len(graph.node_maps["student"]),
                num_questions=len(sequences.question_id_map),
                num_concepts=len(sequences.concept_id_map),
                hidden_dim=int(config["model"].get("hidden_dim", 32)),
                temporal_hidden_dim=int(config["model"].get("temporal_hidden_dim", 32)),
                graph_num_layers=int(config["model"].get("graph_num_layers", 2)),
                use_graph_encoder=bool(config["model"].get("use_graph_encoder", True)),
                use_temporal_encoder=bool(config["model"].get("use_temporal_encoder", True)),
                use_time_features=bool(config["model"].get("use_time_features", True)),
                use_prerequisite_edges=bool(config["model"].get("use_prerequisite_edges", True)),
                use_target_concept_attention=bool(config["model"].get("use_target_concept_attention", False)),
                elapsed_time_scale=float(config["model"].get("elapsed_time_scale", 5000.0)),
                attempt_count_scale=float(config["model"].get("attempt_count_scale", 5.0)),
                dropout=float(config["model"].get("dropout", 0.1)),
            )
        )
    raise ValueError(f"Unsupported model type: {model_type}")
