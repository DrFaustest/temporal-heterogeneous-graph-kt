from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thgkt.config import load_config
from thgkt.experiment import make_ablation_plot, run_experiment_from_config
from thgkt.reporting import save_ablation_bar_chart_svg


DEFAULT_CONFIGS = [
    ROOT / "configs" / "phase12" / "baseline_assistments.yaml",
    ROOT / "configs" / "phase12" / "thgkt_assistments.yaml",
    ROOT / "configs" / "phase12" / "thgkt_ablation_assistments.yaml",
]


class ProgressReporter:
    def __init__(self, total_units: int) -> None:
        self.total_units = max(1, total_units)
        self.completed_units = 0
        self.started_at = time.perf_counter()
        self.last_message = "initialized"
        self.last_output_at = self.started_at
        self._lock = threading.Lock()

    def status(self, message: str) -> None:
        with self._lock:
            self.last_message = message
            self._print_locked(message, self.completed_units)

    def complete_to(self, completed_units: int, message: str) -> None:
        with self._lock:
            self.completed_units = max(self.completed_units, min(completed_units, self.total_units))
            self.last_message = message
            self._print_locked(message, self.completed_units)

    def heartbeat(self) -> None:
        with self._lock:
            self._print_locked(f"{self.last_message} | still running", self.completed_units)

    def _print_locked(self, message: str, completed_units: int) -> None:
        elapsed = time.perf_counter() - self.started_at
        percent = 100.0 * completed_units / self.total_units
        eta = None
        if completed_units > 0:
            eta = elapsed * (self.total_units - completed_units) / completed_units
        eta_text = _format_duration(eta) if eta is not None else "unknown"
        self.last_output_at = time.perf_counter()
        print(
            f"[{percent:6.2f}%] {message} | elapsed {_format_duration(elapsed)} | ETA {eta_text}",
            flush=True,
        )


def _heartbeat_worker(reporter: ProgressReporter, stop_event: threading.Event, interval_seconds: int) -> None:
    while not stop_event.wait(interval_seconds):
        reporter.heartbeat()


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _detect_device(requested: str) -> dict[str, Any]:
    info: dict[str, Any] = {"requested": requested, "selected": "cpu", "cuda_available": False}
    try:
        import torch
    except ModuleNotFoundError:
        info["reason"] = "torch is not installed; falling back to cpu"
        return info

    cuda_available = bool(torch.cuda.is_available())
    info["cuda_available"] = cuda_available
    if requested == "auto":
        info["selected"] = "cuda" if cuda_available else "cpu"
    elif requested.startswith("cuda"):
        info["selected"] = requested if cuda_available else "cpu"
        if not cuda_available:
            info["reason"] = f"requested {requested} but CUDA is unavailable; falling back to cpu"
    else:
        info["selected"] = requested

    if info["selected"].startswith("cuda") and cuda_available:
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def _requires_cpu_fallback_confirmation(device_info: dict[str, Any]) -> bool:
    requested = str(device_info.get("requested", ""))
    selected = str(device_info.get("selected", ""))
    return requested.startswith("cuda") and selected == "cpu"


def _confirm_cpu_fallback(device_info: dict[str, Any]) -> None:
    warning = (
        f"WARNING: requested GPU device '{device_info.get('requested', 'cuda')}', "
        "but the run resolved to CPU."
    )
    reason = device_info.get("reason")
    if reason:
        warning = f"{warning} {reason}"
    print(warning, file=sys.stderr, flush=True)

    try:
        response = input("Continue on CPU instead? [y/N]: ").strip().lower()
    except EOFError:
        raise SystemExit("Aborted: GPU was requested, CPU fallback was not confirmed.") from None

    if response not in {"y", "yes"}:
        raise SystemExit("Aborted: GPU was requested and CPU fallback was declined.")


def _experiment_total_units(config: dict[str, Any]) -> int:
    epochs = int(config["training"].get("epochs", 1))
    has_explainability = bool(config.get("explainability", {}).get("enabled", True)) and str(config["model"]["type"]) == "thgkt"
    return 5 + epochs + (1 if has_explainability else 0)


def _event_completed_units(event: dict[str, Any], *, epochs: int, has_explainability: bool) -> int:
    stage = str(event.get("stage", ""))
    if stage == "data_prepared":
        return 1
    if stage == "split_ready":
        return 2
    if stage == "graph_ready":
        return 3
    if stage == "sequences_ready":
        return 4
    if stage == "training_epoch_completed":
        return 4 + int(event.get("current_epoch", 0))
    if stage in {"training_completed", "evaluation_ready"}:
        return 4 + epochs + (1 if stage == "evaluation_ready" else 0)
    if stage == "explainability_ready":
        return 5 + epochs + (1 if has_explainability else 0)
    if stage == "run_completed":
        return 5 + epochs + (1 if has_explainability else 0)
    return 0


def _event_message(run_name: str, event: dict[str, Any]) -> str:
    stage = str(event.get("stage", "status"))
    if stage == "run_started":
        return f"{run_name}: initializing run directories"
    if stage == "bundle_loading_started":
        return f"{run_name}: loading raw dataset adapter inputs"
    if stage == "bundle_loading_completed":
        return f"{run_name}: raw dataset loaded into canonical adapter"
    if stage == "relation_building_started":
        return f"{run_name}: constructing concept relations"
    if stage == "preprocessing_started":
        return f"{run_name}: preprocessing canonical bundle"
    if stage == "data_prepared":
        return (
            f"{run_name}: canonical bundle ready "
            f"({event.get('interactions', 0)} interactions, {event.get('questions', 0)} questions, {event.get('concepts', 0)} concepts)"
        )
    if stage == "split_started":
        return f"{run_name}: building leakage-safe temporal splits"
    if stage == "split_ready":
        return f"{run_name}: temporal splits built"
    if stage == "graph_building_started":
        return f"{run_name}: building heterogeneous graph"
    if stage == "graph_ready":
        return f"{run_name}: heterogeneous graph built"
    if stage == "sequence_building_started":
        return f"{run_name}: constructing next-response sequences"
    if stage == "sequences_ready":
        return (
            f"{run_name}: sequences ready "
            f"(train={event.get('train_examples', 0)}, val={event.get('val_examples', 0)}, test={event.get('test_examples', 0)})"
        )
    if stage == "training_configured":
        return f"{run_name}: training on {event.get('device', 'cpu')} for {event.get('epochs', 0)} epochs"
    if stage == "training_started":
        return f"{run_name}: training started"
    if stage == "training_epoch_completed":
        val_metrics = event.get("val_metrics", {})
        return (
            f"{run_name}: epoch {event.get('current_epoch', 0)}/{event.get('total_epochs', 0)} complete "
            f"(val_auc={val_metrics.get('auc', 0.0):.4f}, val_loss={val_metrics.get('loss', 0.0):.4f})"
        )
    if stage == "training_completed":
        return f"{run_name}: training complete"
    if stage == "evaluation_started":
        return f"{run_name}: evaluating on held-out data"
    if stage == "evaluation_ready":
        test_metrics = event.get("test_metrics", {})
        return f"{run_name}: evaluation complete (test_auc={test_metrics.get('auc', 0.0):.4f})"
    if stage == "plotting_started":
        return f"{run_name}: generating plots and summaries"
    if stage == "explainability_started":
        return f"{run_name}: generating explainability artifacts"
    if stage == "explainability_ready":
        return f"{run_name}: explainability artifacts saved"
    if stage == "run_completed":
        return f"{run_name}: run completed"
    return f"{run_name}: {stage}"


def _write_resolved_config(source_path: Path, output_root: Path, *, device: str) -> tuple[Path, dict[str, Any]]:
    config = load_config(source_path)
    config.setdefault("training", {})["device"] = device
    resolved_dir = output_root / "resolved_configs"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = resolved_dir / f"{source_path.stem}_{device}.json"
    resolved_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return resolved_path, config


def _run_pytest() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "command": f"{sys.executable} -m pytest -q",
        "stdout": completed.stdout.strip().splitlines(),
        "stderr": completed.stderr.strip().splitlines() if completed.stderr.strip() else [],
    }


def _resolve_source_configs(config_args: list[str] | None) -> list[Path]:
    if not config_args:
        return list(DEFAULT_CONFIGS)
    return [Path(config_arg) if Path(config_arg).is_absolute() else ROOT / config_arg for config_arg in config_args]


def _build_comparison_rows(run_summaries: list[dict[str, Any]], configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary, config in zip(run_summaries, configs):
        model_cfg = config.get("model", {})
        dataset_cfg = config.get("dataset", {})
        metrics = summary["test_metrics"]
        rows.append(
            {
                "run_name": summary["run_name"],
                "model_type": str(model_cfg.get("type", "unknown")),
                "relation_mode": str(dataset_cfg.get("relation_mode", "none")),
                "max_users": int(dataset_cfg.get("max_users", 0)) if dataset_cfg.get("max_users") is not None else 0,
                "use_graph_encoder": model_cfg.get("use_graph_encoder"),
                "use_temporal_encoder": model_cfg.get("use_temporal_encoder"),
                "use_prerequisite_edges": model_cfg.get("use_prerequisite_edges"),
                "auc": float(metrics["auc"]),
                "accuracy": float(metrics["accuracy"]),
                "f1": float(metrics["f1"]),
                "bce_loss": float(metrics["bce_loss"]),
                "run_dir": str(summary["run_dir"]),
                "config_path": str(summary["config_path"]),
            }
        )
    rows.sort(key=lambda row: (-row["auc"], -row["accuracy"], -row["f1"], row["bce_loss"], row["run_name"]))
    for rank, row in enumerate(rows, start=1):
        row["rank_by_auc"] = rank
    return rows


def _write_comparison_artifacts(output_root: Path, run_summaries: list[dict[str, Any]], configs: list[dict[str, Any]]) -> dict[str, str]:
    comparison_dir = output_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    rows = _build_comparison_rows(run_summaries, configs)

    json_path = comparison_dir / "comparison_metrics.json"
    json_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    csv_path = comparison_dir / "comparison_metrics.csv"
    fieldnames = [
        "rank_by_auc",
        "run_name",
        "model_type",
        "relation_mode",
        "max_users",
        "use_graph_encoder",
        "use_temporal_encoder",
        "use_prerequisite_edges",
        "auc",
        "accuracy",
        "f1",
        "bce_loss",
        "run_dir",
        "config_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = comparison_dir / "comparison_report.md"
    lines = [
        "# Comparison Report",
        "",
        f"Best run by AUC: **{rows[0]['run_name']}**",
        "",
        "| Rank | Run | Model | Relation Mode | AUC | Accuracy | F1 | BCE Loss |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank_by_auc']} | {row['run_name']} | {row['model_type']} | {row['relation_mode']} | {row['auc']:.4f} | {row['accuracy']:.4f} | {row['f1']:.4f} | {row['bce_loss']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    labels = [row["run_name"] for row in rows]
    auc_plot = make_ablation_plot(run_summaries, comparison_dir / "comparison_auc.svg", metric_name="auc")
    accuracy_plot = save_ablation_bar_chart_svg(labels, [row["accuracy"] for row in rows], comparison_dir / "comparison_accuracy.svg", title="Comparison by Accuracy", metric_name="accuracy")
    f1_plot = save_ablation_bar_chart_svg(labels, [row["f1"] for row in rows], comparison_dir / "comparison_f1.svg", title="Comparison by F1", metric_name="f1")
    bce_plot = save_ablation_bar_chart_svg(labels, [row["bce_loss"] for row in rows], comparison_dir / "comparison_bce_loss.svg", title="Comparison by BCE Loss (lower is better)", metric_name="bce_loss")

    return {
        "comparison_json": str(json_path),
        "comparison_csv": str(csv_path),
        "comparison_report": str(md_path),
        "comparison_auc_plot": str(auc_plot),
        "comparison_accuracy_plot": str(accuracy_plot),
        "comparison_f1_plot": str(f1_plot),
        "comparison_bce_loss_plot": str(bce_plot),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the THGKT project and run one or more configured experiments.")
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "artifacts" / "final_project_run"),
        help="Directory where run artifacts will be saved.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip the pytest verification pass and run only the selected experiments.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device preference. 'auto' prefers CUDA when available.",
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="Experiment config to run. Pass multiple times to run multiple configs. Defaults to the Phase 12 smoke suite.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=30,
        help="How often to print a still-running heartbeat when no new progress event has arrived.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device_info = _detect_device(args.device)
    if _requires_cpu_fallback_confirmation(device_info):
        _confirm_cpu_fallback(device_info)
    source_configs = _resolve_source_configs(args.configs)

    resolved_configs: list[tuple[Path, dict[str, Any]]] = [
        _write_resolved_config(path, output_root, device=device_info["selected"])
        for path in source_configs
    ]

    total_units = (0 if args.skip_tests else 2) + sum(_experiment_total_units(cfg) for _, cfg in resolved_configs)
    if len(resolved_configs) > 1:
        total_units += 1
    reporter = ProgressReporter(total_units)
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_worker,
        args=(reporter, stop_event, max(5, int(args.heartbeat_seconds))),
        daemon=True,
    )
    heartbeat_thread.start()

    try:
        reporter.status(
            f"Starting project run on device {device_info['selected']}"
            + (f" ({device_info['gpu_name']})" if 'gpu_name' in device_info else "")
            + (f"; {device_info['reason']}" if 'reason' in device_info else "")
        )

        summary: dict[str, Any] = {
            "project_root": str(ROOT),
            "output_root": str(output_root),
            "device": device_info,
            "source_configs": [str(path) for path in source_configs],
            "resolved_configs": [str(path) for path, _ in resolved_configs],
            "heartbeat_seconds": max(5, int(args.heartbeat_seconds)),
        }

        completed_units = 0
        if not args.skip_tests:
            reporter.status("Running pytest verification")
            summary["verification"] = _run_pytest()
            completed_units += 2
            reporter.complete_to(completed_units, "Pytest verification finished")

        run_summaries: list[dict[str, Any]] = []
        resolved_config_payloads: list[dict[str, Any]] = []
        for resolved_path, config in resolved_configs:
            run_name = str(config["run"]["name"])
            epochs = int(config["training"].get("epochs", 1))
            has_explainability = bool(config.get("explainability", {}).get("enabled", True)) and str(config["model"]["type"]) == "thgkt"
            experiment_units = _experiment_total_units(config)
            experiment_base = completed_units
            event_state = {"max_units": 0}

            def _callback(event: dict[str, Any]) -> None:
                message = _event_message(run_name, event)
                units = _event_completed_units(event, epochs=epochs, has_explainability=has_explainability)
                if units > event_state["max_units"]:
                    event_state["max_units"] = units
                    reporter.complete_to(experiment_base + units, message)
                else:
                    reporter.status(message)

            run_summary = run_experiment_from_config(resolved_path, output_root=output_root, progress_callback=_callback)
            run_summaries.append(run_summary)
            resolved_config_payloads.append(config)
            completed_units = experiment_base + experiment_units
            reporter.complete_to(completed_units, f"{run_name}: artifacts finalized")

        summary["runs"] = run_summaries
        if len(run_summaries) == 3 and not args.configs:
            summary["suite"] = {
                "baseline": run_summaries[0],
                "thgkt": run_summaries[1],
                "ablation": run_summaries[2],
            }

        if len(run_summaries) > 1:
            reporter.status("Generating comparison artifacts")
            comparison_artifacts = _write_comparison_artifacts(output_root, run_summaries, resolved_config_payloads)
            completed_units += 1
            reporter.complete_to(completed_units, "Comparison artifacts saved")
            summary["comparison"] = comparison_artifacts
            if "suite" in summary:
                summary["suite"]["comparison"] = comparison_artifacts

        summary_path = output_root / "run_project_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["summary_path"] = str(summary_path)
        print(json.dumps(summary, indent=2))
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
