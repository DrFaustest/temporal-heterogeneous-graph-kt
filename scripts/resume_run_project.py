from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_project
from thgkt.config import load_config
from thgkt.data import save_json
from thgkt.experiment import _build_runtime_context_and_model, collect_predictions, run_experiment_from_config
from thgkt.graph import load_graph_artifacts
from thgkt.reporting import save_roc_curve_svg, save_training_curves_svg
from thgkt.sequences import load_sequence_artifacts
from thgkt.training import Evaluator, build_sequence_loader, load_checkpoint


def _load_config_list(output_root: Path, config_args: list[str] | None) -> list[Path]:
    if config_args:
        return [Path(arg) if Path(arg).is_absolute() else (ROOT / arg).resolve() for arg in config_args]
    summary_path = output_root / 'run_project_summary.json'
    if not summary_path.exists():
        raise FileNotFoundError(f'No config list provided and no summary found at {summary_path}')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    return [Path(path) if Path(path).is_absolute() else (ROOT / path).resolve() for path in summary['resolved_configs']]


def _paths(run_dir: Path) -> dict[str, Path]:
    return {
        'run_summary': run_dir / 'run_summary.json',
        'training_config': run_dir / 'config' / 'training_config.json',
        'metrics': run_dir / 'metrics' / 'metrics.json',
        'test_metrics': run_dir / 'metrics' / 'test_metrics.json',
        'test_predictions': run_dir / 'metrics' / 'test_predictions.json',
        'checkpoint': run_dir / 'checkpoints' / 'best_checkpoint.pt',
        'graph': run_dir / 'data' / 'graph.json',
        'sequences': run_dir / 'data' / 'sequences.json',
        'roc_plot': run_dir / 'plots' / 'roc_curve.svg',
        'training_plot': run_dir / 'plots' / 'training_curves.svg',
        'experiment_config': run_dir / 'config' / 'experiment_config.json',
    }


def _is_complete(config: dict[str, Any], run_dir: Path) -> bool:
    paths = _paths(run_dir)
    if not all(paths[key].exists() for key in ('run_summary', 'training_config', 'metrics', 'test_metrics', 'checkpoint')):
        return False
    training_payload = json.loads(paths['training_config'].read_text(encoding='utf-8'))
    return int(training_payload.get('epochs', -1)) == int(config['training'].get('epochs', -2))


def _is_recoverable_checkpoint(config: dict[str, Any], run_dir: Path) -> bool:
    paths = _paths(run_dir)
    if not all(paths[key].exists() for key in ('checkpoint', 'graph', 'sequences')):
        return False
    if not paths['training_config'].exists():
        return True
    training_payload = json.loads(paths['training_config'].read_text(encoding='utf-8'))
    expected_epochs = int(config['training'].get('epochs', -1))
    current_epochs = int(training_payload.get('epochs', -2))
    if current_epochs == expected_epochs:
        return False
    reference_times = []
    for key in ('training_config', 'metrics', 'run_summary', 'test_metrics'):
        if paths[key].exists():
            reference_times.append(paths[key].stat().st_mtime)
    latest_reference = max(reference_times) if reference_times else 0.0
    return paths['checkpoint'].stat().st_mtime > latest_reference


def _recover_run(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    run_name = str(config['run']['name'])
    run_dir = output_root / run_name
    paths = _paths(run_dir)

    graph = load_graph_artifacts(paths['graph'])
    sequences = load_sequence_artifacts(paths['sequences'])
    model, context = _build_runtime_context_and_model(config, sequences, graph)
    payload = load_checkpoint(model, paths['checkpoint'])

    test_loader = build_sequence_loader(
        sequences.test.examples,
        batch_size=int(config['training']['batch_size']),
        shuffle=False,
        random_seed=int(config['training'].get('random_seed', 42)),
    )
    evaluator = Evaluator()
    test_metrics = evaluator.evaluate(model, test_loader, context=context)
    predictions = collect_predictions(model, test_loader, context=context)
    save_json(test_metrics, paths['test_metrics'])
    save_json(predictions, paths['test_predictions'])
    save_roc_curve_svg(predictions['probs'], predictions['targets'], paths['roc_plot'])

    if paths['metrics'].exists():
        metrics_payload = load_config(paths['metrics'])
        if 'train_history' in metrics_payload and 'val_history' in metrics_payload:
            save_training_curves_svg(metrics_payload['train_history'], metrics_payload['val_history'], paths['training_plot'])

    run_summary = {
        'run_name': run_name,
        'config_path': str(config_path),
        'run_dir': str(run_dir),
        'artifacts': {
            'canonical_bundle': str(run_dir / 'data' / 'canonical_bundle.json'),
            'splits': str(run_dir / 'data' / 'splits.json'),
            'graph': str(paths['graph']),
            'sequences': str(paths['sequences']),
            'checkpoint': str(paths['checkpoint']),
            'metrics': str(paths['metrics']),
            'test_metrics': str(paths['test_metrics']),
            'test_predictions': str(paths['test_predictions']),
            'training_plot': str(paths['training_plot']),
            'roc_plot': str(paths['roc_plot']),
            'experiment_config': str(paths['experiment_config']),
        },
        'best_val_metrics': dict(payload.get('extra', {}).get('val_metrics', {})),
        'test_metrics': test_metrics,
        'recovered_from_checkpoint': True,
        'recovered_best_epoch': payload.get('extra', {}).get('epoch'),
    }
    save_json(run_summary, paths['run_summary'])
    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Resume an interrupted run_project suite.')
    parser.add_argument('--output-root', required=True, help='Suite output root.')
    parser.add_argument('--config', action='append', dest='configs', help='Resolved config(s) to process. Defaults to summary order.')
    args = parser.parse_args()

    output_root = Path(args.output_root)
    previous_summary_path = output_root / 'run_project_summary.json'
    previous_summary = json.loads(previous_summary_path.read_text(encoding='utf-8')) if previous_summary_path.exists() else {}

    resolved_configs = _load_config_list(output_root, args.configs)
    run_summaries: list[dict[str, Any]] = []
    config_payloads: list[dict[str, Any]] = []
    recovered_runs: list[str] = []
    reused_runs: list[str] = []
    executed_runs: list[str] = []

    for config_path in resolved_configs:
        config = load_config(config_path)
        run_name = str(config['run']['name'])
        run_dir = output_root / run_name
        config_payloads.append(config)
        if _is_complete(config, run_dir):
            print(f'[reuse] {run_name}: existing 20-epoch artifacts are complete', flush=True)
            run_summaries.append(json.loads((run_dir / 'run_summary.json').read_text(encoding='utf-8')))
            reused_runs.append(run_name)
            continue
        if _is_recoverable_checkpoint(config, run_dir):
            print(f'[recover] {run_name}: rebuilding summary/evaluation from saved checkpoint', flush=True)
            run_summaries.append(_recover_run(config_path, output_root))
            recovered_runs.append(run_name)
            continue
        print(f'[run] {run_name}: executing experiment from config', flush=True)
        run_summaries.append(run_experiment_from_config(config_path, output_root=output_root))
        executed_runs.append(run_name)

    comparison_artifacts = run_project._write_comparison_artifacts(output_root, run_summaries, config_payloads)
    summary = {
        'project_root': str(ROOT),
        'output_root': str(output_root),
        'device': previous_summary.get('device', {}),
        'source_configs': previous_summary.get('source_configs', [str(path) for path in resolved_configs]),
        'resolved_configs': [str(path) for path in resolved_configs],
        'runs': run_summaries,
        'comparison': comparison_artifacts,
        'resumed': True,
        'reused_runs': reused_runs,
        'recovered_runs': recovered_runs,
        'executed_runs': executed_runs,
    }
    summary_path = output_root / 'run_project_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'summary_path': str(summary_path), 'reused_runs': reused_runs, 'recovered_runs': recovered_runs, 'executed_runs': executed_runs}, indent=2), flush=True)


if __name__ == '__main__':
    main()
