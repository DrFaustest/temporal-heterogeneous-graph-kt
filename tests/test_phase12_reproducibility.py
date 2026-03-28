from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import run_project


ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> dict:
    completed = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)


def test_run_experiment_script_and_plot_script_work(tmp_path) -> None:
    summary = _run(
        "scripts/run_experiment.py",
        "--config",
        "configs/phase12/baseline_assistments.yaml",
        "--output-root",
        str(tmp_path / "runs"),
    )

    run_dir = Path(summary["run_dir"])
    assert run_dir.exists()
    assert Path(summary["artifacts"]["checkpoint"]).exists()
    assert Path(summary["artifacts"]["metrics"]).exists()
    assert Path(summary["artifacts"]["training_plot"]).exists()
    assert Path(summary["artifacts"]["roc_plot"]).exists()

    plot_summary = _run("scripts/plot_run.py", "--run-dir", str(run_dir))
    assert Path(plot_summary["training_plot"]).exists()
    assert Path(plot_summary["roc_plot"]).exists()


def test_phase12_suite_runs_baseline_thgkt_and_ablation(tmp_path) -> None:
    summary = _run("scripts/run_phase12_suite.py", "--output-root", str(tmp_path / "suite_runs"))

    assert Path(summary["baseline"]["run_dir"]).exists()
    assert Path(summary["thgkt"]["run_dir"]).exists()
    assert Path(summary["ablation"]["run_dir"]).exists()
    assert Path(summary["comparison"]["output"]).exists()
    assert Path(summary["thgkt"]["artifacts"]["concept_plot"]).exists()
    assert Path(summary["ablation"]["artifacts"]["experiment_config"]).exists()


def test_readme_contains_reproduction_instructions() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "scripts/run_phase12_suite.py" in readme
    assert "scripts/run_experiment.py" in readme
    assert "python -m pytest -q" in readme
    assert "configs/phase12/thgkt_assistments.yaml" in readme


def test_run_project_multi_config_writes_comparison_artifacts(tmp_path) -> None:
    output_root = tmp_path / "run_project_suite"
    subprocess.run(
        [
            sys.executable,
            "run_project.py",
            "--skip-tests",
            "--output-root",
            str(output_root),
            "--config",
            "configs/phase12/baseline_assistments.yaml",
            "--config",
            "configs/phase12/thgkt_assistments.yaml",
            "--heartbeat-seconds",
            "5",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    summary = json.loads((output_root / "run_project_summary.json").read_text(encoding="utf-8"))
    comparison = summary["comparison"]
    assert Path(comparison["comparison_json"]).exists()
    assert Path(comparison["comparison_csv"]).exists()
    assert Path(comparison["comparison_report"]).exists()
    assert Path(comparison["comparison_auc_plot"]).exists()
    assert Path(comparison["comparison_accuracy_plot"]).exists()
    assert Path(comparison["comparison_f1_plot"]).exists()
    assert Path(comparison["comparison_bce_loss_plot"]).exists()


def test_run_project_requires_confirmation_only_for_explicit_cuda_cpu_fallback() -> None:
    assert run_project._requires_cpu_fallback_confirmation({"requested": "cuda", "selected": "cpu"}) is True
    assert run_project._requires_cpu_fallback_confirmation({"requested": "cuda:0", "selected": "cpu"}) is True
    assert run_project._requires_cpu_fallback_confirmation({"requested": "auto", "selected": "cpu"}) is False
    assert run_project._requires_cpu_fallback_confirmation({"requested": "cpu", "selected": "cpu"}) is False
    assert run_project._requires_cpu_fallback_confirmation({"requested": "cuda", "selected": "cuda"}) is False


def test_run_project_cpu_fallback_confirmation_accepts_yes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "yes")

    run_project._confirm_cpu_fallback(
        {
            "requested": "cuda",
            "selected": "cpu",
            "reason": "requested cuda but CUDA is unavailable; falling back to cpu",
        }
    )

    captured = capsys.readouterr()
    assert "WARNING: requested GPU device 'cuda', but the run resolved to CPU." in captured.err
    assert "falling back to cpu" in captured.err


def test_run_project_cpu_fallback_confirmation_aborts_when_declined(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "n")

    with pytest.raises(SystemExit, match="Aborted: GPU was requested and CPU fallback was declined."):
        run_project._confirm_cpu_fallback(
            {
                "requested": "cuda",
                "selected": "cpu",
                "reason": "requested cuda but CUDA is unavailable; falling back to cpu",
            }
        )

    captured = capsys.readouterr()
    assert "WARNING: requested GPU device 'cuda', but the run resolved to CPU." in captured.err
