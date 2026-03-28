from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> dict:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 12 smoke suite.")
    parser.add_argument("--output-root", required=True, help="Root directory for the smoke runs.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    baseline = _run([sys.executable, "scripts/run_experiment.py", "--config", "configs/phase12/baseline_assistments.yaml", "--output-root", str(output_root)])
    thgkt = _run([sys.executable, "scripts/run_experiment.py", "--config", "configs/phase12/thgkt_assistments.yaml", "--output-root", str(output_root)])
    ablation = _run([sys.executable, "scripts/run_experiment.py", "--config", "configs/phase12/thgkt_ablation_assistments.yaml", "--output-root", str(output_root)])
    comparison = _run([
        sys.executable,
        "scripts/compare_runs.py",
        "--run-dirs",
        baseline["run_dir"],
        thgkt["run_dir"],
        ablation["run_dir"],
        "--metric",
        "auc",
        "--output",
        str(output_root / "ablation_comparison.svg"),
    ])
    summary = {
        "baseline": baseline,
        "thgkt": thgkt,
        "ablation": ablation,
        "comparison": comparison,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
