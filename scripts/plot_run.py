from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thgkt.reporting import save_roc_curve_svg, save_training_curves_svg


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots from a saved run directory.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing metrics outputs.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics = json.loads((run_dir / "metrics" / "metrics.json").read_text(encoding="utf-8"))
    predictions = json.loads((run_dir / "metrics" / "test_predictions.json").read_text(encoding="utf-8"))
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    training = save_training_curves_svg(metrics["train_history"], metrics["val_history"], plots_dir / "training_curves.svg")
    roc = save_roc_curve_svg(predictions["probs"], predictions["targets"], plots_dir / "roc_curve.svg")
    print(json.dumps({"training_plot": str(training), "roc_plot": str(roc)}, indent=2))


if __name__ == "__main__":
    main()
