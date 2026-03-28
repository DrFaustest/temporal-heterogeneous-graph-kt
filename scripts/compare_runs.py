from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thgkt.experiment import make_ablation_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple runs with an ablation bar chart.")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Run directories to compare.")
    parser.add_argument("--metric", default="auc", help="Metric name to compare.")
    parser.add_argument("--output", required=True, help="Output SVG path.")
    args = parser.parse_args()

    summaries = [json.loads((Path(run_dir) / "run_summary.json").read_text(encoding="utf-8")) for run_dir in args.run_dirs]
    output = make_ablation_plot(summaries, args.output, metric_name=args.metric)
    print(json.dumps({"output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
