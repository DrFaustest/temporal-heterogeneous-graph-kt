from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thgkt.experiment import run_experiment_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a THGKT experiment from config.")
    parser.add_argument("--config", required=True, help="Path to experiment config.")
    parser.add_argument("--output-root", default=None, help="Optional output root directory.")
    args = parser.parse_args()

    summary = run_experiment_from_config(args.config, output_root=args.output_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
