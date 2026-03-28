from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thgkt.data.adapters.ednet import EdNetAdapter, EdNetAdapterConfig
from thgkt.explainability import UserTagMapConfig, build_user_tag_map, export_user_tag_map_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a per-user EdNet tag compatibility map.")
    parser.add_argument("--raw-path", required=True, help="Path to the EdNet KT1 directory or one of its parents.")
    parser.add_argument("--contents-path", required=True, help="Path to the EdNet contents directory or one of its parents.")
    parser.add_argument("--user-id", required=True, help="User file stem, for example u1.")
    parser.add_argument(
        "--edge-scope",
        choices=["all_questions", "seen_questions"],
        default="all_questions",
        help="Whether edge counts come from the full question bank or only questions seen by the user.",
    )
    parser.add_argument(
        "--min-edge-question-count",
        type=int,
        default=1,
        help="Drop edges that co-occur in fewer than this many questions.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for JSON and SVG outputs. Defaults to artifacts/user_tag_maps/<user-id>.",
    )
    args = parser.parse_args()

    adapter = EdNetAdapter(config=EdNetAdapterConfig(user_ids=(args.user_id,)))
    bundle = adapter.to_canonical(adapter.load_raw(args.raw_path, contents_path=args.contents_path))
    tag_map = build_user_tag_map(
        bundle,
        args.user_id,
        config=UserTagMapConfig(
            edge_scope=args.edge_scope,
            min_edge_question_count=args.min_edge_question_count,
        ),
    )

    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "artifacts" / "user_tag_maps" / args.user_id
    exported = export_user_tag_map_artifacts(tag_map, output_dir)
    payload = {
        "student_id": tag_map.student_id,
        "output_dir": str(output_dir),
        **exported,
        "metadata": tag_map.metadata,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
