from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thgkt.config import load_config


DEFAULT_MODEL_CONFIGS = [
    ROOT / "configs" / "ednet_20k" / "dkt_baseline.json",
    ROOT / "configs" / "ednet_20k" / "graph_only.json",
    ROOT / "configs" / "ednet_20k" / "sakt_baseline.json",
    ROOT / "configs" / "ednet_20k" / "thgkt_cooccurrence_graph.json",
    ROOT / "configs" / "ednet_20k" / "thgkt_full.json",
    ROOT / "configs" / "ednet_20k" / "thgkt_no_prereq.json",
    ROOT / "configs" / "ednet_20k" / "thgkt_no_temporal.json",
]


def _ensure_cuda_available() -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is not installed, so this GPU-only suite cannot run.") from exc

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. This suite is GPU-only and will not fall back to CPU.")

    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
    }


def _suite_run_name(base_run_name: str, seed: int, *, max_users: int) -> str:
    normalized = base_run_name.replace("20k", f"{max_users // 1000}k")
    if normalized == base_run_name:
        normalized = f"{base_run_name}_{max_users}users"
    return f"{normalized}_seed{seed:02d}"


def _materialize_suite_configs(
    *,
    model_configs: list[Path],
    output_root: Path,
    max_users: int,
    seeds: list[int],
) -> list[Path]:
    generated_dir = output_root / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    generated_paths: list[Path] = []
    for config_path in model_configs:
        config = load_config(config_path)
        base_run_name = str(config["run"]["name"])
        for seed in seeds:
            generated = json.loads(json.dumps(config))
            generated["run"]["name"] = _suite_run_name(base_run_name, seed, max_users=max_users)
            generated["dataset"]["max_users"] = max_users
            generated["split"]["random_seed"] = seed
            generated["training"]["random_seed"] = seed
            generated["training"]["device"] = "cuda"

            generated_path = generated_dir / f"{generated['run']['name']}.json"
            generated_path.write_text(json.dumps(generated, indent=2), encoding="utf-8")
            generated_paths.append(generated_path)
    return generated_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all EdNet benchmark models on 1k students across multiple seeds with GPU-only execution."
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "artifacts" / "ednet_1k_seed_suite"),
        help="Directory where generated configs and run artifacts will be saved.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=1000,
        help="Number of EdNet users to include in each run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(1, 11)),
        help="Random seeds to use for both splitting and training.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=30,
        help="How often run_project should emit heartbeat progress messages.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cuda_info = _ensure_cuda_available()
    generated_configs = _materialize_suite_configs(
        model_configs=DEFAULT_MODEL_CONFIGS,
        output_root=output_root,
        max_users=int(args.max_users),
        seeds=[int(seed) for seed in args.seeds],
    )

    command = [
        sys.executable,
        "run_project.py",
        "--skip-tests",
        "--device",
        "cuda",
        "--output-root",
        str(output_root),
        "--heartbeat-seconds",
        str(max(5, int(args.heartbeat_seconds))),
    ]
    for config_path in generated_configs:
        command.extend(["--config", str(config_path)])

    print(
        json.dumps(
            {
                "gpu": cuda_info,
                "max_users": int(args.max_users),
                "seeds": [int(seed) for seed in args.seeds],
                "model_configs": [str(path) for path in DEFAULT_MODEL_CONFIGS],
                "generated_config_count": len(generated_configs),
                "output_root": str(output_root),
            },
            indent=2,
        ),
        flush=True,
    )

    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
