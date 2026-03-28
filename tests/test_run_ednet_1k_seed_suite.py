from __future__ import annotations

import json
from pathlib import Path

from scripts.run_ednet_1k_seed_suite import _materialize_suite_configs, _suite_run_name


ROOT = Path(__file__).resolve().parents[1]


def test_suite_run_name_relabels_20k_and_appends_seed() -> None:
    assert _suite_run_name("ednet20k_thgkt_full", 3, max_users=1000) == "ednet1k_thgkt_full_seed03"
    assert _suite_run_name("baseline", 12, max_users=1000) == "baseline_1000users_seed12"


def test_materialize_suite_configs_overrides_users_seeds_and_device(tmp_path) -> None:
    generated = _materialize_suite_configs(
        model_configs=[ROOT / "configs" / "ednet_20k" / "thgkt_full.json"],
        output_root=tmp_path,
        max_users=1000,
        seeds=[1, 2],
    )

    assert len(generated) == 2

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in generated]
    assert {payload["run"]["name"] for payload in payloads} == {
        "ednet1k_thgkt_full_seed01",
        "ednet1k_thgkt_full_seed02",
    }
    assert {payload["dataset"]["max_users"] for payload in payloads} == {1000}
    assert {payload["split"]["random_seed"] for payload in payloads} == {1, 2}
    assert {payload["training"]["random_seed"] for payload in payloads} == {1, 2}
    assert {payload["training"]["device"] for payload in payloads} == {"cuda"}
