from __future__ import annotations

from pathlib import Path

import thgkt
from thgkt.config import load_config, load_default_config


def test_package_imports_successfully() -> None:
    assert thgkt.__version__ == "0.1.0"
    assert callable(thgkt.validate_bundle)
    assert thgkt.CanonicalBundle is not None


def test_default_config_loads() -> None:
    config = load_default_config()
    assert config["project"]["name"] == "thgkt"
    assert config["paths"]["artifacts_dir"] == "artifacts"


def test_schema_config_loads() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "schema.yaml"
    config = load_config(config_path)
    assert "interactions" in config["canonical_tables"]
    assert "concept_relations" in config["canonical_tables"]
