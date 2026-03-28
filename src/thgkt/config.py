"""Config loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


class ConfigError(ValueError):
    """Raised when a config file cannot be parsed or validated."""


def _parse_yaml_text(raw_text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        yaml = None

    if yaml is not None:
        loaded = yaml.safe_load(raw_text)
    else:
        try:
            loaded = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                "PyYAML is not installed and the config is not valid JSON-compatible YAML."
            ) from exc

    if not isinstance(loaded, dict):
        raise ConfigError("Config root must be a mapping.")

    return loaded


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary."""

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")

    return _parse_yaml_text(config_path.read_text(encoding="utf-8"))


def load_default_config() -> dict[str, Any]:
    """Load the repository default config."""

    return load_config(CONFIG_DIR / "default.yaml")
