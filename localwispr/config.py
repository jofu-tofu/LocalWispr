"""Configuration loading for LocalWispr."""

import tomllib
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "name": "large-v3",
        "device": "cuda",
        "compute_type": "float16",
    },
}


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from TOML file.

    Args:
        config_path: Path to config file. Defaults to config.toml in project root.

    Returns:
        Configuration dictionary with defaults applied.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.toml"

    config = DEFAULT_CONFIG.copy()

    if config_path.exists():
        with open(config_path, "rb") as f:
            user_config = tomllib.load(f)
        config = _deep_merge(config, user_config)

    return config


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
