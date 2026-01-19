"""Configuration loading for LocalWispr."""

import tomllib
from pathlib import Path
from typing import Any, TypedDict


class ModelConfig(TypedDict):
    """Model configuration settings."""

    name: str
    device: str
    compute_type: str


class HotkeyConfig(TypedDict):
    """Hotkey configuration settings."""

    mode: str  # "push-to-talk" or "toggle"
    modifiers: list[str]  # ["win", "ctrl", "shift"]
    audio_feedback: bool


class Config(TypedDict):
    """Full application configuration."""

    model: ModelConfig
    hotkeys: HotkeyConfig


DEFAULT_CONFIG: Config = {
    "model": {
        "name": "large-v3",
        "device": "cuda",
        "compute_type": "float16",
    },
    "hotkeys": {
        "mode": "push-to-talk",
        "modifiers": ["win", "ctrl", "shift"],
        "audio_feedback": True,
    },
}


def load_config(config_path: Path | None = None) -> Config:
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
