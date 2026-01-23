"""Configuration loading and saving for LocalWispr."""

import copy
import logging
import sys
import threading
import tomllib
from pathlib import Path
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


def _get_config_path() -> Path:
    """Get the path to config.toml.

    When running as a PyInstaller bundle, looks for config.toml next to the EXE.
    Otherwise, looks in the project root (parent of localwispr package).

    Returns:
        Path to config.toml location.
    """
    # Check if running as frozen PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # Running as compiled EXE - look next to the executable
        path = Path(sys.executable).parent / "config.toml"
        logger.info("config_path: frozen mode, path=%s, exists=%s", path, path.exists())
    else:
        # Running as script - look in project root
        path = Path(__file__).parent.parent / "config.toml"
        logger.info("config_path: script mode, path=%s, exists=%s", path, path.exists())
    return path


class ModelConfig(TypedDict, total=False):
    """Model configuration settings."""

    name: str
    device: str
    compute_type: str
    language: str  # Optional language code for transcription


class HotkeyConfig(TypedDict):
    """Hotkey configuration settings."""

    mode: str  # "push-to-talk" or "toggle"
    modifiers: list[str]  # ["win", "ctrl", "shift"]
    audio_feedback: bool
    mute_system: bool  # Mute system audio during recording


class ContextConfig(TypedDict):
    """Context detection configuration settings."""

    coding_apps: list[str]
    planning_apps: list[str]
    coding_keywords: list[str]
    planning_keywords: list[str]


class OutputConfig(TypedDict):
    """Output configuration settings."""

    auto_paste: bool  # Whether to auto-paste or clipboard-only
    paste_delay_ms: int  # Delay before paste to ensure focus


class VocabularyConfig(TypedDict):
    """Vocabulary configuration settings."""

    words: list[str]  # Custom words for better transcription


class Config(TypedDict, total=False):
    """Full application configuration."""

    model: ModelConfig
    hotkeys: HotkeyConfig
    context: ContextConfig
    output: OutputConfig
    vocabulary: VocabularyConfig


DEFAULT_CONFIG: Config = {
    "model": {
        "name": "large-v3",
        "device": "cuda",
        "compute_type": "float16",
        "language": "auto",
    },
    "hotkeys": {
        "mode": "push-to-talk",
        "modifiers": ["win", "ctrl", "shift"],
        "audio_feedback": True,
        "mute_system": False,
    },
    "vocabulary": {
        "words": [],
    },
    "context": {
        "coding_apps": [
            "code",
            "pycharm",
            "intellij",
            "vim",
            "neovim",
            "visual studio",
            "sublime",
            "atom",
            "emacs",
        ],
        "planning_apps": [
            "notion",
            "obsidian",
            "todoist",
            "jira",
            "asana",
            "trello",
            "linear",
        ],
        "coding_keywords": [
            "function",
            "variable",
            "import",
            "class",
            "def",
            "return",
            "async",
            "await",
            "const",
            "let",
            "var",
            "public",
            "private",
            "interface",
            "type",
            "null",
            "undefined",
        ],
        "planning_keywords": [
            "task",
            "project",
            "milestone",
            "deadline",
            "goal",
            "plan",
            "schedule",
            "priority",
            "action",
            "item",
            "todo",
            "complete",
            "review",
        ],
    },
    "output": {
        "auto_paste": True,  # Auto-paste after transcription
        "paste_delay_ms": 50,  # Small delay before paste
    },
}


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from TOML file.

    Args:
        config_path: Path to config file. Defaults to config.toml next to EXE
                     (when frozen) or in project root (when running as script).

    Returns:
        Configuration dictionary with defaults applied.
    """
    if config_path is None:
        config_path = _get_config_path()

    logger.debug("load_config: config_path=%s, exists=%s", config_path, config_path.exists())

    # Use deepcopy to avoid reference issues with nested dicts
    config = copy.deepcopy(DEFAULT_CONFIG)

    if config_path.exists():
        with open(config_path, "rb") as f:
            user_config = tomllib.load(f)
        logger.debug("load_config: loaded user_config from file, hotkeys=%s", user_config.get("hotkeys"))
        config = _deep_merge(config, user_config)
        logger.debug("load_config: after merge, hotkeys=%s", config.get("hotkeys"))
    else:
        # Create default config file for user to edit
        logger.info("load_config: config file not found at %s, creating from defaults", config_path)
        try:
            save_config(config, config_path)
            logger.info("load_config: created default config at %s", config_path)
        except Exception as e:
            logger.warning("load_config: failed to create default config: %s", e)

    return config


def save_config(config: Config, config_path: Path | None = None) -> None:
    """Save configuration to TOML file.

    Args:
        config: Configuration dictionary to save.
        config_path: Path to config file. Defaults to config.toml next to EXE
                     (when frozen) or in project root (when running as script).
    """
    if config_path is None:
        config_path = _get_config_path()

    # Build TOML content with comments
    lines = ["# LocalWispr Configuration", ""]

    # Model section
    lines.append("[model]")
    lines.append("# Whisper model to use: tiny, base, small, medium, large-v2, large-v3")
    lines.append(f'name = "{config["model"]["name"]}"')
    lines.append("")
    lines.append("# Device: cuda (GPU) or cpu")
    lines.append(f'device = "{config["model"]["device"]}"')
    lines.append("")
    lines.append("# Compute type: float16 (GPU), int8 (CPU), or float32")
    lines.append(f'compute_type = "{config["model"]["compute_type"]}"')
    lines.append("")
    lines.append("# Language: auto, en, es, fr, de, it, pt, nl, ru, zh, ja, ko")
    language = config.get("model", {}).get("language", "auto")
    lines.append(f'language = "{language}"')
    lines.append("")

    # Hotkeys section
    lines.append("[hotkeys]")
    lines.append("# Recording activation mode:")
    lines.append('#   "push-to-talk" - Hold keys to record, release to stop and transcribe')
    lines.append('#   "toggle" - Press once to start recording, press again to stop and transcribe')
    lines.append(f'mode = "{config["hotkeys"]["mode"]}"')
    lines.append("")
    lines.append("# Modifier key combination to activate recording")
    lines.append('# Available modifiers: "win", "ctrl", "shift", "alt"')
    modifiers = config["hotkeys"]["modifiers"]
    modifiers_str = ", ".join(f'"{m}"' for m in modifiers)
    lines.append(f"modifiers = [{modifiers_str}]")
    lines.append("")
    lines.append("# Play audio feedback sounds when recording starts/stops")
    audio_fb = "true" if config["hotkeys"]["audio_feedback"] else "false"
    lines.append(f"audio_feedback = {audio_fb}")
    lines.append("")
    lines.append("# Mute system audio during recording (prevents feedback)")
    mute_sys = "true" if config["hotkeys"].get("mute_system", False) else "false"
    lines.append(f"mute_system = {mute_sys}")
    lines.append("")

    # Output section
    lines.append("[output]")
    lines.append("# Auto-paste after transcription (or clipboard-only)")
    auto_paste = "true" if config["output"]["auto_paste"] else "false"
    lines.append(f"auto_paste = {auto_paste}")
    lines.append("")
    lines.append("# Delay before paste to ensure focus (milliseconds)")
    lines.append(f"paste_delay_ms = {config['output']['paste_delay_ms']}")
    lines.append("")

    # Vocabulary section
    vocab = config.get("vocabulary", {}).get("words", [])
    if vocab:
        lines.append("[vocabulary]")
        lines.append("# Custom words for better transcription accuracy")
        vocab_str = ", ".join(f'"{w}"' for w in vocab)
        lines.append(f"words = [{vocab_str}]")
        lines.append("")

    # Write to file
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# Thread-safe config cache
_cached_config: Config | None = None
_config_lock = threading.Lock()


def get_config() -> Config:
    """Get cached config (thread-safe, loads on first call).

    This is the preferred way to access config throughout the application.
    The config is loaded once from disk and cached for subsequent calls.

    Returns:
        Configuration dictionary with defaults applied.
    """
    global _cached_config
    with _config_lock:
        if _cached_config is None:
            logger.debug("Loading config for first time")
            _cached_config = load_config()
        return _cached_config


def reload_config() -> Config:
    """Force reload config from disk (thread-safe, for settings save).

    Call this after saving settings to refresh the cached config.
    If reload fails, keeps the previous config and logs an error.

    Returns:
        Updated configuration dictionary.

    Raises:
        Exception: If reload fails and no previous config exists.
    """
    global _cached_config
    with _config_lock:
        try:
            new_config = load_config()
            _cached_config = new_config
            logger.info("Config reloaded successfully")
            return _cached_config
        except Exception as e:
            logger.error(f"Config reload failed: {e}, keeping previous config")
            if _cached_config is not None:
                return _cached_config
            raise  # Re-raise if no fallback available


def clear_config_cache() -> None:
    """Clear cached config (for testing).

    Resets the cache so next get_config() call will reload from disk.
    """
    global _cached_config
    with _config_lock:
        _cached_config = None
