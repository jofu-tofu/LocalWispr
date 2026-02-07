"""Configuration loading, saving, and caching for LocalWispr."""

from localwispr.config.types import (
    Config,
    ContextConfig,
    DEFAULT_CONFIG,
    HotkeyConfig,
    ModelConfig,
    OutputConfig,
    StreamingConfig,
    VocabularyConfig,
)
from localwispr.config.loader import (
    _deep_merge,
    _get_appdata_config_path,
    _get_defaults_path,
    load_config,
)
from localwispr.config.saver import save_config
from localwispr.config.cache import clear_config_cache, get_config, reload_config
from localwispr.config.migration import _migrate_legacy_config

__all__ = [
    # Types
    "Config",
    "ContextConfig",
    "DEFAULT_CONFIG",
    "HotkeyConfig",
    "ModelConfig",
    "OutputConfig",
    "StreamingConfig",
    "VocabularyConfig",
    # Loading
    "load_config",
    "_deep_merge",
    "_get_appdata_config_path",
    "_get_defaults_path",
    # Saving
    "save_config",
    # Cache
    "get_config",
    "reload_config",
    "clear_config_cache",
    # Migration
    "_migrate_legacy_config",
]
