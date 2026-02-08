"""Configuration loading for LocalWispr."""

import copy
import logging
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

from localwispr.config.types import DEFAULT_CONFIG, Config

logger = logging.getLogger(__name__)


def _get_appdata_config_path() -> Path:
    r"""Get path to user settings in AppData.

    Returns:
        C:\Users\<user>\AppData\Roaming\LocalWispr\<Stable|Test>\user-settings.toml
    """
    # Detect variant from sys.executable name
    variant = "Test" if "Test" in Path(sys.executable).stem else "Stable"

    appdata = os.environ.get("APPDATA")
    if not appdata:
        appdata = Path.home() / "AppData" / "Roaming"

    return Path(appdata) / "LocalWispr" / variant / "user-settings.toml"


def _get_defaults_path() -> Path:
    """Get the path to bundled default config.

    When running as a PyInstaller bundle, looks for config-defaults.toml next to the EXE.
    Otherwise, looks in the project root (parent of localwispr package).

    Returns:
        Path to config-defaults.toml location.
    """
    # Check if running as frozen PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # Running as compiled EXE - look next to the executable
        path = Path(sys.executable).parent / "config-defaults.toml"
        logger.info("defaults_path: frozen mode, path=%s, exists=%s", path, path.exists())
    else:
        # Running as script - look in project root (use config.toml as source of defaults)
        path = Path(__file__).parent.parent.parent / "config.toml"
        logger.info("defaults_path: script mode, path=%s, exists=%s", path, path.exists())
    return path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration with two-tier system.

    Priority (highest to lowest):
    1. User overrides in AppData (persists across rebuilds)
    2. Bundled defaults (overwritten on rebuild)
    3. Hardcoded DEFAULT_CONFIG

    Args:
        config_path: Optional override path for testing. If None, uses two-tier system.

    Returns:
        Configuration dictionary with all tiers merged.
    """
    from localwispr.config.migration import _migrate_legacy_config

    # Migration check (only runs once per install)
    _migrate_legacy_config()

    # If explicit path provided, use legacy single-file loading (for tests)
    if config_path is not None:
        logger.debug("load_config: explicit path=%s, exists=%s", config_path, config_path.exists())
        config = copy.deepcopy(DEFAULT_CONFIG)
        if config_path.exists():
            with open(config_path, "rb") as f:
                user_config = tomllib.load(f)
            config = _deep_merge(config, user_config)
        return config

    # Two-tier loading: Start with hardcoded defaults
    config = copy.deepcopy(DEFAULT_CONFIG)

    # Tier 2: Load bundled defaults (config-defaults.toml)
    defaults_path = _get_defaults_path()
    if defaults_path.exists():
        try:
            with open(defaults_path, "rb") as f:
                bundled_defaults = tomllib.load(f)
            config = _deep_merge(config, bundled_defaults)
            logger.debug("load_config: loaded bundled defaults from %s", defaults_path)
        except Exception as e:
            logger.warning("load_config: failed to load bundled defaults: %s", e)

    # Tier 1: Load user overrides from AppData (highest priority)
    user_path = _get_appdata_config_path()
    if user_path.exists():
        try:
            with open(user_path, "rb") as f:
                user_overrides = tomllib.load(f)
            config = _deep_merge(config, user_overrides)
            logger.info("load_config: loaded user overrides from %s", user_path)
        except Exception as e:
            logger.warning("load_config: failed to load user overrides (using defaults): %s", e)
    else:
        # Bootstrap: persist merged config if no user settings exist yet
        try:
            from localwispr.config.saver import save_config
            save_config(config, user_path)
            logger.info("load_config: bootstrapped user-settings.toml at %s", user_path)
        except Exception as e:
            logger.warning("load_config: bootstrap save failed (non-fatal): %s", e)

    return config
