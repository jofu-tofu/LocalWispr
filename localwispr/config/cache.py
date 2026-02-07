"""Thread-safe configuration cache for LocalWispr."""

import logging
import threading

from localwispr.config.types import Config

logger = logging.getLogger(__name__)

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
            from localwispr.config.loader import load_config

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
            from localwispr.config.loader import load_config

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
