"""Settings manager for centralized settings change handling.

Provides a single point for:
- Declaring what needs to be invalidated when each setting changes
- Registering invalidation handlers
- Applying settings changes with automatic handler invocation

This architecture makes it easy to add new settings - just add the
setting path to SETTINGS_INVALIDATION with the appropriate flags.
"""

from __future__ import annotations

import logging
from enum import Flag, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)


class InvalidationFlags(Flag):
    """What needs to be invalidated when settings change."""

    NONE = 0
    HOTKEY_LISTENER = auto()  # Mode, modifiers changed
    TRANSCRIBER = auto()  # Model, vocabulary, language changed
    MODEL_PRELOAD = auto()  # Model needs reload


# Setting paths mapped to their invalidation requirements
# When adding a new setting, add it here with the appropriate flags
SETTINGS_INVALIDATION: dict[str, InvalidationFlags] = {
    # Hotkey settings
    "hotkeys.mode": InvalidationFlags.HOTKEY_LISTENER,
    "hotkeys.modifiers": InvalidationFlags.HOTKEY_LISTENER,
    "hotkeys.audio_feedback": InvalidationFlags.NONE,
    "hotkeys.mute_system": InvalidationFlags.NONE,
    # Model settings
    "model.name": InvalidationFlags.TRANSCRIBER | InvalidationFlags.MODEL_PRELOAD,
    "model.device": InvalidationFlags.TRANSCRIBER | InvalidationFlags.MODEL_PRELOAD,
    "model.compute_type": InvalidationFlags.TRANSCRIBER | InvalidationFlags.MODEL_PRELOAD,
    "model.language": InvalidationFlags.TRANSCRIBER,
    # Vocabulary
    "vocabulary.words": InvalidationFlags.TRANSCRIBER,
    # Output (read on each use, no invalidation needed)
    "output.auto_paste": InvalidationFlags.NONE,
    "output.paste_delay_ms": InvalidationFlags.NONE,
    # Streaming
    "streaming.enabled": InvalidationFlags.NONE,
    "streaming.min_silence_ms": InvalidationFlags.NONE,
    "streaming.max_segment_duration": InvalidationFlags.NONE,
    "streaming.min_segment_duration": InvalidationFlags.NONE,
    "streaming.overlap_ms": InvalidationFlags.NONE,
}


class SettingsManager:
    """Central settings management with change tracking.

    Allows components to register handlers that are called when settings
    they depend on change. This decouples the settings UI from the
    components that need to respond to settings changes.

    Example:
        >>> manager = get_settings_manager()
        >>> manager.register_handler(
        ...     InvalidationFlags.TRANSCRIBER,
        ...     lambda: print("transcriber needs reload")
        ... )
        >>> manager.apply_settings(old_config, new_config)
    """

    def __init__(self) -> None:
        """Initialize the settings manager."""
        self._handlers: dict[InvalidationFlags, list[Callable[[], None]]] = {}

    def register_handler(
        self,
        flags: InvalidationFlags,
        handler: Callable[[], None],
    ) -> None:
        """Register a handler for settings with given invalidation flags.

        The handler will be called when any setting with matching flags changes.

        Args:
            flags: Invalidation flags to respond to.
            handler: Callable to invoke when matching settings change.
        """
        for flag in InvalidationFlags:
            if flag in flags and flag != InvalidationFlags.NONE:
                self._handlers.setdefault(flag, []).append(handler)
                logger.debug(
                    "settings_manager: registered handler for %s",
                    flag.name,
                )

    def apply_settings(
        self,
        old_config: dict[str, Any],
        new_config: dict[str, Any],
    ) -> None:
        """Compare configs and call handlers for changed settings.

        Args:
            old_config: Previous configuration dictionary.
            new_config: New configuration dictionary.
        """
        changed_flags = InvalidationFlags.NONE

        for path, flags in SETTINGS_INVALIDATION.items():
            old_val = self._get_nested(old_config, path)
            new_val = self._get_nested(new_config, path)
            if old_val != new_val:
                logger.debug(
                    "settings_manager: %s changed: %s -> %s",
                    path,
                    old_val,
                    new_val,
                )
                changed_flags |= flags

        # Call handlers for changed flags
        for flag in InvalidationFlags:
            if flag in changed_flags and flag in self._handlers:
                for handler in self._handlers[flag]:
                    try:
                        handler()
                    except Exception as e:
                        logger.error(
                            "settings_manager: handler failed for %s: %s",
                            flag.name,
                            e,
                        )

    @staticmethod
    def _get_nested(config: dict[str, Any], path: str) -> Any:
        """Get nested config value by dot-separated path.

        Args:
            config: Configuration dictionary.
            path: Dot-separated path (e.g., "model.name").

        Returns:
            Value at path, or None if path doesn't exist.
        """
        keys = path.split(".")
        val: Any = config
        for key in keys:
            if isinstance(val, dict):
                val = val.get(key)
            else:
                return None
        return val


# Global instance (lazy initialization)
_settings_manager: SettingsManager | None = None


def get_settings_manager() -> SettingsManager:
    """Get the global SettingsManager instance.

    Returns:
        The singleton SettingsManager instance.
    """
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager
