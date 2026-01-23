"""Mode manager for LocalWispr.

This module contains the ModeManager class that handles:
- Mode persistence across recordings
- Manual mode override with optional auto-reset
- Mode cycling via hotkey
- Thread-safe mode access

Also includes convenience functions for accessing the global mode manager.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from localwispr.modes.definitions import MODES, MODE_CYCLE_ORDER, Mode, ModeType

logger = logging.getLogger(__name__)


class ModeManager:
    """Manages the current transcription mode.

    Provides:
    - Mode persistence across recordings
    - Manual mode override with optional auto-reset
    - Mode cycling via hotkey
    - Thread-safe mode access

    The manager supports both automatic context detection (from window titles)
    and manual mode selection. Manual mode takes precedence until reset.
    """

    # Auto-reset delay in seconds (5 minutes of inactivity)
    AUTO_RESET_DELAY = 300.0

    def __init__(
        self,
        auto_reset: bool = False,
        on_mode_change: Callable[[Mode], None] | None = None,
    ) -> None:
        """Initialize the mode manager.

        Args:
            auto_reset: If True, reset to auto-detection after inactivity.
            on_mode_change: Callback when mode changes.
        """
        self._current_mode = ModeType.DICTATION  # Default mode
        self._manual_override = False
        self._auto_reset = auto_reset
        self._last_activity_time = time.time()
        self._on_mode_change = on_mode_change
        self._lock = threading.Lock()

        logger.info("mode_manager: initialized, default=%s", self._current_mode.value)

    @property
    def current_mode(self) -> Mode:
        """Get the current mode.

        Returns:
            The current Mode object.
        """
        with self._lock:
            self._check_auto_reset()
            return MODES[self._current_mode]

    @property
    def current_mode_type(self) -> ModeType:
        """Get the current mode type.

        Returns:
            The current ModeType enum value.
        """
        with self._lock:
            self._check_auto_reset()
            return self._current_mode

    @property
    def is_manual_override(self) -> bool:
        """Check if manual mode override is active.

        Returns:
            True if manual override is active.
        """
        with self._lock:
            return self._manual_override

    def _check_auto_reset(self) -> None:
        """Check and perform auto-reset if enabled and time elapsed.

        Must be called with lock held.
        """
        if not self._auto_reset or not self._manual_override:
            return

        elapsed = time.time() - self._last_activity_time
        if elapsed >= self.AUTO_RESET_DELAY:
            logger.info("mode_manager: auto-reset after %.1fs inactivity", elapsed)
            self._manual_override = False
            self._current_mode = ModeType.DICTATION

    def set_mode(self, mode_type: ModeType) -> Mode:
        """Set the current mode (manual override).

        Args:
            mode_type: The mode to set.

        Returns:
            The new Mode object.
        """
        with self._lock:
            old_mode = self._current_mode
            self._current_mode = mode_type
            self._manual_override = True
            self._last_activity_time = time.time()

        logger.info(
            "mode_manager: set mode=%s (was=%s, manual=True)",
            mode_type.value,
            old_mode.value,
        )

        new_mode = MODES[mode_type]

        if self._on_mode_change is not None and old_mode != mode_type:
            try:
                self._on_mode_change(new_mode)
            except Exception:
                pass  # Don't let callback errors propagate

        return new_mode

    def cycle_mode(self) -> Mode:
        """Cycle to the next mode in the sequence.

        Returns:
            The new Mode object.
        """
        with self._lock:
            current_index = MODE_CYCLE_ORDER.index(self._current_mode)
            next_index = (current_index + 1) % len(MODE_CYCLE_ORDER)
            new_mode_type = MODE_CYCLE_ORDER[next_index]

        return self.set_mode(new_mode_type)

    def reset_override(self) -> Mode:
        """Reset manual override and return to auto-detection mode.

        Returns:
            The current Mode after reset.
        """
        with self._lock:
            self._manual_override = False
            self._last_activity_time = time.time()

        logger.info("mode_manager: manual override reset")

        return self.current_mode

    def record_activity(self) -> None:
        """Record user activity to reset auto-reset timer.

        Call this when the user performs any action (recording, etc.)
        to prevent auto-reset.
        """
        with self._lock:
            self._last_activity_time = time.time()

    def get_prompt(self) -> str:
        """Get the prompt text for the current mode.

        Returns:
            The prompt text for Whisper's initial_prompt.
        """
        return self.current_mode.load_prompt()


# Global mode manager instance (lazy initialized)
_mode_manager: ModeManager | None = None
_mode_manager_lock = threading.Lock()


def get_mode_manager(
    auto_reset: bool = False,
    on_mode_change: Callable[[Mode], None] | None = None,
) -> ModeManager:
    """Get or create the global mode manager.

    Args:
        auto_reset: If True, reset to auto-detection after inactivity.
        on_mode_change: Callback when mode changes (only used on creation).

    Returns:
        The global ModeManager instance.
    """
    global _mode_manager

    with _mode_manager_lock:
        if _mode_manager is None:
            _mode_manager = ModeManager(
                auto_reset=auto_reset,
                on_mode_change=on_mode_change,
            )
        return _mode_manager


def get_current_mode() -> Mode:
    """Get the current transcription mode.

    Convenience function that uses the global mode manager.

    Returns:
        The current Mode object.
    """
    return get_mode_manager().current_mode


def set_mode(mode_type: ModeType) -> Mode:
    """Set the current transcription mode.

    Convenience function that uses the global mode manager.

    Args:
        mode_type: The mode to set.

    Returns:
        The new Mode object.
    """
    return get_mode_manager().set_mode(mode_type)


def cycle_mode() -> Mode:
    """Cycle to the next transcription mode.

    Convenience function that uses the global mode manager.

    Returns:
        The new Mode object.
    """
    return get_mode_manager().cycle_mode()


def get_mode_prompt() -> str:
    """Get the prompt text for the current mode.

    Convenience function that uses the global mode manager.

    Returns:
        The prompt text for Whisper's initial_prompt.
    """
    return get_mode_manager().get_prompt()


def get_all_modes() -> list[Mode]:
    """Get all available modes in cycle order.

    Returns:
        List of Mode objects in the cycle order.
    """
    return [MODES[mt] for mt in MODE_CYCLE_ORDER]


def get_mode_by_name(name: str) -> Mode | None:
    """Get a mode by its display name (case-insensitive).

    Args:
        name: The display name to search for.

    Returns:
        The Mode object if found, None otherwise.
    """
    name_lower = name.lower()
    for mode in MODES.values():
        if mode.name.lower() == name_lower:
            return mode
    return None


__all__ = [
    "ModeManager",
    "get_mode_manager",
    "get_current_mode",
    "set_mode",
    "cycle_mode",
    "get_mode_prompt",
    "get_all_modes",
    "get_mode_by_name",
]
