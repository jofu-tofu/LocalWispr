"""System volume control for Windows using pycaw.

Provides functions to mute/unmute system audio, primarily for use when
recording system sounds (loopback) to prevent feedback loops.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from pycaw.pycaw import IAudioEndpointVolume

logger = logging.getLogger(__name__)

# Lazy initialization of audio endpoint
_endpoint_volume: "IAudioEndpointVolume | None" = None


def _get_endpoint_volume():
    """Get the default audio endpoint volume interface.

    Lazily initializes the interface on first call.

    Returns:
        IAudioEndpointVolume interface or None if unavailable.
    """
    global _endpoint_volume

    if _endpoint_volume is not None:
        return _endpoint_volume

    try:
        from pycaw.pycaw import AudioUtilities

        device = AudioUtilities.GetSpeakers()
        if device is None:
            logger.warning("No audio output device found")
            return None

        _endpoint_volume = device.EndpointVolume
        return _endpoint_volume

    except ImportError:
        logger.warning("pycaw not available - volume control disabled")
        return None
    except Exception as e:
        logger.warning("Failed to initialize audio endpoint: %s", e)
        return None


def get_mute_state() -> bool:
    """Check if system audio is currently muted.

    Returns:
        True if muted, False if not muted or if unable to determine.
    """
    endpoint = _get_endpoint_volume()
    if endpoint is None:
        return False

    try:
        return bool(endpoint.GetMute())
    except Exception as e:
        logger.warning("Failed to get mute state: %s", e)
        return False


def mute_system() -> bool:
    """Mute system audio.

    Returns:
        Previous mute state (True if was muted, False if was not muted).
        Can be passed to restore_mute_state() to restore original state.
    """
    endpoint = _get_endpoint_volume()
    if endpoint is None:
        return False

    try:
        was_muted = bool(endpoint.GetMute())
        endpoint.SetMute(True, None)
        logger.debug("System audio muted (was_muted=%s)", was_muted)
        return was_muted
    except Exception as e:
        logger.warning("Failed to mute system: %s", e)
        return False


def unmute_system() -> None:
    """Unmute system audio."""
    endpoint = _get_endpoint_volume()
    if endpoint is None:
        return

    try:
        endpoint.SetMute(False, None)
        logger.debug("System audio unmuted")
    except Exception as e:
        logger.warning("Failed to unmute system: %s", e)


def restore_mute_state(was_muted: bool) -> None:
    """Restore mute state to a previous value.

    Args:
        was_muted: The previous mute state to restore (from mute_system()).
    """
    endpoint = _get_endpoint_volume()
    if endpoint is None:
        return

    try:
        endpoint.SetMute(was_muted, None)
        logger.debug("Restored mute state to %s", was_muted)
    except Exception as e:
        logger.warning("Failed to restore mute state: %s", e)


@contextmanager
def system_muted() -> Generator[None, None, None]:
    """Context manager that mutes system audio during execution.

    Automatically restores the previous mute state when exiting,
    even if an exception occurs.

    Usage:
        with system_muted():
            # Recording happens here
            # System is automatically muted
        # System automatically restored to previous state

    Yields:
        None
    """
    was_muted = mute_system()
    try:
        yield
    finally:
        restore_mute_state(was_muted)


__all__ = [
    "get_mute_state",
    "mute_system",
    "unmute_system",
    "restore_mute_state",
    "system_muted",
]
