"""Audio feedback module for LocalWispr.

Provides non-blocking audio beeps for recording state changes.
Uses winsound on Windows, falls back to silent no-op on other platforms.
"""

from __future__ import annotations

import platform
import threading

# Check if we're on Windows for winsound support
_IS_WINDOWS = platform.system() == "Windows"
_NON_WINDOWS_WARNING_SHOWN = False


def _play_beep_windows(frequency: int, duration_ms: int) -> None:
    """Play a beep on Windows using winsound.

    Args:
        frequency: Frequency in Hz.
        duration_ms: Duration in milliseconds.
    """
    import winsound

    winsound.Beep(frequency, duration_ms)


def _play_beep_async(frequency: int, duration_ms: int) -> None:
    """Play a beep asynchronously in a daemon thread.

    Args:
        frequency: Frequency in Hz.
        duration_ms: Duration in milliseconds.
    """
    global _NON_WINDOWS_WARNING_SHOWN

    if _IS_WINDOWS:
        thread = threading.Thread(
            target=_play_beep_windows,
            args=(frequency, duration_ms),
            daemon=True,
        )
        thread.start()
    else:
        # Non-Windows: print warning once, then silent
        if not _NON_WINDOWS_WARNING_SHOWN:
            print("Note: Audio feedback not available on this platform (Windows only)")
            _NON_WINDOWS_WARNING_SHOWN = True


def play_start_beep() -> None:
    """Play a high-pitched beep to indicate recording start.

    800Hz, 100ms - a quick, higher-pitched tone.
    Non-blocking: plays in a daemon thread.
    """
    _play_beep_async(800, 100)


def play_stop_beep() -> None:
    """Play a lower-pitched beep to indicate recording stop.

    400Hz, 100ms - a quick, lower-pitched tone.
    Non-blocking: plays in a daemon thread.
    """
    _play_beep_async(400, 100)
