"""Output module for LocalWispr.

This module handles transcription output via clipboard and auto-paste:
- Copy text to system clipboard with retry logic
- Simulate Ctrl+V to paste to active window
- Configurable auto-paste vs clipboard-only mode
- Audio feedback before paste to alert user

Privacy Note:
    This module NEVER logs transcription content. Only operation status
    (success/fail) and duration metrics are logged.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import pyperclip
from pynput.keyboard import Controller, Key

if TYPE_CHECKING:
    pass

# Configure module logger
logger = logging.getLogger(__name__)

# Retry configuration
MAX_CLIPBOARD_RETRIES = 3
CLIPBOARD_RETRY_DELAY_MS = 100

# Paste simulation configuration
DEFAULT_PASTE_DELAY_MS = 50

# Lock for serializing paste operations to prevent concurrent pynput Controller calls
_paste_lock = threading.Lock()


class OutputError(Exception):
    """Exception raised when output operations fail."""

    pass


def copy_to_clipboard(text: str, max_retries: int = MAX_CLIPBOARD_RETRIES) -> bool:
    """Copy text to the system clipboard.

    Includes retry logic to handle cases where the clipboard is locked
    by another application.

    Privacy Note:
        This function does NOT log the text content. Only operation status
        (success/fail) is logged.

    Args:
        text: Text to copy to clipboard.
        max_retries: Maximum number of retry attempts.

    Returns:
        True if copy succeeded, False otherwise.
    """
    for attempt in range(max_retries):
        try:
            pyperclip.copy(text)
            logger.debug("clipboard_copy: success, attempt=%d", attempt + 1)
            return True
        except pyperclip.PyperclipException as e:
            logger.warning(
                "clipboard_copy: failed, attempt=%d, error_type=%s",
                attempt + 1,
                type(e).__name__,
            )
            if attempt < max_retries - 1:
                time.sleep(CLIPBOARD_RETRY_DELAY_MS / 1000.0)

    logger.error("clipboard_copy: all retries exhausted")
    return False


def paste_to_active_window(delay_ms: int = DEFAULT_PASTE_DELAY_MS) -> bool:
    """Simulate Ctrl+V to paste clipboard contents to the active window.

    Uses pynput to simulate the keyboard shortcut. A small delay is added
    before the paste to ensure the active window is ready to receive input.

    Thread Safety:
        Uses a module-level lock to serialize paste operations, preventing
        concurrent pynput Controller calls which can cause conflicts.

    Args:
        delay_ms: Delay in milliseconds before paste (default: 50ms).

    Returns:
        True if paste simulation succeeded, False otherwise.
    """
    with _paste_lock:
        try:
            logger.debug("paste_simulation: attempting with delay_ms=%d", delay_ms)

            # Small delay to ensure window focus is stable
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

            keyboard = Controller()

            # Clear any stuck modifier keys before pasting
            # This prevents conflicts with hotkey chord keys (Win+Ctrl+Shift)
            logger.debug("paste_simulation: clearing stuck modifier keys")
            for key in (Key.ctrl, Key.ctrl_l, Key.ctrl_r,
                        Key.shift, Key.shift_l, Key.shift_r,
                        Key.alt, Key.alt_l, Key.alt_r,
                        Key.cmd, Key.cmd_l, Key.cmd_r):
                try:
                    keyboard.release(key)
                except Exception:
                    pass  # Key may not be pressed, ignore

            # Delay for keyboard state to settle after releasing modifiers
            # Increased from 10ms to 100ms to ensure Windows clears modifier state
            logger.debug("paste_simulation: waiting for keyboard state to settle")
            time.sleep(0.1)  # 100ms

            # Simulate Ctrl+V on a clean keyboard state
            logger.debug("paste_simulation: simulating ctrl+v")
            keyboard.press(Key.ctrl)
            keyboard.press("v")
            keyboard.release("v")
            keyboard.release(Key.ctrl)

            logger.debug("paste_simulation: success")
            return True

        except Exception as e:
            logger.error(
                "paste_simulation: failed, error=%s, error_type=%s",
                str(e),
                type(e).__name__,
            )
            return False


def output_transcription(
    text: str,
    auto_paste: bool = True,
    paste_delay_ms: int = DEFAULT_PASTE_DELAY_MS,
    play_feedback: bool = True,
) -> bool:
    """Output transcribed text via clipboard and optional auto-paste.

    This is the main output function that:
    1. Copies text to clipboard (always, as backup)
    2. Optionally plays audio feedback before paste
    3. Optionally simulates Ctrl+V to paste

    Privacy Note:
        This function does NOT log the transcription content.

    Args:
        text: Transcribed text to output.
        auto_paste: Whether to auto-paste after copying (default: True).
        paste_delay_ms: Delay before paste in milliseconds.
        play_feedback: Whether to play audio feedback before paste.

    Returns:
        True if output succeeded (clipboard copy + paste if enabled).
    """
    # Always copy to clipboard first
    if not copy_to_clipboard(text):
        logger.error("output_transcription: clipboard copy failed")
        return False

    if not auto_paste:
        logger.debug("output_transcription: clipboard-only mode")
        return True

    # Play audio feedback before paste to alert user
    if play_feedback:
        try:
            from localwispr.feedback import play_stop_beep

            play_stop_beep()
        except Exception:
            # Don't fail output if feedback fails
            logger.warning("output_transcription: audio feedback failed")

    # Simulate paste
    if not paste_to_active_window(paste_delay_ms):
        logger.warning("output_transcription: paste failed, text is in clipboard")
        return False

    logger.debug("output_transcription: success")
    return True
