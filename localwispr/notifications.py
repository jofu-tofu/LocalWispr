"""Notification module for LocalWispr.

This module provides toast/system notification support using winotify for:
- Recording state feedback (Recording..., Transcribing..., Done)
- Error notifications
- Fast, native Windows 10+ toast notifications

Privacy Note:
    Notifications NEVER show transcription content. Only generic status
    messages are displayed to prevent shoulder-surfing.
"""

from __future__ import annotations

import logging

from winotify import Notification, audio

# Configure module logger
logger = logging.getLogger(__name__)

# Application identifier for notifications
APP_ID = "LocalWispr"


def show_notification(
    title: str,
    message: str,
    timeout: int = 5,
) -> bool:
    """Show a Windows toast notification.

    Args:
        title: Notification title.
        message: Notification message body.
        timeout: Display duration hint (5 or less = "short", more = "long").

    Returns:
        True if notification was shown, False otherwise.
    """
    try:
        toast = Notification(
            app_id=APP_ID,
            title=title,
            msg=message,
            duration="short" if timeout <= 5 else "long",
        )
        toast.set_audio(audio.Default, loop=False)
        toast.show()
        logger.debug("notification: shown, title=%s", title)
        return True
    except Exception as e:
        logger.warning(
            "notification: failed, error_type=%s",
            type(e).__name__,
        )
        return False


def show_recording_started() -> bool:
    """Show "Recording..." notification.

    Returns:
        True if notification was shown, False otherwise.
    """
    return show_notification(
        title="LocalWispr",
        message="Recording...",
        timeout=30,  # Long duration - will be replaced by next notification
    )


def show_transcribing() -> bool:
    """Show "Transcribing..." notification.

    Returns:
        True if notification was shown, False otherwise.
    """
    return show_notification(
        title="LocalWispr",
        message="Transcribing...",
        timeout=30,  # Long duration - will be replaced by completion
    )


def show_complete() -> bool:
    """Show completion notification.

    Privacy Note:
        This notification does NOT show the transcribed text.
        It only indicates that transcription is complete.

    Returns:
        True if notification was shown, False otherwise.
    """
    return show_notification(
        title="LocalWispr",
        message="Done - text pasted",
        timeout=3,
    )


def show_error(message: str) -> bool:
    """Show error notification.

    Args:
        message: Error description (will be sanitized).

    Returns:
        True if notification was shown, False otherwise.
    """
    # Sanitize message - remove any potential sensitive data
    # Only keep first 100 chars and remove newlines
    safe_message = message[:100].replace("\n", " ").replace("\r", "")

    return show_notification(
        title="LocalWispr - Error",
        message=safe_message,
        timeout=5,
    )


def show_clipboard_only() -> bool:
    """Show notification for clipboard-only mode.

    Returns:
        True if notification was shown, False otherwise.
    """
    return show_notification(
        title="LocalWispr",
        message="Done - text copied to clipboard",
        timeout=3,
    )
