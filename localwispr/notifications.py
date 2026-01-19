"""Notification module for LocalWispr.

This module provides toast/system notification support for:
- Recording state feedback (Recording..., Transcribing..., Done)
- Error notifications
- Graceful fallback when notifications unavailable

Privacy Note:
    Notifications NEVER show transcription content. Only generic status
    messages are displayed to prevent shoulder-surfing.
"""

from __future__ import annotations

import logging

# Configure module logger
logger = logging.getLogger(__name__)

# Track if we've already warned about notification issues
_notification_warned = False


def _check_notification_support() -> bool:
    """Check if notification support is available.

    Returns:
        True if notifications are supported, False otherwise.
    """
    global _notification_warned

    try:
        from plyer import notification

        # Try to access the notification module to verify it's working
        _ = notification.notify
        return True
    except ImportError:
        if not _notification_warned:
            logger.warning("notifications: plyer not available")
            _notification_warned = True
        return False
    except Exception as e:
        if not _notification_warned:
            logger.warning(
                "notifications: initialization failed, error_type=%s",
                type(e).__name__,
            )
            _notification_warned = True
        return False


def show_notification(
    title: str,
    message: str,
    timeout: int = 5,
    app_name: str = "LocalWispr",
) -> bool:
    """Show a toast notification.

    Args:
        title: Notification title.
        message: Notification message body.
        timeout: Display duration in seconds.
        app_name: Application name for notification.

    Returns:
        True if notification was shown, False otherwise.
    """
    if not _check_notification_support():
        return False

    try:
        from plyer import notification

        notification.notify(
            title=title,
            message=message,
            app_name=app_name,
            timeout=timeout,
        )
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
        timeout=30,  # Long timeout - will be replaced by next notification
    )


def show_transcribing() -> bool:
    """Show "Transcribing..." notification.

    Returns:
        True if notification was shown, False otherwise.
    """
    return show_notification(
        title="LocalWispr",
        message="Transcribing...",
        timeout=30,  # Long timeout - will be replaced by completion
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
