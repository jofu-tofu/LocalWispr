"""UI components: tray, overlay, notifications, first-run wizard."""

from localwispr.ui.tray import TrayApp, TrayState, STATE_COLORS, create_icon_image
from localwispr.ui.overlay import OverlayWidget, OverlayState
from localwispr.ui.notifications import (
    show_notification,
    show_recording_started,
    show_transcribing,
    show_complete,
    show_error,
    show_clipboard_only,
)
from localwispr.ui.first_run import (
    FirstRunWizard,
    is_first_run,
    show_first_run_wizard,
    ensure_model_available,
)

__all__ = [
    # tray
    "TrayApp",
    "TrayState",
    "STATE_COLORS",
    "create_icon_image",
    # overlay
    "OverlayWidget",
    "OverlayState",
    # notifications
    "show_notification",
    "show_recording_started",
    "show_transcribing",
    "show_complete",
    "show_error",
    "show_clipboard_only",
    # first_run
    "FirstRunWizard",
    "is_first_run",
    "show_first_run_wizard",
    "ensure_model_available",
]
