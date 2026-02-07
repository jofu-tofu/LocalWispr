"""Settings view protocol for LocalWispr.

Defines the interface that any GUI implementation must provide.
This allows the settings controller to work with any frontend
(Tkinter, PyQt, web, etc.) without modification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from localwispr.settings.model import SettingsSnapshot, ValidationResult


class SettingsViewProtocol(Protocol):
    """Interface that any settings GUI must implement.

    The controller interacts with the view only through this interface,
    making it easy to swap GUI frameworks without changing business logic.

    Callbacks:
        on_save_requested: Called when user clicks Save button.
        on_cancel_requested: Called when user clicks Cancel button.
        on_setting_changed: Called when any setting is modified (field, value).

    Example implementation:
        >>> class TkinterSettingsView:
        ...     def __init__(self):
        ...         self.on_save_requested = None
        ...         self.on_cancel_requested = None
        ...         self.on_setting_changed = None
        ...
        ...     def populate(self, settings: SettingsSnapshot) -> None:
        ...         # Fill UI widgets with settings values
        ...         ...
        ...
        ...     def collect(self) -> SettingsSnapshot:
        ...         # Read current UI state into a snapshot
        ...         ...
    """

    # Callbacks set by controller
    on_save_requested: Callable[[], None] | None
    on_cancel_requested: Callable[[], None] | None
    on_setting_changed: Callable[[str, Any], None] | None

    def populate(self, settings: "SettingsSnapshot") -> None:
        """Populate the view with settings values.

        Called by controller when opening settings or after a successful save.

        Args:
            settings: Settings snapshot to display.
        """
        ...

    def collect(self) -> "SettingsSnapshot":
        """Collect current UI state into a settings snapshot.

        Called by controller before validation and save.

        Returns:
            SettingsSnapshot reflecting current UI values.
        """
        ...

    def set_dirty(self, is_dirty: bool) -> None:
        """Update the dirty state indicator.

        When dirty:
        - Window title should show "*" suffix
        - Save button should be enabled
        - Cancel should prompt for confirmation

        Args:
            is_dirty: True if there are unsaved changes.
        """
        ...

    def show_validation_errors(self, result: "ValidationResult") -> None:
        """Display validation errors to the user.

        Called when save is blocked due to invalid settings.
        Should highlight the invalid fields and show error messages.

        Args:
            result: Validation result with errors dict.
        """
        ...

    def confirm_discard_changes(self) -> bool:
        """Ask user to confirm discarding unsaved changes.

        Called when user tries to close with unsaved changes.

        Returns:
            True if user confirms discard, False to cancel close.
        """
        ...

    def close(self) -> None:
        """Close the settings window.

        Called after successful save or confirmed discard.
        Should clean up resources and destroy the window.
        """
        ...

    def show(self) -> None:
        """Show the settings window.

        Called by controller after initial population.
        Should display the window and start the event loop.
        """
        ...
