"""Settings controller for LocalWispr.

Orchestrates the settings flow: load -> populate -> edit -> validate -> save.
Decoupled from any specific GUI framework.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Callable

from localwispr.settings_model import SettingsSnapshot, SettingsValidator

if TYPE_CHECKING:
    from localwispr.settings_view import SettingsViewProtocol

logger = logging.getLogger(__name__)


class SettingsController:
    """Orchestrates settings flow - decoupled from GUI.

    The controller manages:
    - Loading settings from disk
    - Tracking dirty state (unsaved changes)
    - Validation before save
    - Saving and applying settings via SettingsManager

    The view (GUI) is accessed only through the SettingsViewProtocol,
    making it easy to swap GUI frameworks without changing this code.

    Example:
        >>> view = TkinterSettingsView()
        >>> controller = SettingsController(view)
        >>> controller.open()  # Load, populate, show
    """

    def __init__(
        self,
        view: "SettingsViewProtocol",
        on_settings_applied: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the settings controller.

        Args:
            view: GUI implementation of SettingsViewProtocol.
            on_settings_applied: Optional callback after settings are saved and applied.
        """
        self._view = view
        self._on_settings_applied = on_settings_applied

        # State tracking
        self._original: SettingsSnapshot | None = None  # Disk state at open time
        self._current: SettingsSnapshot | None = None   # Working copy

        # Wire up view callbacks
        self._view.on_save_requested = self._handle_save
        self._view.on_cancel_requested = self._handle_cancel
        self._view.on_setting_changed = self._handle_setting_changed

    @property
    def is_dirty(self) -> bool:
        """Check if there are unsaved changes.

        Returns:
            True if current settings differ from original.
        """
        if self._original is None or self._current is None:
            return False
        return self._original != self._current

    def open(self) -> None:
        """Open the settings window.

        Loads current config from disk, populates the view, and shows it.
        """
        from localwispr.config import load_config

        logger.info("settings_controller: opening settings")

        # Load from disk
        config = load_config()
        self._original = SettingsSnapshot.from_config_dict(config)
        self._current = self._original  # Start with same values

        logger.debug(
            "settings_controller: loaded snapshot, model=%s, mode=%s",
            self._original.model_name,
            self._original.hotkey_mode,
        )

        # Populate and show view
        self._view.populate(self._original)
        self._view.set_dirty(False)
        self._view.show()

    def _handle_save(self) -> None:
        """Handle Save button click.

        Flow:
        1. Collect current UI state
        2. Validate all settings
        3. If invalid: show errors and abort
        4. Save to disk
        5. Apply settings via SettingsManager
        6. Update original to match (no longer dirty)
        """
        from localwispr.config import reload_config, save_config
        from localwispr.settings_manager import get_settings_manager

        logger.info("settings_controller: save requested")

        # Collect current UI state
        try:
            self._current = self._view.collect()
        except Exception as e:
            logger.error("settings_controller: collect failed: %s", e)
            return

        # Validate
        result = SettingsValidator.validate(self._current)
        if not result.is_valid:
            logger.warning(
                "settings_controller: validation failed, errors=%s",
                list(result.errors.keys()),
            )
            self._view.show_validation_errors(result)
            return

        # Save to disk
        try:
            config_dict = self._current.to_config_dict()
            save_config(config_dict)
            logger.info("settings_controller: config saved to disk")
        except Exception as e:
            logger.error("settings_controller: save failed: %s", e)
            # Show save error via validation errors
            from localwispr.settings_model import ValidationResult
            error_result = ValidationResult()
            error_result.add_error("_save", f"Failed to save: {e}")
            self._view.show_validation_errors(error_result)
            return

        # Reload to refresh cache
        try:
            reload_config()
        except Exception as e:
            logger.error("settings_controller: reload failed: %s", e)

        # Apply settings via manager (dispatches handlers)
        old_config = self._original.to_config_dict() if self._original else {}
        new_config = self._current.to_config_dict()

        try:
            manager = get_settings_manager()
            manager.apply_settings(old_config, new_config)
            logger.info("settings_controller: settings applied via manager")
        except Exception as e:
            logger.error("settings_controller: apply_settings failed: %s", e)

        # Update original (no longer dirty)
        self._original = self._current
        self._view.set_dirty(False)

        # Invoke callback
        if self._on_settings_applied is not None:
            try:
                self._on_settings_applied()
            except Exception as e:
                logger.error("settings_controller: on_settings_applied callback failed: %s", e)

        logger.info("settings_controller: save complete")

    def _handle_cancel(self) -> None:
        """Handle Cancel button click or window close.

        If dirty, prompts for confirmation before closing.
        """
        logger.debug("settings_controller: cancel requested, is_dirty=%s", self.is_dirty)

        # Collect current state to check dirty
        try:
            self._current = self._view.collect()
        except Exception:
            pass  # If collect fails, assume not dirty

        if self.is_dirty:
            if not self._view.confirm_discard_changes():
                logger.debug("settings_controller: discard cancelled by user")
                return

        logger.info("settings_controller: closing settings")
        self._view.close()

    def _handle_setting_changed(self, field: str, value: object) -> None:
        """Handle individual setting change.

        Updates dirty state based on whether current differs from original.

        Args:
            field: Name of changed field.
            value: New value (for logging, not used directly).
        """
        # Collect current state from view
        try:
            self._current = self._view.collect()
        except Exception:
            return  # View not ready

        # Update dirty indicator
        is_dirty = self.is_dirty
        self._view.set_dirty(is_dirty)

        logger.debug(
            "settings_controller: setting changed, field=%s, is_dirty=%s",
            field,
            is_dirty,
        )
