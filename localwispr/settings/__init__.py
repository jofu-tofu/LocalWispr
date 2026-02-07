"""Settings MVC system for LocalWispr."""

from localwispr.settings.model import (
    SettingsSnapshot,
    SettingsValidator,
    ValidationResult,
    VALID_COMPUTE_TYPES,
    VALID_DEVICES,
    VALID_HOTKEY_MODES,
    VALID_LANGUAGES,
    VALID_MODEL_NAMES,
    VALID_MODIFIERS,
)
from localwispr.settings.view import SettingsViewProtocol
from localwispr.settings.controller import SettingsController
from localwispr.settings.manager import (
    InvalidationFlags,
    SettingsManager,
    SETTINGS_INVALIDATION,
    get_settings_manager,
)
from localwispr.settings.components import (
    create_button_row,
    create_checkbox,
    create_combobox,
    create_info_label,
    create_labeled_frame,
    create_radio_group,
    create_spinbox_row,
)

__all__ = [
    # Model
    "SettingsSnapshot",
    "SettingsValidator",
    "ValidationResult",
    "VALID_COMPUTE_TYPES",
    "VALID_DEVICES",
    "VALID_HOTKEY_MODES",
    "VALID_LANGUAGES",
    "VALID_MODEL_NAMES",
    "VALID_MODIFIERS",
    # View
    "SettingsViewProtocol",
    # Controller
    "SettingsController",
    # Manager
    "InvalidationFlags",
    "SettingsManager",
    "SETTINGS_INVALIDATION",
    "get_settings_manager",
    # Components
    "create_button_row",
    "create_checkbox",
    "create_combobox",
    "create_info_label",
    "create_labeled_frame",
    "create_radio_group",
    "create_spinbox_row",
]
