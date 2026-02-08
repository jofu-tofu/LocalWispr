"""Tests for the SettingsManager module.

Tests the centralized settings change handling, including:
- Invalidation flag detection
- Handler registration and invocation
- Nested config value extraction
"""

from __future__ import annotations

from unittest.mock import MagicMock


from localwispr.settings.manager import (
    InvalidationFlags,
    SettingsManager,
    SETTINGS_INVALIDATION,
    get_settings_manager,
)


class TestSettingsInvalidation:
    """Tests for the SETTINGS_INVALIDATION mapping."""

    def test_model_settings_require_transcriber_invalidation(self):
        """Test that model settings trigger transcriber invalidation."""
        assert InvalidationFlags.TRANSCRIBER in SETTINGS_INVALIDATION["model.name"]
        assert InvalidationFlags.TRANSCRIBER in SETTINGS_INVALIDATION["model.device"]
        assert InvalidationFlags.TRANSCRIBER in SETTINGS_INVALIDATION["model.language"]

    def test_model_settings_require_preload(self):
        """Test that model name/device changes require model preload."""
        assert InvalidationFlags.MODEL_PRELOAD in SETTINGS_INVALIDATION["model.name"]
        assert InvalidationFlags.MODEL_PRELOAD in SETTINGS_INVALIDATION["model.device"]
        # Language change doesn't require full model reload
        assert InvalidationFlags.MODEL_PRELOAD not in SETTINGS_INVALIDATION["model.language"]

    def test_hotkey_settings_require_listener_restart(self):
        """Test that hotkey settings trigger listener restart."""
        assert InvalidationFlags.HOTKEY_LISTENER in SETTINGS_INVALIDATION["hotkeys.mode"]
        assert InvalidationFlags.HOTKEY_LISTENER in SETTINGS_INVALIDATION["hotkeys.modifiers"]

    def test_output_settings_no_invalidation(self):
        """Test that output settings don't require invalidation."""
        assert SETTINGS_INVALIDATION["output.auto_paste"] == InvalidationFlags.NONE
        assert SETTINGS_INVALIDATION["output.paste_delay_ms"] == InvalidationFlags.NONE

    def test_vocabulary_triggers_transcriber(self):
        """Test that vocabulary changes trigger transcriber invalidation."""
        assert InvalidationFlags.TRANSCRIBER in SETTINGS_INVALIDATION["vocabulary.words"]


class TestSettingsManager:
    """Tests for the SettingsManager class."""

    def test_register_handler_is_invoked_on_matching_change(self):
        """Test registered handler is invoked when a matching setting changes."""
        manager = SettingsManager()
        called = []
        manager.register_handler(InvalidationFlags.TRANSCRIBER, lambda: called.append(True))

        manager.apply_settings({"model": {"name": "small"}}, {"model": {"name": "large-v3"}})

        assert len(called) == 1

    def test_register_handler_multiple_flags_invoked_for_each(self):
        """Test handler registered for combined flags is invoked for each flag type."""
        manager = SettingsManager()
        called = []
        combined = InvalidationFlags.TRANSCRIBER | InvalidationFlags.MODEL_PRELOAD
        manager.register_handler(combined, lambda: called.append(True))

        # model.name triggers both TRANSCRIBER and MODEL_PRELOAD
        manager.apply_settings({"model": {"name": "small"}}, {"model": {"name": "large-v3"}})

        # Handler should be called (once, since apply_settings deduplicates handlers)
        assert len(called) >= 1

    def test_apply_settings_detects_change(self):
        """Test that apply_settings detects config changes."""
        manager = SettingsManager()
        handler = MagicMock()
        manager.register_handler(InvalidationFlags.TRANSCRIBER, handler)

        old_config = {"model": {"name": "small"}}
        new_config = {"model": {"name": "large-v3"}}

        manager.apply_settings(old_config, new_config)

        handler.assert_called_once()

    def test_apply_settings_no_change(self):
        """Test that apply_settings doesn't call handlers if nothing changed."""
        manager = SettingsManager()
        handler = MagicMock()
        manager.register_handler(InvalidationFlags.TRANSCRIBER, handler)

        old_config = {"model": {"name": "large-v3"}}
        new_config = {"model": {"name": "large-v3"}}

        manager.apply_settings(old_config, new_config)

        handler.assert_not_called()

    def test_apply_settings_calls_correct_handler(self):
        """Test that only the correct handler is called for a change."""
        manager = SettingsManager()
        transcriber_handler = MagicMock()
        hotkey_handler = MagicMock()

        manager.register_handler(InvalidationFlags.TRANSCRIBER, transcriber_handler)
        manager.register_handler(InvalidationFlags.HOTKEY_LISTENER, hotkey_handler)

        # Change model setting (should trigger transcriber, not hotkey)
        old_config = {"model": {"language": "en"}}
        new_config = {"model": {"language": "es"}}

        manager.apply_settings(old_config, new_config)

        transcriber_handler.assert_called_once()
        hotkey_handler.assert_not_called()

    def test_apply_settings_handler_exception(self):
        """Test that handler exceptions don't stop other handlers."""
        manager = SettingsManager()
        bad_handler = MagicMock(side_effect=RuntimeError("test error"))
        good_handler = MagicMock()

        manager.register_handler(InvalidationFlags.TRANSCRIBER, bad_handler)
        manager.register_handler(InvalidationFlags.TRANSCRIBER, good_handler)

        old_config = {"model": {"name": "small"}}
        new_config = {"model": {"name": "large-v3"}}

        # Should not raise, should still call good_handler
        manager.apply_settings(old_config, new_config)

        bad_handler.assert_called_once()
        good_handler.assert_called_once()

    def test_get_nested_simple(self):
        """Test _get_nested with simple path."""
        config = {"model": {"name": "large-v3"}}
        result = SettingsManager._get_nested(config, "model.name")
        assert result == "large-v3"

    def test_get_nested_missing(self):
        """Test _get_nested with missing path."""
        config = {"model": {}}
        result = SettingsManager._get_nested(config, "model.name")
        assert result is None

    def test_get_nested_partial_path(self):
        """Test _get_nested with partially missing path."""
        config = {"model": {"name": "large-v3"}}
        result = SettingsManager._get_nested(config, "model.device")
        assert result is None

    def test_get_nested_deep_path(self):
        """Test _get_nested with deeper nesting."""
        config = {"a": {"b": {"c": "value"}}}
        result = SettingsManager._get_nested(config, "a.b.c")
        assert result == "value"


class TestGetSettingsManager:
    """Tests for the get_settings_manager singleton function."""

    def test_returns_same_instance(self):
        """Test that get_settings_manager returns the same instance."""
        manager1 = get_settings_manager()
        manager2 = get_settings_manager()
        assert manager1 is manager2

    def test_returns_settings_manager(self):
        """Test that get_settings_manager returns a SettingsManager."""
        manager = get_settings_manager()
        assert isinstance(manager, SettingsManager)
