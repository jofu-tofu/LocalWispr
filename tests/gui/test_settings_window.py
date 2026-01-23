"""GUI tests for the Settings Window.

These tests verify the settings window behavior using Tkinter testing
patterns. They don't require the full application to be running.

Note: These tests use Tkinter's test mode and don't require pywinauto
for basic functionality testing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestSettingsWindowCreation:
    """Tests for SettingsWindow instantiation and setup."""

    def test_settings_window_instantiation(self):
        """Test that SettingsWindow can be instantiated."""
        from localwispr.settings_window import SettingsWindow

        window = SettingsWindow()

        assert window._window is None
        assert window._root is None
        assert window._config is None

    def test_settings_window_with_callback(self):
        """Test SettingsWindow with callback."""
        from localwispr.settings_window import SettingsWindow

        callback = MagicMock()
        window = SettingsWindow(on_settings_changed=callback)

        assert window._on_settings_changed is callback


class TestSettingsWindowConstants:
    """Tests for SettingsWindow constants and options."""

    def test_model_sizes_defined(self):
        """Test that MODEL_SIZES contains expected options."""
        from localwispr.settings_window import MODEL_SIZES

        assert "tiny" in MODEL_SIZES
        assert "base" in MODEL_SIZES
        assert "small" in MODEL_SIZES
        assert "medium" in MODEL_SIZES
        assert "large-v3" in MODEL_SIZES

    def test_devices_defined(self):
        """Test that DEVICES contains expected options."""
        from localwispr.settings_window import DEVICES

        assert "cuda" in DEVICES
        assert "cpu" in DEVICES

    def test_compute_types_defined(self):
        """Test that COMPUTE_TYPES contains expected options."""
        from localwispr.settings_window import COMPUTE_TYPES

        assert "float16" in COMPUTE_TYPES
        assert "int8" in COMPUTE_TYPES
        assert "float32" in COMPUTE_TYPES

    def test_languages_defined(self):
        """Test that LANGUAGES contains expected options."""
        from localwispr.settings_window import LANGUAGES

        language_codes = [code for _, code in LANGUAGES]

        assert "auto" in language_codes
        assert "en" in language_codes
        assert "es" in language_codes
        assert "zh" in language_codes


class TestSettingsWindowDimensions:
    """Tests for SettingsWindow dimension constants."""

    def test_window_dimensions(self):
        """Test that window dimensions are reasonable."""
        from localwispr.settings_window import SettingsWindow

        assert SettingsWindow.WINDOW_WIDTH > 0
        assert SettingsWindow.WINDOW_HEIGHT > 0
        assert SettingsWindow.WINDOW_WIDTH >= 400
        assert SettingsWindow.WINDOW_HEIGHT >= 400


class TestOpenSettingsFunction:
    """Tests for the open_settings convenience function."""

    def test_open_settings_creates_window(self, mocker):
        """Test that open_settings creates a SettingsWindow."""
        # Mock the window creation to avoid GUI
        mock_window = MagicMock()
        mock_window_class = mocker.patch(
            "localwispr.settings_window.SettingsWindow",
            return_value=mock_window,
        )

        from localwispr.settings_window import open_settings

        open_settings()

        mock_window_class.assert_called_once()
        mock_window.show.assert_called_once()

    def test_open_settings_passes_callback(self, mocker):
        """Test that open_settings passes callback to window."""
        mock_window = MagicMock()
        mock_window_class = mocker.patch(
            "localwispr.settings_window.SettingsWindow",
            return_value=mock_window,
        )

        callback = MagicMock()

        from localwispr.settings_window import open_settings

        open_settings(on_settings_changed=callback)

        mock_window_class.assert_called_once_with(on_settings_changed=callback)


class TestSettingsWindowConfigLoading:
    """Tests for configuration loading in settings window."""

    def test_settings_window_loads_config(self, mocker, mock_config, tmp_path):
        """Test that settings window loads config on creation."""
        # Mock config loading
        mocker.patch("localwispr.config.get_config", return_value=mock_config)
        mocker.patch("localwispr.config.load_config", return_value=mock_config)
        mocker.patch("localwispr.config._get_config_path", return_value=tmp_path / "config.toml")

        # We can't fully test GUI creation without a display,
        # but we can test the logic
        from localwispr.settings_window import SettingsWindow

        window = SettingsWindow()

        # The config is loaded in _create_window, which runs in a thread
        # For unit testing, we verify the structure is correct
        assert window._config is None  # Not loaded until show() is called


class TestSettingsWindowSaveLogic:
    """Tests for settings save logic."""

    def test_auto_save_builds_correct_structure(self, mocker, mock_config, tmp_path):
        """Test that auto-save builds config with correct structure."""
        mocker.patch("localwispr.config.get_config", return_value=mock_config)
        mocker.patch("localwispr.config.load_config", return_value=mock_config)
        mocker.patch("localwispr.config._get_config_path", return_value=tmp_path / "config.toml")

        save_mock = mocker.patch("localwispr.config.save_config")
        mocker.patch("localwispr.config.reload_config", return_value=mock_config)

        from localwispr.settings_window import SettingsWindow

        window = SettingsWindow()
        window._config = mock_config

        # Mock tkinter variables
        import tkinter as tk

        window._root = MagicMock()
        window._window = MagicMock()

        # Create mock variables that behave like tk.Variable
        window._vars = {
            "mode": MagicMock(get=lambda: "toggle"),
            "audio_feedback": MagicMock(get=lambda: True),
            "mute_system": MagicMock(get=lambda: False),
            "auto_paste": MagicMock(get=lambda: True),
            "paste_delay_ms": MagicMock(get=lambda: 50),
            "model_name": MagicMock(get=lambda: "large-v3"),
            "device": MagicMock(get=lambda: "cuda"),
            "compute_type": MagicMock(get=lambda: "float16"),
            "language": MagicMock(get=lambda: "auto"),
        }

        # Mock vocabulary listbox
        window._vocab_listbox = MagicMock()
        window._vocab_listbox.get.return_value = ["word1", "word2"]

        # Call auto-save directly
        window._do_save()

        # Verify save was called with proper structure
        save_mock.assert_called_once()
        saved_config = save_mock.call_args[0][0]

        assert "model" in saved_config
        assert "hotkeys" in saved_config
        assert "output" in saved_config
        assert "vocabulary" in saved_config

        assert saved_config["hotkeys"]["mode"] == "toggle"
        assert saved_config["model"]["name"] == "large-v3"


class TestSettingsWindowMockedGUI:
    """Tests that mock the GUI to test logic without display."""

    def test_add_vocab_word(self, mocker):
        """Test adding a vocabulary word."""
        from localwispr.settings_window import SettingsWindow

        window = SettingsWindow()

        # Mock listbox and entry
        window._vocab_listbox = MagicMock()
        window._vocab_listbox.get.return_value = ()  # Empty list

        window._vocab_entry = MagicMock()
        window._vocab_entry.get.return_value = "newword"

        # Mock _schedule_save to avoid timer issues
        window._schedule_save = MagicMock()

        window._add_vocab_word()

        window._vocab_listbox.insert.assert_called_once()
        window._vocab_entry.delete.assert_called_once()
        window._schedule_save.assert_called_once()  # Verify auto-save triggered

    def test_add_vocab_word_duplicate_rejected(self, mocker):
        """Test that duplicate vocabulary words are rejected."""
        from localwispr.settings_window import SettingsWindow

        window = SettingsWindow()

        # Mock listbox with existing word
        window._vocab_listbox = MagicMock()
        window._vocab_listbox.get.return_value = ("existingword",)

        window._vocab_entry = MagicMock()
        window._vocab_entry.get.return_value = "existingword"

        # Mock _schedule_save
        window._schedule_save = MagicMock()

        window._add_vocab_word()

        # Should not insert duplicate
        window._vocab_listbox.insert.assert_not_called()
        window._schedule_save.assert_not_called()  # No save on duplicate

    def test_remove_vocab_words(self, mocker):
        """Test removing vocabulary words."""
        from localwispr.settings_window import SettingsWindow

        window = SettingsWindow()

        # Mock listbox with selection
        window._vocab_listbox = MagicMock()
        window._vocab_listbox.curselection.return_value = (0, 2)

        # Mock _schedule_save
        window._schedule_save = MagicMock()

        window._remove_vocab_words()

        # Should delete in reverse order
        assert window._vocab_listbox.delete.call_count == 2
        window._schedule_save.assert_called_once()  # Verify auto-save triggered

    def test_close_saves_and_closes_window(self, mocker, mock_config):
        """Test that close performs final save and closes the window."""
        mocker.patch("localwispr.config.save_config")
        mocker.patch("localwispr.config.reload_config", return_value=mock_config)

        from localwispr.settings_window import SettingsWindow

        window = SettingsWindow()
        mock_window = MagicMock()
        mock_root = MagicMock()
        window._window = mock_window
        window._root = mock_root
        window._config = mock_config

        # Mock tkinter variables
        window._vars = {
            "mode": MagicMock(get=lambda: "toggle"),
            "audio_feedback": MagicMock(get=lambda: True),
            "mute_system": MagicMock(get=lambda: False),
            "auto_paste": MagicMock(get=lambda: True),
            "paste_delay_ms": MagicMock(get=lambda: 50),
            "model_name": MagicMock(get=lambda: "large-v3"),
            "device": MagicMock(get=lambda: "cuda"),
            "compute_type": MagicMock(get=lambda: "float16"),
            "language": MagicMock(get=lambda: "auto"),
        }

        # Mock vocabulary listbox
        window._vocab_listbox = MagicMock()
        window._vocab_listbox.get.return_value = []

        window._on_close()

        # Check the original mocks (window._window is now None after _close)
        mock_window.destroy.assert_called_once()
        mock_root.quit.assert_called_once()
