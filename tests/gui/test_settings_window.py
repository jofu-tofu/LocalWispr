"""GUI tests for the Settings Window.

These tests verify the settings window behavior using Tkinter testing
patterns. They don't require the full application to be running.

Note: These tests use Tkinter's test mode and don't require pywinauto
for basic functionality testing.
"""

from __future__ import annotations

from unittest.mock import MagicMock


class TestSettingsWindowCreation:
    """Tests for TkinterSettingsView instantiation and setup."""

    def test_settings_window_instantiation(self):
        """Test that TkinterSettingsView can be instantiated."""
        from localwispr.settings.window import TkinterSettingsView

        window = TkinterSettingsView()

        assert window._window is None
        assert window._root is None
        assert window.on_save_requested is None
        assert window.on_cancel_requested is None
        assert window.on_setting_changed is None

    def test_settings_window_alias(self):
        """Test that SettingsWindow alias works."""
        from localwispr.settings.window import SettingsWindow, TkinterSettingsView

        assert SettingsWindow is TkinterSettingsView


class TestSettingsWindowConstants:
    """Tests for SettingsWindow constants and options."""

    def test_model_sizes_defined(self):
        """Test that MODEL_SIZES contains expected options."""
        from localwispr.settings.window import MODEL_SIZES

        assert "tiny" in MODEL_SIZES
        assert "base" in MODEL_SIZES
        assert "small" in MODEL_SIZES
        assert "medium" in MODEL_SIZES
        assert "large-v3" in MODEL_SIZES

    def test_devices_defined(self):
        """Test that DEVICES contains expected options."""
        from localwispr.settings.window import DEVICES

        assert "cuda" in DEVICES
        assert "cpu" in DEVICES

    def test_compute_types_defined(self):
        """Test that COMPUTE_TYPES contains expected options."""
        from localwispr.settings.window import COMPUTE_TYPES

        assert "float16" in COMPUTE_TYPES
        assert "int8" in COMPUTE_TYPES
        assert "float32" in COMPUTE_TYPES

    def test_languages_defined(self):
        """Test that LANGUAGES contains expected options."""
        from localwispr.settings.window import LANGUAGES

        language_codes = [code for _, code in LANGUAGES]

        assert "auto" in language_codes
        assert "en" in language_codes
        assert "es" in language_codes
        assert "zh" in language_codes


class TestSettingsWindowDimensions:
    """Tests for SettingsWindow dimension constants."""

    def test_window_dimensions(self):
        """Test that window dimensions are reasonable."""
        from localwispr.settings.window import TkinterSettingsView

        assert TkinterSettingsView.WINDOW_WIDTH > 0
        assert TkinterSettingsView.WINDOW_HEIGHT > 0
        assert TkinterSettingsView.WINDOW_WIDTH >= 400
        assert TkinterSettingsView.WINDOW_HEIGHT >= 400


class TestOpenSettingsFunction:
    """Tests for the open_settings convenience function."""

    def test_open_settings_creates_controller(self, mocker):
        """Test that open_settings creates a SettingsController."""
        mock_view_instance = MagicMock()
        mocker.patch(
            "localwispr.settings.window.TkinterSettingsView",
            return_value=mock_view_instance,
        )
        mock_controller_instance = MagicMock()
        mock_controller_class = mocker.patch(
            "localwispr.settings.controller.SettingsController",
            return_value=mock_controller_instance,
        )

        from localwispr.settings.window import open_settings

        open_settings()

        mock_controller_class.assert_called_once_with(
            mock_view_instance, on_settings_applied=None
        )
        mock_controller_instance.open.assert_called_once()

    def test_open_settings_passes_callback(self, mocker):
        """Test that open_settings passes callback to controller."""
        mock_view_instance = MagicMock()
        mocker.patch(
            "localwispr.settings.window.TkinterSettingsView",
            return_value=mock_view_instance,
        )
        mock_controller_instance = MagicMock()
        mock_controller_class = mocker.patch(
            "localwispr.settings.controller.SettingsController",
            return_value=mock_controller_instance,
        )

        callback = MagicMock()

        from localwispr.settings.window import open_settings

        open_settings(on_settings_changed=callback)

        mock_controller_class.assert_called_once_with(
            mock_view_instance, on_settings_applied=callback
        )


class TestVocabularyValidation:
    """Tests for vocabulary word validation."""

    def test_validate_vocab_word_valid(self):
        """Test validation accepts valid words."""
        from localwispr.settings.window import TkinterSettingsView

        window = TkinterSettingsView()

        is_valid, result = window._validate_vocab_word("hello")
        assert is_valid is True
        assert result == "hello"

    def test_validate_vocab_word_strips_whitespace(self):
        """Test validation strips leading/trailing whitespace."""
        from localwispr.settings.window import TkinterSettingsView

        window = TkinterSettingsView()

        is_valid, result = window._validate_vocab_word("  hello  ")
        assert is_valid is True
        assert result == "hello"

    def test_validate_vocab_word_empty(self):
        """Test validation rejects empty words."""
        from localwispr.settings.window import TkinterSettingsView

        window = TkinterSettingsView()

        is_valid, result = window._validate_vocab_word("")
        assert is_valid is False
        assert "empty" in result.lower()

    def test_validate_vocab_word_whitespace_only(self):
        """Test validation rejects whitespace-only input."""
        from localwispr.settings.window import TkinterSettingsView

        window = TkinterSettingsView()

        is_valid, result = window._validate_vocab_word("   ")
        assert is_valid is False
        assert "empty" in result.lower()

    def test_validate_vocab_word_too_long(self):
        """Test validation rejects words over 100 characters."""
        from localwispr.settings.window import TkinterSettingsView

        window = TkinterSettingsView()

        long_word = "a" * 101
        is_valid, result = window._validate_vocab_word(long_word)
        assert is_valid is False
        assert "long" in result.lower()

    def test_validate_vocab_word_rejects_quotes(self):
        """Test validation rejects double quotes (TOML unsafe)."""
        from localwispr.settings.window import TkinterSettingsView

        window = TkinterSettingsView()

        is_valid, result = window._validate_vocab_word('hello"world')
        assert is_valid is False
        assert "invalid" in result.lower() or "quote" in result.lower()

    def test_validate_vocab_word_rejects_backslash(self):
        """Test validation rejects backslashes (TOML unsafe)."""
        from localwispr.settings.window import TkinterSettingsView

        window = TkinterSettingsView()

        is_valid, result = window._validate_vocab_word("hello\\world")
        assert is_valid is False
        assert "invalid" in result.lower() or "backslash" in result.lower()

    def test_validate_vocab_word_rejects_newline(self):
        """Test validation rejects newlines (TOML unsafe)."""
        from localwispr.settings.window import TkinterSettingsView

        window = TkinterSettingsView()

        is_valid, result = window._validate_vocab_word("hello\nworld")
        assert is_valid is False
        assert "invalid" in result.lower() or "newline" in result.lower()

    def test_validate_vocab_word_rejects_tab(self):
        """Test validation rejects tabs (TOML unsafe)."""
        from localwispr.settings.window import TkinterSettingsView

        window = TkinterSettingsView()

        is_valid, result = window._validate_vocab_word("hello\tworld")
        assert is_valid is False
        assert "invalid" in result.lower() or "tab" in result.lower()

    def test_validate_vocab_word_accepts_spaces(self):
        """Test validation accepts words with internal spaces."""
        from localwispr.settings.window import TkinterSettingsView

        window = TkinterSettingsView()

        is_valid, result = window._validate_vocab_word("hello world")
        assert is_valid is True
        assert result == "hello world"

    def test_validate_vocab_word_accepts_unicode(self):
        """Test validation accepts unicode characters."""
        from localwispr.settings.window import TkinterSettingsView

        window = TkinterSettingsView()

        # Common unicode scenarios
        is_valid, result = window._validate_vocab_word("cafe")
        assert is_valid is True


class TestSettingsWindowMockedGUI:
    """Tests that mock the GUI to test logic without display."""

    def test_add_vocab_word(self, mocker):
        """Test adding a vocabulary word."""
        from localwispr.settings.window import TkinterSettingsView

        window = TkinterSettingsView()

        # Mock listbox and entry
        window._vocab_listbox = MagicMock()
        window._vocab_listbox.get.return_value = ()  # Empty list

        window._vocab_entry = MagicMock()
        window._vocab_entry.get.return_value = "newword"

        # Set up on_setting_changed callback
        callback = MagicMock()
        window.on_setting_changed = callback

        window._add_vocab_word()

        window._vocab_listbox.insert.assert_called_once()
        window._vocab_entry.delete.assert_called_once()
        callback.assert_called_once_with("vocabulary_words", None)

    def test_add_vocab_word_duplicate_rejected(self, mocker):
        """Test that duplicate vocabulary words are rejected."""
        from localwispr.settings.window import TkinterSettingsView

        # Mock messagebox in window_tabs where _add_vocab_word lives
        mocker.patch("localwispr.settings.window_tabs.messagebox")

        window = TkinterSettingsView()

        # Mock listbox with existing word
        window._vocab_listbox = MagicMock()
        window._vocab_listbox.get.return_value = ("existingword",)

        window._vocab_entry = MagicMock()
        window._vocab_entry.get.return_value = "existingword"

        # Set up on_setting_changed callback
        callback = MagicMock()
        window.on_setting_changed = callback

        window._add_vocab_word()

        # Should not insert duplicate
        window._vocab_listbox.insert.assert_not_called()
        callback.assert_not_called()

    def test_remove_vocab_words(self, mocker):
        """Test removing vocabulary words."""
        from localwispr.settings.window import TkinterSettingsView

        window = TkinterSettingsView()

        # Mock listbox with selection
        window._vocab_listbox = MagicMock()
        window._vocab_listbox.curselection.return_value = (0, 2)

        # Set up on_setting_changed callback
        callback = MagicMock()
        window.on_setting_changed = callback

        window._remove_vocab_words()

        # Should delete in reverse order
        assert window._vocab_listbox.delete.call_count == 2
        callback.assert_called_once_with("vocabulary_words", None)

    def test_close_destroys_window(self, mocker):
        """Test that close destroys window and root."""
        from localwispr.settings.window import TkinterSettingsView

        window = TkinterSettingsView()
        mock_window = MagicMock()
        mock_root = MagicMock()
        window._window = mock_window
        window._root = mock_root

        window.close()

        mock_window.destroy.assert_called_once()
        mock_root.quit.assert_called_once()
        mock_root.destroy.assert_called_once()
        assert window._window is None
        assert window._root is None
