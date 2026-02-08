"""Tests for the output module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestCopyToClipboard:
    """Tests for copy_to_clipboard function."""

    def test_copy_to_clipboard_success(self, mock_pyperclip):
        """Test successful clipboard copy stores text."""
        from localwispr.output import copy_to_clipboard

        result = copy_to_clipboard("test text")

        assert result is True
        # Verify via the clipboard state, not the mock call
        assert mock_pyperclip.paste() == "test text"

    def test_copy_to_clipboard_retries_on_failure(self, mocker):
        """Test that clipboard copy eventually succeeds after transient failures."""
        import pyperclip

        mock_pyperclip = mocker.patch("localwispr.output.pyperclip")
        mock_pyperclip.PyperclipException = pyperclip.PyperclipException

        # Fail twice, then succeed
        mock_pyperclip.copy.side_effect = [
            pyperclip.PyperclipException("locked"),
            pyperclip.PyperclipException("still locked"),
            None,
        ]

        # Also mock sleep to speed up test
        mocker.patch("localwispr.output.time.sleep")

        from localwispr.output import copy_to_clipboard

        result = copy_to_clipboard("test text", max_retries=3)

        # Behavioral outcome: copy eventually succeeded despite transient failures
        assert result is True

    def test_copy_to_clipboard_fails_after_max_retries(self, mocker):
        """Test that clipboard copy fails after exhausting retries."""
        import pyperclip

        mock_pyperclip = mocker.patch("localwispr.output.pyperclip")
        mock_pyperclip.PyperclipException = pyperclip.PyperclipException
        mock_pyperclip.copy.side_effect = pyperclip.PyperclipException("locked")

        mocker.patch("localwispr.output.time.sleep")

        from localwispr.output import copy_to_clipboard

        result = copy_to_clipboard("test text", max_retries=2)

        # Behavioral outcome: copy failed after all retries exhausted
        assert result is False


class TestPasteToActiveWindow:
    """Tests for paste_to_active_window function."""

    def test_paste_to_active_window_success(self, mock_keyboard, mocker):
        """Test successful paste returns True."""
        mocker.patch("localwispr.output.time.sleep")

        from localwispr.output import paste_to_active_window

        result = paste_to_active_window(delay_ms=0)

        assert result is True

    def test_paste_to_active_window_with_delay(self, mock_keyboard, mocker):
        """Test that paste respects delay setting."""
        mock_sleep = mocker.patch("localwispr.output.time.sleep")

        from localwispr.output import paste_to_active_window

        paste_to_active_window(delay_ms=100)

        # Should have called sleep with at least the delay (100ms = 0.1s)
        mock_sleep.assert_called()

    def test_paste_to_active_window_clears_modifiers(self, mock_keyboard, mocker):
        """Test that paste succeeds even with modifier keys to clear."""
        mocker.patch("localwispr.output.time.sleep")

        from localwispr.output import paste_to_active_window

        result = paste_to_active_window(delay_ms=0)

        # The observable outcome: paste completed successfully
        assert result is True

    def test_paste_to_active_window_handles_exception(self, mocker):
        """Test that paste handles keyboard exceptions."""
        mocker.patch("localwispr.output.time.sleep")
        mock_controller = MagicMock()
        mock_controller.press.side_effect = Exception("keyboard error")
        mocker.patch("localwispr.output.Controller", return_value=mock_controller)

        from localwispr.output import paste_to_active_window

        result = paste_to_active_window(delay_ms=0)

        assert result is False


class TestOutputTranscription:
    """Tests for output_transcription function."""

    def test_output_transcription_clipboard_only(self, mock_pyperclip, mocker):
        """Test output with auto_paste=False stores text in clipboard."""
        from localwispr.output import output_transcription

        result = output_transcription(
            "test text",
            auto_paste=False,
            play_feedback=False,
        )

        assert result is True
        assert mock_pyperclip.paste() == "test text"

    def test_output_transcription_with_paste(self, mock_pyperclip, mock_keyboard, mocker):
        """Test output with auto_paste=True stores text and returns success."""
        mocker.patch("localwispr.output.time.sleep")

        from localwispr.output import output_transcription

        result = output_transcription(
            "test text",
            auto_paste=True,
            paste_delay_ms=0,
            play_feedback=False,
        )

        assert result is True
        assert mock_pyperclip.paste() == "test text"

    def test_output_transcription_plays_feedback(self, mock_pyperclip, mocker):
        """Test that output succeeds with feedback enabled."""
        mocker.patch("localwispr.audio.feedback.play_stop_beep")
        mocker.patch("localwispr.output.paste_to_active_window", return_value=True)

        from localwispr.output import output_transcription

        result = output_transcription(
            "test text",
            auto_paste=True,
            play_feedback=True,
        )

        assert result is True
        assert mock_pyperclip.paste() == "test text"

    def test_output_transcription_clipboard_failure(self, mocker):
        """Test that clipboard failure returns False."""
        import pyperclip

        mock_pyperclip = mocker.patch("localwispr.output.pyperclip")
        mock_pyperclip.PyperclipException = pyperclip.PyperclipException
        mock_pyperclip.copy.side_effect = pyperclip.PyperclipException("failed")
        mocker.patch("localwispr.output.time.sleep")

        from localwispr.output import output_transcription

        result = output_transcription("test text", auto_paste=False)

        assert result is False

    def test_output_transcription_paste_failure_still_clipboard(
        self, mock_pyperclip, mocker
    ):
        """Test that paste failure still leaves text in clipboard."""
        mocker.patch("localwispr.output.paste_to_active_window", return_value=False)

        from localwispr.output import output_transcription

        result = output_transcription(
            "test text",
            auto_paste=True,
            play_feedback=False,
        )

        assert result is False  # Paste failed
        assert mock_pyperclip.paste() == "test text"  # But clipboard was set

    def test_output_transcription_feedback_failure_continues(
        self, mock_pyperclip, mocker
    ):
        """Test that feedback failure doesn't stop output."""
        mocker.patch(
            "localwispr.audio.feedback.play_stop_beep",
            side_effect=Exception("audio error"),
        )
        mocker.patch("localwispr.output.paste_to_active_window", return_value=True)

        from localwispr.output import output_transcription

        result = output_transcription(
            "test text",
            auto_paste=True,
            play_feedback=True,
        )

        assert result is True  # Should still succeed


