"""Tests for the notifications module."""

from __future__ import annotations

from unittest.mock import MagicMock



class TestShowNotification:
    """Tests for show_notification function."""

    def test_show_notification_success(self, mocker):
        """Test that show_notification returns True on success."""
        mock_notification = MagicMock()
        mocker.patch(
            "localwispr.ui.notifications.Notification",
            return_value=mock_notification,
        )

        from localwispr.ui.notifications import show_notification

        result = show_notification("Title", "Message")

        assert result is True

    def test_show_notification_returns_false_on_error(self, mocker):
        """Test that show_notification returns False on exception."""
        mocker.patch(
            "localwispr.ui.notifications.Notification",
            side_effect=Exception("Notification failed"),
        )

        from localwispr.ui.notifications import show_notification

        result = show_notification("Title", "Message")

        assert result is False

class TestShowRecordingStarted:
    """Tests for show_recording_started function."""

    def test_show_recording_started_returns_true(self, mocker):
        """Test that show_recording_started succeeds with Notification mocked."""
        mock_notification = MagicMock()
        mocker.patch(
            "localwispr.ui.notifications.Notification",
            return_value=mock_notification,
        )

        from localwispr.ui.notifications import show_recording_started

        assert show_recording_started() is True

    def test_show_recording_started_returns_false_on_error(self, mocker):
        """Test that show_recording_started returns False on failure."""
        mocker.patch(
            "localwispr.ui.notifications.Notification",
            side_effect=Exception("toast error"),
        )

        from localwispr.ui.notifications import show_recording_started

        assert show_recording_started() is False


class TestShowTranscribing:
    """Tests for show_transcribing function."""

    def test_show_transcribing_returns_true(self, mocker):
        """Test that show_transcribing succeeds with Notification mocked."""
        mock_notification = MagicMock()
        mocker.patch(
            "localwispr.ui.notifications.Notification",
            return_value=mock_notification,
        )

        from localwispr.ui.notifications import show_transcribing

        assert show_transcribing() is True


class TestShowComplete:
    """Tests for show_complete function."""

    def test_show_complete_returns_true(self, mocker):
        """Test that show_complete succeeds with Notification mocked."""
        mock_notification = MagicMock()
        mocker.patch(
            "localwispr.ui.notifications.Notification",
            return_value=mock_notification,
        )

        from localwispr.ui.notifications import show_complete

        assert show_complete() is True


class TestShowError:
    """Tests for show_error function."""

    def test_show_error_sanitizes_message(self, mocker):
        """Test that error messages are sanitized."""
        mock_show = mocker.patch(
            "localwispr.ui.notifications.show_notification",
            return_value=True,
        )

        from localwispr.ui.notifications import show_error

        # Message with newlines
        show_error("Error\nwith\nnewlines")

        # show_notification is called with keyword args
        call_kwargs = mock_show.call_args[1]
        message = call_kwargs["message"]
        assert "\n" not in message
        assert "\r" not in message

    def test_show_error_truncates_long_message(self, mocker):
        """Test that long messages are truncated."""
        mock_show = mocker.patch(
            "localwispr.ui.notifications.show_notification",
            return_value=True,
        )

        from localwispr.ui.notifications import show_error

        # Very long message
        long_message = "x" * 200
        show_error(long_message)

        call_kwargs = mock_show.call_args[1]
        message = call_kwargs["message"]
        assert len(message) <= 100


class TestShowClipboardOnly:
    """Tests for show_clipboard_only function."""

    def test_show_clipboard_only_correct_message(self, mocker):
        """Test that clipboard-only notification has correct message."""
        mock_show = mocker.patch(
            "localwispr.ui.notifications.show_notification",
            return_value=True,
        )

        from localwispr.ui.notifications import show_clipboard_only

        show_clipboard_only()

        call_kwargs = mock_show.call_args[1]
        message = call_kwargs["message"]
        assert "clipboard" in message.lower()
