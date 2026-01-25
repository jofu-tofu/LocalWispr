"""Tests for the notifications module."""

from __future__ import annotations

from unittest.mock import MagicMock



class TestShowNotification:
    """Tests for show_notification function."""

    def test_show_notification_success(self, mocker):
        """Test that show_notification returns True on success."""
        mock_notification = MagicMock()
        mocker.patch(
            "localwispr.notifications.Notification",
            return_value=mock_notification,
        )

        from localwispr.notifications import show_notification

        result = show_notification("Title", "Message")

        assert result is True
        mock_notification.show.assert_called_once()

    def test_show_notification_returns_false_on_error(self, mocker):
        """Test that show_notification returns False on exception."""
        mocker.patch(
            "localwispr.notifications.Notification",
            side_effect=Exception("Notification failed"),
        )

        from localwispr.notifications import show_notification

        result = show_notification("Title", "Message")

        assert result is False

    def test_show_notification_short_duration(self, mocker):
        """Test that timeout <= 5 uses 'short' duration."""
        mock_notification = MagicMock()
        mock_notification_class = mocker.patch(
            "localwispr.notifications.Notification",
            return_value=mock_notification,
        )

        from localwispr.notifications import show_notification

        show_notification("Title", "Message", timeout=5)

        # Check duration parameter
        call_kwargs = mock_notification_class.call_args[1]
        assert call_kwargs["duration"] == "short"

    def test_show_notification_long_duration(self, mocker):
        """Test that timeout > 5 uses 'long' duration."""
        mock_notification = MagicMock()
        mock_notification_class = mocker.patch(
            "localwispr.notifications.Notification",
            return_value=mock_notification,
        )

        from localwispr.notifications import show_notification

        show_notification("Title", "Message", timeout=10)

        # Check duration parameter
        call_kwargs = mock_notification_class.call_args[1]
        assert call_kwargs["duration"] == "long"

    def test_show_notification_sets_audio(self, mocker):
        """Test that notification has audio set."""
        mock_notification = MagicMock()
        mocker.patch(
            "localwispr.notifications.Notification",
            return_value=mock_notification,
        )

        from localwispr.notifications import show_notification

        show_notification("Title", "Message")

        mock_notification.set_audio.assert_called_once()


class TestShowRecordingStarted:
    """Tests for show_recording_started function."""

    def test_show_recording_started_uses_long_duration(self, mocker):
        """Test that recording notification uses long duration."""
        mock_show = mocker.patch(
            "localwispr.notifications.show_notification",
            return_value=True,
        )

        from localwispr.notifications import show_recording_started

        show_recording_started()

        mock_show.assert_called_once()
        call_kwargs = mock_show.call_args[1]
        assert call_kwargs["timeout"] == 30


class TestShowTranscribing:
    """Tests for show_transcribing function."""

    def test_show_transcribing_uses_long_duration(self, mocker):
        """Test that transcribing notification uses long duration."""
        mock_show = mocker.patch(
            "localwispr.notifications.show_notification",
            return_value=True,
        )

        from localwispr.notifications import show_transcribing

        show_transcribing()

        mock_show.assert_called_once()
        call_kwargs = mock_show.call_args[1]
        assert call_kwargs["timeout"] == 30


class TestShowComplete:
    """Tests for show_complete function."""

    def test_show_complete_uses_short_duration(self, mocker):
        """Test that complete notification uses short duration."""
        mock_show = mocker.patch(
            "localwispr.notifications.show_notification",
            return_value=True,
        )

        from localwispr.notifications import show_complete

        show_complete()

        mock_show.assert_called_once()
        call_kwargs = mock_show.call_args[1]
        assert call_kwargs["timeout"] == 3


class TestShowError:
    """Tests for show_error function."""

    def test_show_error_sanitizes_message(self, mocker):
        """Test that error messages are sanitized."""
        mock_show = mocker.patch(
            "localwispr.notifications.show_notification",
            return_value=True,
        )

        from localwispr.notifications import show_error

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
            "localwispr.notifications.show_notification",
            return_value=True,
        )

        from localwispr.notifications import show_error

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
            "localwispr.notifications.show_notification",
            return_value=True,
        )

        from localwispr.notifications import show_clipboard_only

        show_clipboard_only()

        call_kwargs = mock_show.call_args[1]
        message = call_kwargs["message"]
        assert "clipboard" in message.lower()
