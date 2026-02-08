"""GUI tests for the System Tray application.

These tests verify tray functionality using mocks to avoid
requiring a running system tray.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.gui



class TestTrayState:
    """Tests for TrayState enum."""

    def test_tray_state_values(self):
        """Test TrayState enum values."""
        from localwispr.ui.tray import TrayState

        assert TrayState.IDLE is not None
        assert TrayState.RECORDING is not None
        assert TrayState.TRANSCRIBING is not None

    def test_tray_states_are_distinct(self):
        """Test that TrayState values are distinct."""
        from localwispr.ui.tray import TrayState

        states = [TrayState.IDLE, TrayState.RECORDING, TrayState.TRANSCRIBING]
        assert len(set(states)) == 3


class TestStateColors:
    """Tests for STATE_COLORS configuration."""

    def test_state_colors_defined_for_all_states(self):
        """Test that colors are defined for all states."""
        from localwispr.ui.tray import STATE_COLORS, TrayState

        for state in TrayState:
            assert state in STATE_COLORS
            assert "wave" in STATE_COLORS[state]
            assert "letter" in STATE_COLORS[state]

    def test_state_colors_are_valid(self):
        """Test that state colors are valid hex codes."""
        from localwispr.ui.tray import STATE_COLORS

        for state, colors in STATE_COLORS.items():
            for key, value in colors.items():
                if value is not None:
                    assert value.startswith("#") or value is None


class TestCreateIconImage:
    """Tests for create_icon_image function."""

    def test_create_icon_default_state(self):
        """Test creating icon with default state."""
        from localwispr.ui.tray import create_icon_image

        image = create_icon_image()

        assert image is not None
        assert image.size == (64, 64)
        assert image.mode == "RGBA"

    def test_create_icon_recording_state(self):
        """Test creating icon with recording state."""
        from localwispr.ui.tray import TrayState, create_icon_image

        image = create_icon_image(TrayState.RECORDING)

        assert image is not None
        assert image.size == (64, 64)

    def test_create_icon_transcribing_state(self):
        """Test creating icon with transcribing state."""
        from localwispr.ui.tray import TrayState, create_icon_image

        image = create_icon_image(TrayState.TRANSCRIBING)

        assert image is not None
        assert image.size == (64, 64)

    def test_create_icon_custom_size(self):
        """Test creating icon with custom size."""
        from localwispr.ui.tray import create_icon_image

        image = create_icon_image(size=128)

        assert image.size == (128, 128)

    def test_create_icon_same_state_colors(self):
        """Test that different states use the same color (no state transitions)."""
        from localwispr.ui.tray import TrayState, create_icon_image

        idle_icon = create_icon_image(TrayState.IDLE)
        recording_icon = create_icon_image(TrayState.RECORDING)
        transcribing_icon = create_icon_image(TrayState.TRANSCRIBING)

        # Images should be identical (same colors for all states)
        # Compare by converting to bytes
        idle_data = list(idle_icon.getdata())
        recording_data = list(recording_icon.getdata())
        transcribing_data = list(transcribing_icon.getdata())

        assert idle_data == recording_data
        assert idle_data == transcribing_data


class TestTrayAppInitialization:
    """Tests for TrayApp initialization."""

    def test_tray_app_initialization(self, mocker, reset_mode_manager):
        """Test TrayApp can be initialized."""
        # Mock overlay to avoid GUI
        mocker.patch("localwispr.ui.overlay.OverlayWidget")
        mocker.patch("localwispr.modes.get_mode_manager")

        from localwispr.ui.tray import TrayApp, TrayState

        app = TrayApp()

        assert app.state == TrayState.IDLE
        assert app._icon is None

    def test_tray_app_with_callback(self, mocker, reset_mode_manager):
        """Test TrayApp with state change callback."""
        mocker.patch("localwispr.ui.overlay.OverlayWidget")
        mocker.patch("localwispr.modes.get_mode_manager")

        callback = MagicMock()

        from localwispr.ui.tray import TrayApp

        app = TrayApp(on_state_change=callback)

        assert app._on_state_change is callback


class TestTrayAppState:
    """Tests for TrayApp state management."""

    def test_update_state_queues_update(self, mocker, reset_mode_manager):
        """Test that update_state queues state updates."""
        mocker.patch("localwispr.ui.overlay.OverlayWidget")
        mocker.patch("localwispr.modes.get_mode_manager")

        from localwispr.ui.tray import TrayApp, TrayState

        app = TrayApp()

        app.update_state(TrayState.RECORDING)

        # State should be queued
        assert not app._update_queue.empty()

    def test_process_queue_applies_state(self, mocker, reset_mode_manager):
        """Test that _process_queue applies queued states."""
        mocker.patch("localwispr.ui.overlay.OverlayWidget")
        mocker.patch("localwispr.modes.get_mode_manager")

        from localwispr.ui.tray import TrayApp, TrayState

        app = TrayApp()
        app._icon = MagicMock()

        app.update_state(TrayState.RECORDING)
        app._process_queue()

        assert app.state == TrayState.RECORDING

    def test_set_state_invokes_callback(self, mocker, reset_mode_manager):
        """Test that _set_state invokes the state change callback."""
        mocker.patch("localwispr.ui.overlay.OverlayWidget")
        mocker.patch("localwispr.modes.get_mode_manager")

        callback = MagicMock()

        from localwispr.ui.tray import TrayApp, TrayState

        app = TrayApp(on_state_change=callback)
        app._icon = MagicMock()

        app._set_state(TrayState.RECORDING)

        callback.assert_called_once_with(TrayState.RECORDING)

    def test_set_state_same_state_no_callback(self, mocker, reset_mode_manager):
        """Test that setting same state doesn't invoke callback."""
        mocker.patch("localwispr.ui.overlay.OverlayWidget")
        mocker.patch("localwispr.modes.get_mode_manager")

        callback = MagicMock()

        from localwispr.ui.tray import TrayApp, TrayState

        app = TrayApp(on_state_change=callback)

        # Already in IDLE state
        app._set_state(TrayState.IDLE)

        callback.assert_not_called()


class TestTrayAppBackgroundThreads:
    """Tests for TrayApp background thread management."""

    def test_register_background_thread(self, mocker, reset_mode_manager):
        """Test registering a background thread."""
        mocker.patch("localwispr.ui.overlay.OverlayWidget")
        mocker.patch("localwispr.modes.get_mode_manager")

        from localwispr.ui.tray import TrayApp

        app = TrayApp()
        thread = threading.Thread(target=lambda: None)

        app.register_background_thread(thread)

        assert thread in app._background_threads

    def test_stop_event_shared(self, mocker, reset_mode_manager):
        """Test that stop_event is accessible."""
        mocker.patch("localwispr.ui.overlay.OverlayWidget")
        mocker.patch("localwispr.modes.get_mode_manager")

        from localwispr.ui.tray import TrayApp

        app = TrayApp()

        assert app.stop_event is not None
        assert not app.stop_event.is_set()


class TestTrayAppShutdown:
    """Tests for TrayApp shutdown sequence."""

    def test_shutdown_sets_stop_event(self, mocker, reset_mode_manager):
        """Test that shutdown sets the stop event."""
        mocker.patch("localwispr.ui.overlay.OverlayWidget")
        mocker.patch("localwispr.modes.get_mode_manager")

        from localwispr.ui.tray import TrayApp

        app = TrayApp()
        app._icon = MagicMock()
        app._hotkey_listener = MagicMock()

        app._shutdown()

        assert app._stop_event.is_set()

    def test_shutdown_stops_hotkey_listener(self, mocker, reset_mode_manager):
        """Test that shutdown stops the hotkey listener."""
        mocker.patch("localwispr.ui.overlay.OverlayWidget")
        mocker.patch("localwispr.modes.get_mode_manager")

        from localwispr.ui.tray import TrayApp

        app = TrayApp()
        app._icon = MagicMock()

        mock_listener = MagicMock()
        app._hotkey_listener = mock_listener

        app._shutdown()

        mock_listener.stop.assert_called_once()

    def test_shutdown_stops_overlay(self, mocker, reset_mode_manager):
        """Test that shutdown stops the overlay widget."""
        mock_overlay = MagicMock()
        mocker.patch("localwispr.ui.overlay.OverlayWidget", return_value=mock_overlay)
        mocker.patch("localwispr.modes.get_mode_manager")

        from localwispr.ui.tray import TrayApp

        app = TrayApp()
        app._icon = MagicMock()

        app._shutdown()

        mock_overlay.stop.assert_called_once()


class TestTrayAppModeSwitching:
    """Tests for tray mode switching functionality."""

    def test_is_mode_ptt(self, mocker, reset_mode_manager, mock_config):
        """Test is_mode_ptt method."""
        mocker.patch("localwispr.ui.overlay.OverlayWidget")
        mocker.patch("localwispr.modes.get_mode_manager")
        mocker.patch("localwispr.config.get_config", return_value=mock_config)

        from localwispr.ui.tray import TrayApp

        app = TrayApp()

        # mock_config has mode = "push-to-talk"
        assert app._is_mode_ptt() is True

    def test_on_mode_changed_shows_notification(self, mocker, reset_mode_manager):
        """Test that mode change shows notification."""
        mocker.patch("localwispr.ui.overlay.OverlayWidget")
        mocker.patch("localwispr.modes.get_mode_manager")

        mock_notification = mocker.patch("localwispr.ui.notifications.show_notification")

        from localwispr.modes import Mode, ModeType
        from localwispr.ui.tray import TrayApp

        app = TrayApp()
        app._icon = MagicMock()

        test_mode = Mode(
            mode_type=ModeType.CODE,
            name="Code",
            description="Test description",
            prompt_file="coding",
            icon="</>",
        )

        app._on_mode_changed(test_mode)

        mock_notification.assert_called_once()


class TestTrayAppRecordingCallbacks:
    """Tests for tray recording callbacks."""

    def test_on_record_start_updates_state(self, mocker, reset_mode_manager, mock_config):
        """Test that recording start updates state."""
        mocker.patch("localwispr.ui.overlay.OverlayWidget")
        mocker.patch("localwispr.modes.get_mode_manager")
        mocker.patch("localwispr.config.get_config", return_value=mock_config)
        mocker.patch("localwispr.audio.feedback.play_start_beep")

        mock_recorder = MagicMock()
        mock_recorder.is_recording = False
        mocker.patch("localwispr.audio.AudioRecorder", return_value=mock_recorder)

        from localwispr.ui.tray import TrayApp, TrayState

        app = TrayApp()

        app._on_record_start()

        # Should queue RECORDING state
        assert not app._update_queue.empty()
        queued_state = app._update_queue.get()
        assert queued_state == TrayState.RECORDING

    def test_finish_transcription_resets_state(self, mocker, reset_mode_manager):
        """Test that finishing transcription resets state."""
        mocker.patch("localwispr.ui.overlay.OverlayWidget")
        mocker.patch("localwispr.modes.get_mode_manager")

        from localwispr.ui.tray import TrayApp, TrayState

        app = TrayApp()
        app._hotkey_listener = MagicMock()

        app._finish_transcription()

        # Should queue IDLE state
        assert not app._update_queue.empty()
        queued_state = app._update_queue.get()
        assert queued_state == TrayState.IDLE


class TestTrayAppAudioLevel:
    """Tests for tray audio level functionality."""

    def test_get_current_audio_level_not_recording(self, mocker, reset_mode_manager):
        """Test audio level when not recording."""
        mocker.patch("localwispr.ui.overlay.OverlayWidget")
        mocker.patch("localwispr.modes.get_mode_manager")

        from localwispr.ui.tray import TrayApp

        app = TrayApp()

        level = app._get_current_audio_level()

        assert level == 0.0

    def test_get_current_audio_level_while_recording(self, mocker, reset_mode_manager):
        """Test audio level while recording."""
        mocker.patch("localwispr.ui.overlay.OverlayWidget")
        mocker.patch("localwispr.modes.get_mode_manager")

        mock_recorder = MagicMock()
        mock_recorder.is_recording = True
        mock_recorder.get_rms_level.return_value = 0.75

        from localwispr.ui.tray import TrayApp

        app = TrayApp()
        app._pipeline._recorder = mock_recorder

        level = app._get_current_audio_level()

        assert level == 0.75
