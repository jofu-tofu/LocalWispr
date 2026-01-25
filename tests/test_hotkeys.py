"""Tests for the hotkeys module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestHotkeyState:
    """Tests for HotkeyState enum."""

    def test_hotkey_state_values(self):
        """Test that HotkeyState has expected values."""
        from localwispr.hotkeys import HotkeyState

        assert hasattr(HotkeyState, "IDLE")
        assert hasattr(HotkeyState, "RECORDING")
        assert hasattr(HotkeyState, "TRANSCRIBING")


class TestHotkeyMode:
    """Tests for HotkeyMode enum."""

    def test_hotkey_mode_values(self):
        """Test that HotkeyMode has expected values."""
        from localwispr.hotkeys import HotkeyMode

        assert hasattr(HotkeyMode, "PUSH_TO_TALK")
        assert hasattr(HotkeyMode, "TOGGLE")


class TestHotkeyListenerError:
    """Tests for HotkeyListenerError exception."""

    def test_error_with_message(self):
        """Test error with just a message."""
        from localwispr.hotkeys import HotkeyListenerError

        error = HotkeyListenerError("Test error")

        assert "Test error" in str(error)

    def test_error_with_suggestion(self):
        """Test error with custom suggestion."""
        from localwispr.hotkeys import HotkeyListenerError

        error = HotkeyListenerError("Test error", suggestion="Try this")

        assert "Try this" in str(error)
        assert error.suggestion == "Try this"


class TestHotkeyListener:
    """Tests for HotkeyListener class."""

    @pytest.fixture
    def mock_config(self, mocker):
        """Mock get_config to return test configuration."""
        config = {
            "hotkeys": {
                "mode": "push-to-talk",
                "modifiers": ["win", "ctrl", "shift"],
                "audio_feedback": False,
                "mute_system": False,
            }
        }
        mocker.patch("localwispr.hotkeys.get_config", return_value=config)
        return config

    @pytest.fixture
    def mock_keyboard(self, mocker):
        """Mock pynput keyboard module."""
        mock_listener = MagicMock()
        mock_listener_class = mocker.patch(
            "localwispr.hotkeys.keyboard.Listener",
            return_value=mock_listener,
        )
        return {
            "listener_class": mock_listener_class,
            "listener": mock_listener,
        }

    def test_listener_initialization(self, mock_config, mock_keyboard):
        """Test that listener initializes with correct state."""
        from localwispr.hotkeys import HotkeyListener, HotkeyMode, HotkeyState

        listener = HotkeyListener()

        assert listener.state == HotkeyState.IDLE
        assert listener.mode == HotkeyMode.PUSH_TO_TALK
        assert not listener.is_running

    def test_listener_initialization_toggle_mode(self, mock_config, mock_keyboard):
        """Test listener initialization in toggle mode."""
        from localwispr.hotkeys import HotkeyListener, HotkeyMode

        listener = HotkeyListener(mode=HotkeyMode.TOGGLE)

        assert listener.mode == HotkeyMode.TOGGLE

    def test_listener_reads_modifiers_from_config(self, mocker, mock_keyboard):
        """Test that listener reads modifiers from config."""
        config = {
            "hotkeys": {
                "modifiers": ["ctrl", "alt", "shift"],
            }
        }
        mocker.patch("localwispr.hotkeys.get_config", return_value=config)

        from localwispr.hotkeys import HotkeyListener

        listener = HotkeyListener()

        assert listener._chord_modifiers == {"ctrl", "alt", "shift"}

    def test_listener_falls_back_to_defaults(self, mocker, mock_keyboard):
        """Test that listener uses defaults if modifiers invalid."""
        config = {
            "hotkeys": {
                "modifiers": ["invalid"],
            }
        }
        mocker.patch("localwispr.hotkeys.get_config", return_value=config)

        from localwispr.hotkeys import HotkeyListener

        listener = HotkeyListener()

        # Should fall back to default Win+Ctrl+Shift
        assert listener._chord_modifiers == {"win", "ctrl", "shift"}

    def test_listener_start(self, mock_config, mock_keyboard):
        """Test that start() creates and starts the pynput listener."""
        from localwispr.hotkeys import HotkeyListener

        listener = HotkeyListener()
        listener.start()

        assert listener.is_running
        mock_keyboard["listener_class"].assert_called_once()
        mock_keyboard["listener"].start.assert_called_once()

    def test_listener_start_twice_no_op(self, mock_config, mock_keyboard):
        """Test that calling start() twice doesn't create duplicate listeners."""
        from localwispr.hotkeys import HotkeyListener

        listener = HotkeyListener()
        listener.start()
        listener.start()

        # Should only be called once
        assert mock_keyboard["listener_class"].call_count == 1

    def test_listener_stop(self, mock_config, mock_keyboard):
        """Test that stop() stops the pynput listener."""
        from localwispr.hotkeys import HotkeyListener

        listener = HotkeyListener()
        listener.start()
        listener.stop()

        assert not listener.is_running
        mock_keyboard["listener"].stop.assert_called_once()

    def test_listener_stop_without_start(self, mock_config, mock_keyboard):
        """Test that stop() without start() is safe."""
        from localwispr.hotkeys import HotkeyListener

        listener = HotkeyListener()
        listener.stop()  # Should not raise

        assert not listener.is_running

    def test_listener_context_manager(self, mock_config, mock_keyboard):
        """Test that listener works as context manager."""
        from localwispr.hotkeys import HotkeyListener

        with HotkeyListener() as listener:
            assert listener.is_running

        assert not listener.is_running

    def test_on_transcription_complete(self, mock_config, mock_keyboard):
        """Test on_transcription_complete transitions from TRANSCRIBING to IDLE."""
        from localwispr.hotkeys import HotkeyListener, HotkeyState

        listener = HotkeyListener()
        listener._state = HotkeyState.TRANSCRIBING

        listener.on_transcription_complete()

        assert listener.state == HotkeyState.IDLE

    def test_on_transcription_complete_from_idle(self, mock_config, mock_keyboard):
        """Test on_transcription_complete does nothing from IDLE state."""
        from localwispr.hotkeys import HotkeyListener, HotkeyState

        listener = HotkeyListener()
        listener._state = HotkeyState.IDLE

        listener.on_transcription_complete()

        # Should remain IDLE
        assert listener.state == HotkeyState.IDLE


class TestModifierKeyDetection:
    """Tests for modifier key detection methods."""

    @pytest.fixture
    def mock_config(self, mocker):
        """Mock get_config."""
        config = {
            "hotkeys": {
                "modifiers": ["win", "ctrl", "shift"],
            }
        }
        mocker.patch("localwispr.hotkeys.get_config", return_value=config)
        return config

    @pytest.fixture
    def listener(self, mock_config, mocker):
        """Create a listener for testing."""
        mocker.patch("localwispr.hotkeys.keyboard.Listener")
        from localwispr.hotkeys import HotkeyListener

        return HotkeyListener()

    def test_is_win_key(self, listener):
        """Test _is_win_key detection."""
        from pynput.keyboard import Key

        assert listener._is_win_key(Key.cmd)
        assert listener._is_win_key(Key.cmd_l)
        assert listener._is_win_key(Key.cmd_r)
        assert not listener._is_win_key(Key.ctrl)

    def test_is_ctrl_key(self, listener):
        """Test _is_ctrl_key detection."""
        from pynput.keyboard import Key

        assert listener._is_ctrl_key(Key.ctrl)
        assert listener._is_ctrl_key(Key.ctrl_l)
        assert listener._is_ctrl_key(Key.ctrl_r)
        assert not listener._is_ctrl_key(Key.shift)

    def test_is_shift_key(self, listener):
        """Test _is_shift_key detection."""
        from pynput.keyboard import Key

        assert listener._is_shift_key(Key.shift)
        assert listener._is_shift_key(Key.shift_l)
        assert listener._is_shift_key(Key.shift_r)
        assert not listener._is_shift_key(Key.alt)

    def test_is_alt_key(self, listener):
        """Test _is_alt_key detection."""
        from pynput.keyboard import Key

        assert listener._is_alt_key(Key.alt)
        assert listener._is_alt_key(Key.alt_l)
        assert listener._is_alt_key(Key.alt_r)
        assert listener._is_alt_key(Key.alt_gr)
        assert not listener._is_alt_key(Key.ctrl)

    def test_is_modifier(self, listener):
        """Test _is_modifier returns True only for modifiers."""
        from pynput.keyboard import Key, KeyCode

        # Modifiers should return True
        assert listener._is_modifier(Key.cmd)
        assert listener._is_modifier(Key.ctrl)
        assert listener._is_modifier(Key.shift)
        assert listener._is_modifier(Key.alt)

        # Non-modifiers should return False
        assert not listener._is_modifier(Key.space)
        assert not listener._is_modifier(Key.enter)
        assert not listener._is_modifier(KeyCode.from_char("a"))


class TestChordDetection:
    """Tests for chord detection logic."""

    @pytest.fixture
    def mock_config(self, mocker):
        """Mock get_config."""
        config = {
            "hotkeys": {
                "modifiers": ["win", "ctrl", "shift"],
            }
        }
        mocker.patch("localwispr.hotkeys.get_config", return_value=config)
        return config

    @pytest.fixture
    def listener(self, mock_config, mocker):
        """Create a listener for testing."""
        mocker.patch("localwispr.hotkeys.keyboard.Listener")
        from localwispr.hotkeys import HotkeyListener

        return HotkeyListener()

    def test_is_chord_pressed_all_modifiers(self, listener):
        """Test chord detection when all required modifiers are pressed."""
        import time

        listener._win_pressed = True
        listener._ctrl_pressed = True
        listener._shift_pressed = True
        listener._alt_pressed = False

        # Set press times to be recent
        now = time.time()
        listener._win_press_time = now
        listener._ctrl_press_time = now
        listener._shift_press_time = now

        assert listener._is_chord_pressed()

    def test_is_chord_pressed_missing_modifier(self, listener):
        """Test chord detection when a modifier is missing."""
        listener._win_pressed = True
        listener._ctrl_pressed = True
        listener._shift_pressed = False  # Missing

        assert not listener._is_chord_pressed()

    def test_is_chord_pressed_exclusion_modifier(self, listener):
        """Test chord detection blocked by exclusion modifier."""
        import time

        listener._win_pressed = True
        listener._ctrl_pressed = True
        listener._shift_pressed = True
        listener._alt_pressed = True  # Exclusion modifier pressed

        now = time.time()
        listener._win_press_time = now
        listener._ctrl_press_time = now
        listener._shift_press_time = now

        # Alt is the exclusion modifier for Win+Ctrl+Shift chord
        assert not listener._is_chord_pressed()

    def test_is_mode_chord_pressed(self, listener):
        """Test mode cycle chord detection (Win+Ctrl+Alt)."""
        listener._win_pressed = True
        listener._ctrl_pressed = True
        listener._alt_pressed = True
        listener._shift_pressed = False

        assert listener._is_mode_chord_pressed()

    def test_is_mode_chord_pressed_with_shift(self, listener):
        """Test mode chord blocked when Shift is pressed."""
        listener._win_pressed = True
        listener._ctrl_pressed = True
        listener._alt_pressed = True
        listener._shift_pressed = True  # Should block mode chord

        assert not listener._is_mode_chord_pressed()


class TestStateTransitions:
    """Tests for state machine transitions."""

    @pytest.fixture
    def mock_config(self, mocker):
        """Mock get_config."""
        config = {
            "hotkeys": {
                "modifiers": ["win", "ctrl", "shift"],
            }
        }
        mocker.patch("localwispr.hotkeys.get_config", return_value=config)
        return config

    @pytest.fixture
    def listener(self, mock_config, mocker):
        """Create a listener for testing."""
        mocker.patch("localwispr.hotkeys.keyboard.Listener")
        from localwispr.hotkeys import HotkeyListener

        return HotkeyListener()

    def test_transition_to_recording(self, listener):
        """Test transition from IDLE to RECORDING."""
        from localwispr.hotkeys import HotkeyState

        callback_called = []
        listener._on_record_start = lambda: callback_called.append(True)

        listener._transition_to_recording()

        assert listener.state == HotkeyState.RECORDING
        assert callback_called == [True]

    def test_transition_to_recording_not_from_idle(self, listener):
        """Test that transition only works from IDLE."""
        from localwispr.hotkeys import HotkeyState

        listener._state = HotkeyState.RECORDING

        listener._transition_to_recording()

        # Should remain RECORDING, not create duplicate transition
        assert listener.state == HotkeyState.RECORDING

    def test_transition_to_transcribing(self, listener):
        """Test transition from RECORDING to TRANSCRIBING."""
        from localwispr.hotkeys import HotkeyState

        listener._state = HotkeyState.RECORDING
        callback_called = []
        listener._on_record_stop = lambda: callback_called.append(True)

        listener._transition_to_transcribing()

        assert listener.state == HotkeyState.TRANSCRIBING
        assert callback_called == [True]

    def test_transition_to_transcribing_not_from_recording(self, listener):
        """Test that transition only works from RECORDING."""
        from localwispr.hotkeys import HotkeyState

        listener._state = HotkeyState.IDLE

        listener._transition_to_transcribing()

        # Should remain IDLE
        assert listener.state == HotkeyState.IDLE

    def test_toggle_recording_from_idle(self, listener):
        """Test toggle from IDLE starts recording."""
        from localwispr.hotkeys import HotkeyMode, HotkeyState

        listener._mode = HotkeyMode.TOGGLE
        callback_called = []
        listener._on_record_start = lambda: callback_called.append("start")
        listener._on_record_stop = lambda: callback_called.append("stop")

        listener._toggle_recording()

        assert listener.state == HotkeyState.RECORDING
        assert callback_called == ["start"]

    def test_toggle_recording_from_recording(self, listener):
        """Test toggle from RECORDING stops recording."""
        from localwispr.hotkeys import HotkeyMode, HotkeyState

        listener._mode = HotkeyMode.TOGGLE
        listener._state = HotkeyState.RECORDING
        callback_called = []
        listener._on_record_start = lambda: callback_called.append("start")
        listener._on_record_stop = lambda: callback_called.append("stop")

        listener._toggle_recording()

        assert listener.state == HotkeyState.TRANSCRIBING
        assert callback_called == ["stop"]

    def test_toggle_recording_from_transcribing(self, listener):
        """Test toggle does nothing during transcription."""
        from localwispr.hotkeys import HotkeyMode, HotkeyState

        listener._mode = HotkeyMode.TOGGLE
        listener._state = HotkeyState.TRANSCRIBING
        callback_called = []
        listener._on_record_start = lambda: callback_called.append("start")
        listener._on_record_stop = lambda: callback_called.append("stop")

        listener._toggle_recording()

        # Should remain TRANSCRIBING
        assert listener.state == HotkeyState.TRANSCRIBING
        assert callback_called == []


class TestValidModifiers:
    """Tests for VALID_MODIFIERS constant."""

    def test_valid_modifiers_contains_expected(self):
        """Test that VALID_MODIFIERS has all expected values."""
        from localwispr.hotkeys import VALID_MODIFIERS

        assert "win" in VALID_MODIFIERS
        assert "ctrl" in VALID_MODIFIERS
        assert "shift" in VALID_MODIFIERS
        assert "alt" in VALID_MODIFIERS
        assert len(VALID_MODIFIERS) == 4
