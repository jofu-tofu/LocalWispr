"""Global hotkey listener module for LocalWispr.

This module implements global hotkey detection for controlling speech-to-text
recording. It uses pynput for keyboard monitoring with strict privacy controls:
only modifier keys (Win, Ctrl, Shift) are tracked - no alphanumeric keys.

Privacy Note (V1-N03):
    This module ONLY tracks modifier key states. The on_press and on_release
    callbacks explicitly ignore all non-modifier keys. No keystroke logging
    of alphanumeric or other keys occurs.
"""

from __future__ import annotations

import threading
import time
from enum import Enum, auto
from typing import Callable

from pynput import keyboard
from pynput.keyboard import Key


class HotkeyListenerError(Exception):
    """Exception raised when the hotkey listener fails to start.

    This typically occurs due to:
    - Antivirus software blocking keyboard hooks
    - Group Policy restrictions on input monitoring
    - Insufficient permissions (may need to run as administrator)
    """

    def __init__(self, message: str, suggestion: str | None = None) -> None:
        """Initialize the error.

        Args:
            message: Error description.
            suggestion: Optional suggestion for resolving the issue.
        """
        if suggestion is None:
            suggestion = (
                "Try: 1) Check antivirus/security software settings, "
                "2) Verify no Group Policy restrictions, "
                "3) Run as administrator"
            )
        full_message = f"{message}. {suggestion}"
        super().__init__(full_message)
        self.suggestion = suggestion


class HotkeyState(Enum):
    """State machine states for hotkey-controlled recording.

    States:
        IDLE: Not recording, waiting for hotkey press.
        RECORDING: Actively recording audio.
        TRANSCRIBING: Recording stopped, transcription in progress.
    """

    IDLE = auto()
    RECORDING = auto()
    TRANSCRIBING = auto()


class HotkeyMode(Enum):
    """Hotkey activation modes.

    Modes:
        PUSH_TO_TALK: Hold chord to record, release to stop and transcribe.
        TOGGLE: Press chord to start, press again to stop and transcribe.
    """

    PUSH_TO_TALK = auto()
    TOGGLE = auto()


class HotkeyListener:
    """Global hotkey listener for Win+Ctrl+Shift chord detection.

    Listens for the Win+Ctrl+Shift key combination to control recording.
    Supports both push-to-talk (hold to record) and toggle (press to start/stop)
    modes.

    Privacy (V1-N03):
        This listener ONLY tracks modifier key states. All non-modifier keys
        are explicitly ignored in the on_press and on_release callbacks.
        No keystroke logging of alphanumeric keys occurs.

    Thread Safety:
        All state transitions are protected by a threading.Lock to ensure
        atomic transitions in the state machine.

    Example:
        >>> listener = HotkeyListener(
        ...     on_record_start=lambda: print("Recording..."),
        ...     on_record_stop=lambda: print("Stopped"),
        ...     mode=HotkeyMode.PUSH_TO_TALK,
        ... )
        >>> listener.start()
        >>> # Press Win+Ctrl+Shift to record
        >>> listener.stop()
    """

    # Toggle mode debounce time in seconds
    TOGGLE_DEBOUNCE_MS = 100

    def __init__(
        self,
        on_record_start: Callable[[], None] | None = None,
        on_record_stop: Callable[[], None] | None = None,
        mode: HotkeyMode = HotkeyMode.PUSH_TO_TALK,
    ) -> None:
        """Initialize the hotkey listener.

        Args:
            on_record_start: Callback invoked when recording should start.
            on_record_stop: Callback invoked when recording should stop.
            mode: Hotkey activation mode (push-to-talk or toggle).
        """
        self._on_record_start = on_record_start
        self._on_record_stop = on_record_stop
        self._mode = mode

        # State machine
        self._state = HotkeyState.IDLE
        self._state_lock = threading.Lock()

        # Modifier key tracking (only these keys are tracked for privacy)
        self._win_pressed = False
        self._ctrl_pressed = False
        self._shift_pressed = False
        self._modifier_lock = threading.Lock()

        # Chord state tracking
        self._chord_active = False
        self._last_toggle_time: float = 0.0

        # Listener
        self._listener: keyboard.Listener | None = None
        self._running = False

    @property
    def state(self) -> HotkeyState:
        """Get the current state.

        Returns:
            Current HotkeyState.
        """
        with self._state_lock:
            return self._state

    @property
    def mode(self) -> HotkeyMode:
        """Get the current hotkey mode.

        Returns:
            Current HotkeyMode.
        """
        return self._mode

    @property
    def is_running(self) -> bool:
        """Check if the listener is running.

        Returns:
            True if the listener is active.
        """
        return self._running

    def _is_win_key(self, key: Key | keyboard.KeyCode) -> bool:
        """Check if a key is a Windows/Command key.

        Args:
            key: The key to check.

        Returns:
            True if the key is Win/Cmd (any variant).
        """
        return key in (Key.cmd, Key.cmd_l, Key.cmd_r)

    def _is_ctrl_key(self, key: Key | keyboard.KeyCode) -> bool:
        """Check if a key is a Control key.

        Args:
            key: The key to check.

        Returns:
            True if the key is Ctrl (any variant).
        """
        return key in (Key.ctrl, Key.ctrl_l, Key.ctrl_r)

    def _is_shift_key(self, key: Key | keyboard.KeyCode) -> bool:
        """Check if a key is a Shift key.

        Args:
            key: The key to check.

        Returns:
            True if the key is Shift (any variant).
        """
        return key in (Key.shift, Key.shift_l, Key.shift_r)

    def _is_modifier(self, key: Key | keyboard.KeyCode) -> bool:
        """Check if a key is a tracked modifier key.

        Privacy (V1-N03):
            This method returns True ONLY for Win, Ctrl, and Shift keys.
            All other keys (alphanumeric, punctuation, function keys, etc.)
            return False and are ignored by the listener.

        Args:
            key: The key to check.

        Returns:
            True only if the key is Win, Ctrl, or Shift.
        """
        return self._is_win_key(key) or self._is_ctrl_key(key) or self._is_shift_key(key)

    def _is_chord_pressed(self) -> bool:
        """Check if the full Win+Ctrl+Shift chord is pressed.

        Returns:
            True if all three modifier types are currently pressed.
        """
        with self._modifier_lock:
            return self._win_pressed and self._ctrl_pressed and self._shift_pressed

    def _on_key_press(self, key: Key | keyboard.KeyCode | None) -> None:
        """Handle key press events.

        Privacy (V1-N03):
            This callback ONLY processes modifier keys. Non-modifier keys
            are immediately ignored without any logging or processing.

        Args:
            key: The key that was pressed.
        """
        if key is None:
            return

        # PRIVACY: Only process modifier keys, ignore all others
        if not self._is_modifier(key):
            return

        # Update modifier state
        with self._modifier_lock:
            if self._is_win_key(key):
                self._win_pressed = True
            elif self._is_ctrl_key(key):
                self._ctrl_pressed = True
            elif self._is_shift_key(key):
                self._shift_pressed = True

        # Check for chord activation
        if self._is_chord_pressed() and not self._chord_active:
            self._chord_active = True
            self._on_chord_down()

    def _on_key_release(self, key: Key | keyboard.KeyCode | None) -> None:
        """Handle key release events.

        Privacy (V1-N03):
            This callback ONLY processes modifier keys. Non-modifier keys
            are immediately ignored without any logging or processing.

        Args:
            key: The key that was released.
        """
        if key is None:
            return

        # PRIVACY: Only process modifier keys, ignore all others
        if not self._is_modifier(key):
            return

        # Check for chord deactivation before updating state
        was_chord_active = self._chord_active
        chord_pressed_before = self._is_chord_pressed()

        # Update modifier state
        with self._modifier_lock:
            if self._is_win_key(key):
                self._win_pressed = False
            elif self._is_ctrl_key(key):
                self._ctrl_pressed = False
            elif self._is_shift_key(key):
                self._shift_pressed = False

        # Check if chord was just released
        if was_chord_active and chord_pressed_before and not self._is_chord_pressed():
            self._chord_active = False
            self._on_chord_up()

    def _on_chord_down(self) -> None:
        """Handle chord press event.

        In push-to-talk mode: Start recording.
        In toggle mode: Toggle recording state (with debounce).
        """
        if self._mode == HotkeyMode.PUSH_TO_TALK:
            self._transition_to_recording()
        else:
            # Toggle mode with debounce
            current_time = time.time() * 1000  # Convert to milliseconds
            if current_time - self._last_toggle_time >= self.TOGGLE_DEBOUNCE_MS:
                self._last_toggle_time = current_time
                self._toggle_recording()

    def _on_chord_up(self) -> None:
        """Handle chord release event.

        In push-to-talk mode: Stop recording and start transcription.
        In toggle mode: No action (toggle only responds to press).
        """
        if self._mode == HotkeyMode.PUSH_TO_TALK:
            self._transition_to_transcribing()

    def _transition_to_recording(self) -> None:
        """Transition from IDLE to RECORDING state.

        Thread-safe state transition that invokes the on_record_start callback.
        """
        with self._state_lock:
            if self._state != HotkeyState.IDLE:
                return
            self._state = HotkeyState.RECORDING

        # Invoke callback outside lock to prevent deadlocks
        if self._on_record_start is not None:
            self._on_record_start()

    def _transition_to_transcribing(self) -> None:
        """Transition from RECORDING to TRANSCRIBING state.

        Thread-safe state transition that invokes the on_record_stop callback.
        """
        with self._state_lock:
            if self._state != HotkeyState.RECORDING:
                return
            self._state = HotkeyState.TRANSCRIBING

        # Invoke callback outside lock to prevent deadlocks
        if self._on_record_stop is not None:
            self._on_record_stop()

    def _toggle_recording(self) -> None:
        """Toggle between IDLE and RECORDING states (for toggle mode).

        Thread-safe state toggle.
        """
        with self._state_lock:
            if self._state == HotkeyState.IDLE:
                self._state = HotkeyState.RECORDING
                callback = self._on_record_start
            elif self._state == HotkeyState.RECORDING:
                self._state = HotkeyState.TRANSCRIBING
                callback = self._on_record_stop
            else:
                # TRANSCRIBING state - don't allow toggle during transcription
                return

        # Invoke callback outside lock to prevent deadlocks
        if callback is not None:
            callback()

    def on_transcription_complete(self) -> None:
        """Signal that transcription has completed.

        Transitions from TRANSCRIBING back to IDLE state.
        Should be called by the transcription pipeline when done.
        """
        with self._state_lock:
            if self._state == HotkeyState.TRANSCRIBING:
                self._state = HotkeyState.IDLE

    def start(self) -> None:
        """Start the hotkey listener.

        Begins listening for keyboard events in a background thread.

        Raises:
            HotkeyListenerError: If the listener fails to start (e.g., due to
                antivirus blocking, policy restrictions, or permission issues).
        """
        if self._running:
            return

        try:
            self._listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release,
            )
            self._listener.start()
            self._running = True
        except OSError as e:
            raise HotkeyListenerError(
                f"Failed to start keyboard listener: {e}",
                suggestion=(
                    "This may be caused by antivirus software, Group Policy "
                    "restrictions, or insufficient permissions. Try running "
                    "as administrator or check your security software settings."
                ),
            ) from e
        except Exception as e:
            raise HotkeyListenerError(
                f"Unexpected error starting keyboard listener: {e}",
            ) from e

    def stop(self) -> None:
        """Stop the hotkey listener.

        Stops listening for keyboard events and cleans up resources.
        """
        if not self._running:
            return

        self._running = False

        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass  # Ignore errors during cleanup
            self._listener = None

        # Reset modifier states
        with self._modifier_lock:
            self._win_pressed = False
            self._ctrl_pressed = False
            self._shift_pressed = False
            self._chord_active = False

    def __enter__(self) -> HotkeyListener:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
