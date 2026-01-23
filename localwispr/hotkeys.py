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

import logging
import threading
import time
from enum import Enum, auto
from typing import Callable

logger = logging.getLogger(__name__)

from pynput import keyboard
from pynput.keyboard import Key

from localwispr.config import get_config


# Valid modifier names that can be used in config
VALID_MODIFIERS = {"win", "ctrl", "shift", "alt"}


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
    """Global hotkey listener for configurable chord detection.

    Listens for a configurable modifier key combination (from config.toml)
    to control recording. Supports both push-to-talk (hold to record) and
    toggle (press to start/stop) modes.

    Default chord: Win+Ctrl+Shift (configurable via hotkeys.modifiers in config)
    Mode cycle chord: Win+Ctrl+Alt (hardcoded)

    The recording chord is read from config["hotkeys"]["modifiers"], allowing
    different builds to use different hotkeys (e.g., stable uses Win+Ctrl+Shift,
    test uses Ctrl+Alt+Shift).

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
        ...     on_mode_cycle=lambda: print("Mode cycled!"),
        ...     mode=HotkeyMode.PUSH_TO_TALK,
        ... )
        >>> listener.start()
        >>> # Press configured chord to record (default: Win+Ctrl+Shift)
        >>> # Press Win+Ctrl+Alt to cycle modes
        >>> listener.stop()
    """

    # Toggle mode debounce time in seconds
    TOGGLE_DEBOUNCE_MS = 100

    # Mode cycle debounce time in milliseconds
    MODE_CYCLE_DEBOUNCE_MS = 300

    # Startup grace period - ignore events during Windows hook stabilization
    _STARTUP_GRACE_MS = 150

    # Chord window - all chord keys must be pressed within this window (ms)
    _CHORD_WINDOW_MS = 500

    def __init__(
        self,
        on_record_start: Callable[[], None] | None = None,
        on_record_stop: Callable[[], None] | None = None,
        on_mode_cycle: Callable[[], None] | None = None,
        mode: HotkeyMode = HotkeyMode.PUSH_TO_TALK,
    ) -> None:
        """Initialize the hotkey listener.

        Args:
            on_record_start: Callback invoked when recording should start.
            on_record_stop: Callback invoked when recording should stop.
            on_mode_cycle: Callback invoked when mode cycle chord is pressed.
            mode: Hotkey activation mode (push-to-talk or toggle).
        """
        self._on_record_start = on_record_start
        self._on_record_stop = on_record_stop
        self._on_mode_cycle = on_mode_cycle
        self._mode = mode

        # Load recording chord modifiers from config
        config = get_config()
        config_modifiers = config["hotkeys"].get("modifiers", ["win", "ctrl", "shift"])
        self._chord_modifiers: set[str] = set()
        for mod in config_modifiers:
            mod_lower = mod.lower()
            if mod_lower in VALID_MODIFIERS:
                self._chord_modifiers.add(mod_lower)
            else:
                logger.warning("hotkey: ignoring invalid modifier '%s'", mod)

        # Default to Win+Ctrl+Shift if no valid modifiers
        if len(self._chord_modifiers) < 2:
            logger.warning("hotkey: insufficient modifiers, using default Win+Ctrl+Shift")
            self._chord_modifiers = {"win", "ctrl", "shift"}

        # Determine exclusion modifier (the one NOT in the chord that blocks activation)
        # This prevents accidental triggers when pressing 4 modifiers
        all_mods = {"win", "ctrl", "shift", "alt"}
        excluded = all_mods - self._chord_modifiers
        self._exclusion_modifier: str | None = excluded.pop() if len(excluded) == 1 else None

        logger.info(
            "hotkey: configured chord=%s, exclusion=%s",
            "+".join(sorted(self._chord_modifiers)),
            self._exclusion_modifier,
        )

        # State machine
        self._state = HotkeyState.IDLE
        self._state_lock = threading.Lock()

        # Modifier key tracking (only these keys are tracked for privacy)
        self._win_pressed = False
        self._ctrl_pressed = False
        self._shift_pressed = False
        self._alt_pressed = False
        self._modifier_lock = threading.Lock()

        # Chord state tracking
        self._chord_active = False
        self._mode_chord_active = False
        self._last_toggle_time: float = 0.0
        self._last_mode_cycle_time: float = 0.0

        # Timestamp tracking for "clean press" detection
        self._win_press_time: float = 0.0
        self._ctrl_press_time: float = 0.0
        self._shift_press_time: float = 0.0
        self._alt_press_time: float = 0.0

        # Listener
        self._listener: keyboard.Listener | None = None
        self._running = False
        self._start_time: float | None = None

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

    def _is_alt_key(self, key: Key | keyboard.KeyCode) -> bool:
        """Check if a key is an Alt key.

        Args:
            key: The key to check.

        Returns:
            True if the key is Alt (any variant).
        """
        return key in (Key.alt, Key.alt_l, Key.alt_r, Key.alt_gr)

    def _is_modifier(self, key: Key | keyboard.KeyCode) -> bool:
        """Check if a key is a tracked modifier key.

        Privacy (V1-N03):
            This method returns True ONLY for Win, Ctrl, Shift, and Alt keys.
            All other keys (alphanumeric, punctuation, function keys, etc.)
            return False and are ignored by the listener.

        Args:
            key: The key to check.

        Returns:
            True only if the key is Win, Ctrl, Shift, or Alt.
        """
        return (
            self._is_win_key(key)
            or self._is_ctrl_key(key)
            or self._is_shift_key(key)
            or self._is_alt_key(key)
        )

    def _is_modifier_pressed(self, modifier: str) -> bool:
        """Check if a specific modifier is pressed.

        Args:
            modifier: One of "win", "ctrl", "shift", "alt".

        Returns:
            True if the modifier is currently pressed.
        """
        if modifier == "win":
            return self._win_pressed
        elif modifier == "ctrl":
            return self._ctrl_pressed
        elif modifier == "shift":
            return self._shift_pressed
        elif modifier == "alt":
            return self._alt_pressed
        return False

    def _is_chord_pressed(self) -> bool:
        """Check if the configured recording chord is pressed.

        The chord modifiers are read from config (e.g., ["win", "ctrl", "shift"]
        or ["ctrl", "alt", "shift"]). An exclusion modifier blocks activation
        to prevent accidental triggers when all 4 modifiers are pressed.

        Returns:
            True if all configured modifiers are pressed AND exclusion is NOT.
        """
        with self._modifier_lock:
            # Check all required modifiers are pressed
            for mod in self._chord_modifiers:
                if not self._is_modifier_pressed(mod):
                    return False

            # Check exclusion modifier is NOT pressed (mutual exclusion)
            if self._exclusion_modifier and self._is_modifier_pressed(self._exclusion_modifier):
                return False

            return True

    def _get_modifier_press_time(self, modifier: str) -> float:
        """Get the press timestamp for a specific modifier.

        Args:
            modifier: One of "win", "ctrl", "shift", "alt".

        Returns:
            Timestamp when the modifier was last pressed.
        """
        if modifier == "win":
            return self._win_press_time
        elif modifier == "ctrl":
            return self._ctrl_press_time
        elif modifier == "shift":
            return self._shift_press_time
        elif modifier == "alt":
            return self._alt_press_time
        return 0.0

    def _is_chord_fresh(self) -> bool:
        """Check if all chord keys were pressed recently (within 500ms window).

        This prevents "stale" modifier state from triggering the chord.
        All configured chord keys must have been pressed within _CHORD_WINDOW_MS
        of each other and within _CHORD_WINDOW_MS of the current time.

        Returns:
            True if all chord keys were pressed within the time window.
        """
        with self._modifier_lock:
            now = time.time()
            window_sec = self._CHORD_WINDOW_MS / 1000.0
            times = [self._get_modifier_press_time(mod) for mod in self._chord_modifiers]
            oldest = min(times)
            newest = max(times)
            # All presses must be within window of each other AND recent
            return (now - oldest) < window_sec and (newest - oldest) < window_sec

    def _is_mode_chord_pressed(self) -> bool:
        """Check if the mode cycle chord (Win+Ctrl+Alt) is pressed.

        Returns:
            True if Win+Ctrl+Alt are pressed (but NOT Shift).
        """
        with self._modifier_lock:
            return (
                self._win_pressed
                and self._ctrl_pressed
                and self._alt_pressed
                and not self._shift_pressed
            )

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

        # Update modifier state and record press timestamps
        now = time.time()
        with self._modifier_lock:
            if self._is_win_key(key):
                self._win_pressed = True
                self._win_press_time = now
            elif self._is_ctrl_key(key):
                self._ctrl_pressed = True
                self._ctrl_press_time = now
            elif self._is_shift_key(key):
                self._shift_pressed = True
                self._shift_press_time = now
            elif self._is_alt_key(key):
                self._alt_pressed = True
                self._alt_press_time = now

        # Check for recording chord activation (Win+Ctrl+Shift)
        # Must pass all checks: chord pressed, keys are fresh, not already active
        if self._is_chord_pressed() and self._is_chord_fresh() and not self._chord_active:
            # Ignore during startup grace period (Windows hook stabilization)
            if self._start_time and (time.time() - self._start_time) < (self._STARTUP_GRACE_MS / 1000):
                logger.debug("hotkey: ignoring chord during startup grace period")
                return
            self._chord_active = True
            self._on_chord_down()

        # Check for mode cycle chord activation (Win+Ctrl+Alt, not Shift)
        if self._is_mode_chord_pressed() and not self._mode_chord_active:
            self._mode_chord_active = True
            self._on_mode_chord_down()

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
        was_mode_chord_active = self._mode_chord_active
        chord_pressed_before = self._is_chord_pressed()
        mode_chord_pressed_before = self._is_mode_chord_pressed()

        # Update modifier state
        with self._modifier_lock:
            if self._is_win_key(key):
                self._win_pressed = False
            elif self._is_ctrl_key(key):
                self._ctrl_pressed = False
            elif self._is_shift_key(key):
                self._shift_pressed = False
            elif self._is_alt_key(key):
                self._alt_pressed = False

        # Check if recording chord was just released
        if was_chord_active and chord_pressed_before and not self._is_chord_pressed():
            self._chord_active = False
            self._on_chord_up()

        # Check if mode chord was just released
        if was_mode_chord_active and mode_chord_pressed_before and not self._is_mode_chord_pressed():
            self._mode_chord_active = False
            # Mode cycle is triggered on press, not release, so nothing to do here

    def _on_chord_down(self) -> None:
        """Handle recording chord press event (Win+Ctrl+Shift).

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

    def _on_mode_chord_down(self) -> None:
        """Handle mode cycle chord press event (Win+Ctrl+Alt).

        Cycles through transcription modes with debounce.
        """
        current_time = time.time() * 1000  # Convert to milliseconds
        if current_time - self._last_mode_cycle_time >= self.MODE_CYCLE_DEBOUNCE_MS:
            self._last_mode_cycle_time = current_time
            if self._on_mode_cycle is not None:
                self._on_mode_cycle()

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
            self._start_time = time.time()
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
            self._alt_pressed = False
            self._chord_active = False
            self._mode_chord_active = False

    def __enter__(self) -> HotkeyListener:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
