"""System tray application module for LocalWispr.

This module implements the system tray UI using pystray, providing:
- Custom "L through sound wave" branded icon with state colors
- Tray menu with recording mode, transcription modes, about, and exit options
- Thread-safe state management for external callers
- Integration with hotkey listener in background thread
- Mode cycling via Win+Ctrl+Alt hotkey
- Notification popups for recording state feedback
- Clean shutdown sequence to prevent orphaned threads

Logging Policy:
    This module uses structured logging with an explicit field whitelist.
    NEVER logged: transcription content, audio data, exception payloads with user text.
    Only logged: timestamp, level, event_type, duration_ms, success/fail status.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from PIL import Image, ImageDraw

# Import build variant detection
from localwispr import IS_TEST_BUILD

# App display name varies by build variant
APP_DISPLAY_NAME = "LocalWispr [TEST]" if IS_TEST_BUILD else "LocalWispr"

if TYPE_CHECKING:
    import pystray


# Configure module logger with rotation
def _setup_logger() -> logging.Logger:
    """Set up the LocalWispr logger with rotation.

    Test builds use a separate log file (localwispr-test.log) to avoid
    conflicts when running both versions simultaneously.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("localwispr")

    # Only set up handlers if not already configured
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # File handler with rotation (10MB max, keep 2 backups)
        # Test builds use separate log file
        log_filename = "localwispr-test.log" if IS_TEST_BUILD else "localwispr.log"
        log_file = Path.cwd() / log_filename
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=2,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)

        # Structured format - explicit fields only
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = _setup_logger()


class TrayState(Enum):
    """Visual states for the tray icon.

    States:
        IDLE: Default state - ready for recording.
        RECORDING: Actively recording audio.
        TRANSCRIBING: Processing audio, generating transcription.
    """

    IDLE = auto()
    RECORDING = auto()
    TRANSCRIBING = auto()


# Icon colors for each state (STABLE version - blue theme)
STATE_COLORS = {
    TrayState.IDLE: {
        "wave": "#4A90D9",     # Blue wave
        "letter": "#FFFFFF",   # White L
        "background": None,    # Transparent
    },
    TrayState.RECORDING: {
        "wave": "#DC3545",     # Red wave
        "letter": "#FFFFFF",   # White L
        "background": None,    # Transparent
    },
    TrayState.TRANSCRIBING: {
        "wave": "#FFC107",     # Amber/yellow wave
        "letter": "#FFFFFF",   # White L
        "background": None,    # Transparent
    },
}

# Icon colors for TEST version (orange theme for visual differentiation)
TEST_STATE_COLORS = {
    TrayState.IDLE: {
        "wave": "#FF8C00",     # Orange wave (dark orange)
        "letter": "#FFFFFF",   # White L
        "background": None,    # Transparent
    },
    TrayState.RECORDING: {
        "wave": "#DC3545",     # Red wave (same as stable)
        "letter": "#FFFFFF",   # White L
        "background": None,    # Transparent
    },
    TrayState.TRANSCRIBING: {
        "wave": "#FFC107",     # Amber/yellow wave (same as stable)
        "letter": "#FFFFFF",   # White L
        "background": None,    # Transparent
    },
}


def create_icon_image(state: TrayState = TrayState.IDLE, size: int = 64) -> Image.Image:
    """Create the LocalWispr icon: "L" through a sound wave.

    The icon shows a stylized letter "L" with sound wave lines emanating
    from the right side, representing speech-to-text functionality.

    Test builds use orange wave colors for visual differentiation.

    Args:
        state: Current tray state for color selection.
        size: Icon size in pixels (square).

    Returns:
        PIL Image of the icon.
    """
    color_map = TEST_STATE_COLORS if IS_TEST_BUILD else STATE_COLORS
    colors = color_map[state]

    # Create transparent RGBA image
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Calculate scaling
    scale = size / 64

    # Draw the "L" letter (simplified geometric shape)
    letter_color = colors["letter"]
    # Vertical bar of L
    l_left = int(12 * scale)
    l_top = int(10 * scale)
    l_width = int(8 * scale)
    l_height = int(44 * scale)
    draw.rectangle(
        [l_left, l_top, l_left + l_width, l_top + l_height],
        fill=letter_color,
    )
    # Horizontal bar of L
    h_left = int(12 * scale)
    h_top = int(46 * scale)
    h_width = int(22 * scale)
    h_height = int(8 * scale)
    draw.rectangle(
        [h_left, h_top, h_left + h_width, h_top + h_height],
        fill=letter_color,
    )

    # Draw sound waves (3 arcs on the right side)
    wave_color = colors["wave"]
    center_x = int(40 * scale)
    center_y = int(32 * scale)

    # Draw 3 concentric arc segments representing sound waves
    for i, radius in enumerate([int(12 * scale), int(20 * scale), int(28 * scale)]):
        # Vary line width based on wave position
        line_width = max(2, int(3 * scale) - i)
        bbox = [
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        ]
        # Draw arc from -45 to 45 degrees (right-facing)
        draw.arc(bbox, start=-45, end=45, fill=wave_color, width=line_width)

    return image


class TrayApp:
    """System tray application for LocalWispr.

    Manages the system tray icon, menu, and coordinates with the hotkey
    listener for recording functionality. Integrates notifications and
    the full transcription pipeline.

    Thread Safety:
        State updates from external threads should use update_state() which
        puts updates on a queue for processing in the tray event loop.

    Shutdown Sequence:
        1. User selects Exit from menu
        2. _on_exit() sets stop event
        3. Background threads receive stop signal
        4. Threads joined with timeout
        5. Tray icon stopped

    Example:
        >>> app = TrayApp()
        >>> app.run()  # Blocks until exit
    """

    # Queue poll interval in seconds
    QUEUE_POLL_INTERVAL = 0.1

    # Thread join timeout in seconds
    THREAD_JOIN_TIMEOUT = 2.0

    def __init__(
        self,
        on_state_change: Callable[[TrayState], None] | None = None,
    ) -> None:
        """Initialize the tray application.

        Args:
            on_state_change: Optional callback when state changes.
        """
        self._on_state_change = on_state_change

        # Current state
        self._state = TrayState.IDLE
        self._state_lock = threading.Lock()

        # Thread-safe queue for state updates from external threads
        self._update_queue: queue.Queue[TrayState] = queue.Queue()

        # Stop event for coordinating shutdown
        self._stop_event = threading.Event()

        # pystray icon reference
        self._icon: "pystray.Icon | None" = None

        # Background threads to join on shutdown
        self._background_threads: list[threading.Thread] = []

        # Recording pipeline components (lazy loaded)
        self._recorder = None
        self._transcriber = None
        self._detector = None
        self._hotkey_listener = None

        # Lock for thread-safe recorder access
        self._recorder_lock = threading.Lock()

        # Lock for thread-safe transcriber access
        self._transcriber_lock = threading.Lock()

        # Volume mute state tracking
        self._was_muted_before_recording = False

        # Model preloading state
        self._model_preload_complete = threading.Event()
        self._model_preload_error: Exception | None = None

        # Background transcription processing
        self._transcription_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="transcribe",
        )
        self._current_generation: int = 0
        self._generation_lock = threading.Lock()

        # Floating overlay widget for visual feedback
        from localwispr.overlay import OverlayWidget

        self._overlay = OverlayWidget(
            audio_level_callback=self._get_current_audio_level,
            model_name_callback=self._get_model_name,
        )

        # Initialize mode manager with notification callback
        from localwispr.modes import get_mode_manager

        self._mode_manager = get_mode_manager(
            auto_reset=False,
            on_mode_change=self._on_mode_changed,
        )

        logger.info("tray_app: initialized")

    @property
    def state(self) -> TrayState:
        """Get the current tray state.

        Returns:
            Current TrayState.
        """
        with self._state_lock:
            return self._state

    @property
    def stop_event(self) -> threading.Event:
        """Get the stop event for background threads.

        Returns:
            Threading Event that signals shutdown.
        """
        return self._stop_event

    def register_background_thread(self, thread: threading.Thread) -> None:
        """Register a background thread to be joined on shutdown.

        Args:
            thread: Thread to join during shutdown sequence.
        """
        self._background_threads.append(thread)

    def update_state(self, new_state: TrayState) -> None:
        """Update the tray state from any thread.

        Thread-safe method to update state. Puts the update on a queue
        that is processed in the tray event loop.

        Args:
            new_state: New state to transition to.
        """
        self._update_queue.put(new_state)

    def _set_state(self, new_state: TrayState) -> None:
        """Internal method to set state and update icon.

        Should only be called from the main tray thread.

        Args:
            new_state: New state to set.
        """
        with self._state_lock:
            if self._state == new_state:
                return
            old_state = self._state
            self._state = new_state

        logger.debug("tray_state: changed, from=%s, to=%s", old_state.name, new_state.name)

        # Update icon image
        if self._icon is not None:
            self._icon.icon = create_icon_image(new_state)

        # Invoke callback
        if self._on_state_change is not None:
            try:
                self._on_state_change(new_state)
            except Exception:
                pass  # Don't let callback errors crash the app

    def _process_queue(self) -> None:
        """Process pending state updates from the queue.

        Non-blocking - processes all available updates.
        """
        while True:
            try:
                new_state = self._update_queue.get_nowait()
                self._set_state(new_state)
            except queue.Empty:
                break

    def _create_menu(self) -> "pystray.Menu":
        """Create the tray context menu.

        Returns:
            pystray Menu object.
        """
        import pystray

        from localwispr.modes import ModeType, get_all_modes

        # Build transcription modes submenu
        mode_items = []
        for mode in get_all_modes():
            mode_items.append(
                pystray.MenuItem(
                    f"{mode.icon} {mode.name}",
                    self._make_mode_callback(mode.mode_type),
                    checked=self._make_mode_check(mode.mode_type),
                    radio=True,
                )
            )

        return pystray.Menu(
            pystray.MenuItem(
                "Transcription Mode",
                pystray.Menu(*mode_items),
            ),
            pystray.MenuItem(
                "Recording Mode",
                pystray.Menu(
                    pystray.MenuItem(
                        "Push-to-Talk",
                        self._on_mode_ptt,
                        checked=lambda item: self._is_mode_ptt(),
                        radio=True,
                    ),
                    pystray.MenuItem(
                        "Toggle",
                        self._on_mode_toggle,
                        checked=lambda item: not self._is_mode_ptt(),
                        radio=True,
                    ),
                ),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Settings", self._on_settings),
            pystray.MenuItem(f"About {APP_DISPLAY_NAME}", self._on_about),
            pystray.MenuItem("Exit", self._on_exit),
        )

    def _make_mode_callback(self, mode_type: "ModeType"):
        """Create a callback for setting a specific transcription mode.

        Args:
            mode_type: The mode type to set.

        Returns:
            Callback function.
        """
        def callback(icon, item):
            from localwispr.modes import set_mode

            set_mode(mode_type)

        return callback

    def _make_mode_check(self, mode_type: "ModeType"):
        """Create a check function for a specific transcription mode.

        Args:
            mode_type: The mode type to check.

        Returns:
            Check function.
        """
        def check(item):
            return self._mode_manager.current_mode_type == mode_type

        return check

    def _is_mode_ptt(self) -> bool:
        """Check if current mode is push-to-talk.

        Returns:
            True if push-to-talk mode.
        """
        from localwispr.config import get_config

        config = get_config()
        return config["hotkeys"]["mode"] == "push-to-talk"

    def _on_mode_ptt(self, icon: "pystray.Icon", item: "pystray.MenuItem") -> None:
        """Handle push-to-talk mode selection."""
        # Mode change requires restart for V1
        # TODO: Save to config when settings integration is added
        pass

    def _on_mode_toggle(self, icon: "pystray.Icon", item: "pystray.MenuItem") -> None:
        """Handle toggle mode selection."""
        # Mode change requires restart for V1
        # TODO: Save to config when settings integration is added
        pass

    def _on_settings(self, icon: "pystray.Icon", item: "pystray.MenuItem") -> None:
        """Handle Settings menu item."""
        from localwispr.settings_window import open_settings

        logger.info("tray_app: opening settings window")
        open_settings(on_settings_changed=self._on_settings_changed)

    def _on_mode_changed(self, mode) -> None:
        """Handle mode change callback.

        Shows a toast notification with the new mode name.

        Args:
            mode: The new Mode object.
        """
        from localwispr.notifications import show_notification

        logger.info("tray_app: mode changed to %s", mode.name)

        # Show toast notification
        show_notification(
            title=f"{APP_DISPLAY_NAME} Mode",
            message=f"{mode.icon} {mode.name}: {mode.description}",
            timeout=2,
        )

        # Update the tray tooltip
        self._update_tooltip()

    def _on_mode_cycle(self) -> None:
        """Handle mode cycle hotkey callback.

        Cycles to the next transcription mode.
        """
        from localwispr.modes import cycle_mode

        new_mode = cycle_mode()
        logger.info("tray_app: mode cycled to %s", new_mode.name)

    def _update_tooltip(self) -> None:
        """Update the tray icon tooltip with current mode."""
        if self._icon is not None:
            mode = self._mode_manager.current_mode
            self._icon.title = f"{APP_DISPLAY_NAME} - {mode.name} Mode"

    def _on_about(self, icon: "pystray.Icon", item: "pystray.MenuItem") -> None:
        """Handle About menu item."""
        from localwispr.notifications import show_notification

        variant_info = " [TEST BUILD]" if IS_TEST_BUILD else ""
        show_notification(
            title=f"About {APP_DISPLAY_NAME}",
            message=f"LocalWispr v0.1.0{variant_info}\nLocal speech-to-text with Whisper\nPrivacy-first, no cloud APIs",
            timeout=5,
        )

    def _on_exit(self, icon: "pystray.Icon", item: "pystray.MenuItem") -> None:
        """Handle Exit menu item.

        Initiates clean shutdown sequence.
        """
        logger.info("tray_app: exit requested")
        self._shutdown()

    def _shutdown(self) -> None:
        """Perform clean shutdown.

        Shutdown sequence:
        1. Signal stop to all background threads
        2. Stop hotkey listener
        3. Stop the overlay widget
        4. Shutdown transcription executor
        5. Join threads with timeout
        6. Stop the tray icon
        """
        logger.info("tray_app: shutdown started")

        # Signal stop to all threads
        self._stop_event.set()

        # Stop the hotkey listener
        if self._hotkey_listener is not None:
            try:
                self._hotkey_listener.stop()
            except Exception:
                pass

        # Stop the overlay widget
        try:
            self._overlay.stop()
        except Exception:
            pass

        # Shutdown transcription executor (don't wait for pending work)
        try:
            self._transcription_executor.shutdown(wait=False, cancel_futures=True)
            logger.debug("tray_app: transcription executor shutdown")
        except Exception:
            pass

        # Join background threads with timeout
        for thread in self._background_threads:
            if thread.is_alive():
                thread.join(timeout=self.THREAD_JOIN_TIMEOUT)
                if thread.is_alive():
                    logger.warning("tray_app: thread join timeout")

        # Stop the tray icon
        if self._icon is not None:
            self._icon.stop()

        logger.info("tray_app: shutdown complete")

    def _on_record_start(self) -> None:
        """Callback when recording should start.

        Called by the hotkey listener when the hotkey chord is activated.
        """
        from localwispr.config import get_config
        from localwispr.feedback import play_start_beep

        logger.info("recording: start")
        start_time = time.time()

        try:
            # Load config for settings
            config = get_config()

            # Mute system audio if enabled
            if config["hotkeys"].get("mute_system", False):
                from localwispr.volume import mute_system

                self._was_muted_before_recording = mute_system()
                logger.debug("recording: system muted (was_muted=%s)", self._was_muted_before_recording)

            # Initialize recorder on first use
            if self._recorder is None:
                from localwispr.audio import AudioRecorder

                self._recorder = AudioRecorder()
                logger.debug("recording: recorder initialized")

            with self._recorder_lock:
                if self._recorder.is_recording:
                    logger.warning("recording: already recording")
                    return
                self._recorder.start_recording()

            # Update state and show overlay
            self.update_state(TrayState.RECORDING)
            self._overlay.show_recording()

            # Play audio feedback if enabled
            if config["hotkeys"]["audio_feedback"]:
                play_start_beep()

            duration_ms = int((time.time() - start_time) * 1000)
            logger.info("recording: started, duration_ms=%d", duration_ms)

        except Exception as e:
            logger.error("recording: start failed, error_type=%s", type(e).__name__)
            # Reset state on error
            self.update_state(TrayState.IDLE)
            if self._hotkey_listener is not None:
                self._hotkey_listener.on_transcription_complete()

            self._overlay.show_error("Failed to start recording")

    def _restore_system_audio(self) -> None:
        """Restore system audio mute state after recording."""
        from localwispr.config import get_config

        config = get_config()
        if config["hotkeys"].get("mute_system", False):
            from localwispr.volume import restore_mute_state

            restore_mute_state(self._was_muted_before_recording)
            logger.debug("recording: system mute restored to %s", self._was_muted_before_recording)

    def _get_recorded_audio(self) -> "np.ndarray | None":
        """Get recorded audio from the recorder.

        Returns:
            Audio array if successful, None if no audio or not recording.
        """
        import numpy as np

        with self._recorder_lock:
            if self._recorder is None or not self._recorder.is_recording:
                logger.warning("recording: not recording, skipping transcription")
                return None
            audio = self._recorder.get_whisper_audio()

        # Check if we have audio
        audio_duration = len(audio) / 16000.0
        if audio_duration < 0.1:
            logger.warning("recording: no audio captured")
            self._overlay.show_error("No audio captured")
            return None

        logger.debug("recording: audio captured, duration_s=%.2f", audio_duration)
        return audio

    def _wait_for_model(self) -> bool:
        """Wait for the Whisper model to finish preloading.

        Returns:
            True if model is ready, False on timeout.
        """
        if not self._model_preload_complete.is_set():
            logger.info("transcriber: waiting for model preload")
            # Wait up to 60 seconds for model to load
            if not self._model_preload_complete.wait(timeout=60.0):
                logger.error("transcriber: preload timeout")
                self._overlay.show_error("Model load timeout")
                return False
        return True

    def _get_transcriber(self):
        """Get or create the transcriber instance.

        Returns:
            WhisperTranscriber instance.
        """
        from localwispr.transcribe import WhisperTranscriber

        with self._transcriber_lock:
            # Check if preload failed
            if self._model_preload_error is not None:
                logger.error("transcriber: preload had error, retrying sync")
                # Clear the error and try sync load
                self._model_preload_error = None
                self._transcriber = None

            # Initialize transcriber if preload failed or wasn't done
            if self._transcriber is None:
                logger.info("transcriber: initializing (sync fallback)")
                self._transcriber = WhisperTranscriber()
                # Force model load
                _ = self._transcriber.model
                logger.info("transcriber: model loaded")

            return self._transcriber

    def _perform_transcription(self, audio: "np.ndarray", transcriber):
        """Perform transcription using the appropriate mode.

        Args:
            audio: Audio array to transcribe.
            transcriber: WhisperTranscriber instance.

        Returns:
            Transcription result.
        """
        from localwispr.modes import get_mode_prompt

        if self._mode_manager.is_manual_override:
            # Use mode's prompt directly
            initial_prompt = get_mode_prompt()
            result = transcriber.transcribe(audio, initial_prompt=initial_prompt)
            logger.debug(
                "transcription: using mode prompt, mode=%s",
                self._mode_manager.current_mode.name,
            )
        else:
            # Use context detection for automatic mode selection
            if self._detector is None:
                from localwispr.context import ContextDetector

                self._detector = ContextDetector()

            from localwispr.transcribe import transcribe_with_context

            result = transcribe_with_context(audio, transcriber, self._detector)

        inference_time_ms = int(result.inference_time * 1000)
        logger.info(
            "transcription: complete, duration_ms=%d, was_retranscribed=%s",
            inference_time_ms,
            result.was_retranscribed,
        )
        return result

    def _output_result(self, result, start_time: float) -> None:
        """Output the transcription result.

        Args:
            result: Transcription result.
            start_time: Pipeline start time for logging.
        """
        from localwispr.config import get_config
        from localwispr.output import output_transcription

        config = get_config()
        auto_paste = config["output"]["auto_paste"]
        paste_delay_ms = config["output"]["paste_delay_ms"]

        success = output_transcription(
            text=result.text,
            auto_paste=auto_paste,
            paste_delay_ms=paste_delay_ms,
            play_feedback=config["hotkeys"]["audio_feedback"],
        )

        # Hide overlay on completion (success or paste failure)
        self._overlay.hide()

        if not success:
            logger.warning("output: paste failed, text is in clipboard")

        total_time_ms = int((time.time() - start_time) * 1000)
        logger.info("pipeline: complete, total_ms=%d", total_time_ms)

    def _on_record_stop(self) -> None:
        """Callback when recording should stop and transcription should begin.

        Called by the hotkey listener when the hotkey chord is released.
        Returns immediately after starting background transcription to keep
        the hotkey listener responsive.
        """
        logger.info("recording: stop, starting background transcription")

        # Restore system audio mute state
        self._restore_system_audio()

        # Update state and show overlay
        self.update_state(TrayState.TRANSCRIBING)
        self._overlay.show_transcribing()

        # Get audio from recorder (quick operation)
        audio = self._get_recorded_audio()
        if audio is None or len(audio) == 0:
            self._finish_transcription()
            return

        # Increment generation to track this transcription
        with self._generation_lock:
            self._current_generation += 1
            generation = self._current_generation

        # Reset hotkey listener state immediately - user can start new recording
        # Note: We keep TRANSCRIBING state visible until background work completes
        if self._hotkey_listener is not None:
            self._hotkey_listener.on_transcription_complete()

        # Process in background - this returns immediately
        self._transcription_executor.submit(
            self._process_transcription_background,
            audio,
            generation,
        )

    def _finish_transcription(self) -> None:
        """Finish transcription and reset state."""
        self.update_state(TrayState.IDLE)
        self._overlay.hide()
        if self._hotkey_listener is not None:
            self._hotkey_listener.on_transcription_complete()

    def _process_transcription_background(
        self,
        audio: np.ndarray,
        generation: int,
    ) -> None:
        """Background transcription with timeout and generation check.

        This method runs in a ThreadPoolExecutor to avoid blocking the
        hotkey listener thread. It checks the generation ID before outputting
        to handle cases where user started a new recording.

        Args:
            audio: Audio array to transcribe.
            generation: Generation ID for this transcription request.
        """
        start_time = time.time()
        HARD_TIMEOUT = 120.0  # 2 minutes

        try:
            # Check if this transcription is still current
            with self._generation_lock:
                if generation != self._current_generation:
                    logger.info(
                        "transcription: discarding stale request, gen=%d, current=%d",
                        generation,
                        self._current_generation,
                    )
                    return

            # Wait for model (with timeout)
            if not self._model_preload_complete.wait(timeout=30.0):
                logger.error("transcription: model preload timeout")
                self._overlay.show_error("Model load timeout")
                self._finish_transcription()
                return

            # Get or create transcriber
            transcriber = self._get_transcriber()
            if transcriber is None:
                logger.error("transcription: failed to get transcriber")
                self._overlay.show_error("Transcription failed")
                self._finish_transcription()
                return

            # Perform transcription
            result = self._perform_transcription(audio, transcriber)

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > HARD_TIMEOUT:
                logger.warning(
                    "transcription: exceeded timeout, elapsed_s=%.1f",
                    elapsed,
                )
                self._overlay.show_error("Transcription timeout")
                self._finish_transcription()
                return

            # Check generation again before output
            with self._generation_lock:
                if generation != self._current_generation:
                    logger.info(
                        "transcription: discarding stale result, gen=%d, current=%d",
                        generation,
                        self._current_generation,
                    )
                    return

            # Output result
            self._output_result(result, start_time)

        except Exception as e:
            logger.error("transcription: background failed, error_type=%s", type(e).__name__)
            self._overlay.show_error("Transcription failed")

        finally:
            # Always reset state when done (unless a newer generation took over)
            with self._generation_lock:
                if generation == self._current_generation:
                    self.update_state(TrayState.IDLE)
                    self._overlay.hide()

    def _get_current_audio_level(self) -> float:
        """Get current audio level from recorder for overlay visualization.

        Returns:
            Audio RMS level (0.0-1.0), or 0.0 if not recording.
        """
        with self._recorder_lock:
            if self._recorder is not None and self._recorder.is_recording:
                return self._recorder.get_rms_level()
        return 0.0

    def _get_model_name(self) -> str:
        """Get current Whisper model name for overlay display.

        Returns:
            Model name (e.g., "large-v3"), or empty string if not loaded.
        """
        with self._transcriber_lock:
            if self._transcriber is not None:
                return self._transcriber.model_name
        return ""

    def _preload_model_async(self) -> None:
        """Load Whisper model in background thread.

        This method is run in a daemon thread during startup to preload
        the ~2GB Whisper model before the user's first transcription request.
        """
        try:
            from localwispr.transcribe import WhisperTranscriber

            logger.info("model_preload: starting")
            transcriber = WhisperTranscriber()
            # Trigger actual model load by accessing the model property
            _ = transcriber.model
            # Store with lock to avoid race with _on_record_stop
            with self._transcriber_lock:
                self._transcriber = transcriber
            logger.info("model_preload: complete")
        except Exception as e:
            logger.error("model_preload: failed, error_type=%s", type(e).__name__)
            self._model_preload_error = e
        finally:
            self._model_preload_complete.set()

    def _start_hotkey_listener(self) -> None:
        """Start the hotkey listener in a background thread."""
        from localwispr.config import get_config
        from localwispr.hotkeys import HotkeyListener, HotkeyMode

        config = get_config()
        hotkey_config = config["hotkeys"]

        # Determine mode
        if hotkey_config["mode"] == "toggle":
            mode = HotkeyMode.TOGGLE
        else:
            mode = HotkeyMode.PUSH_TO_TALK

        # Create listener with mode cycle callback
        self._hotkey_listener = HotkeyListener(
            on_record_start=self._on_record_start,
            on_record_stop=self._on_record_stop,
            on_mode_cycle=self._on_mode_cycle,
            mode=mode,
        )

        # Start listener
        self._hotkey_listener.start()
        logger.info("hotkey_listener: started, mode=%s", mode.name)

    def _restart_hotkey_listener(self) -> None:
        """Restart the hotkey listener with current config.

        Stops the existing listener and starts a new one with fresh config.
        Used when settings change (e.g., mode changed from PTT to Toggle).
        """
        logger.info("hotkey_listener: restarting")

        # Stop existing listener
        if self._hotkey_listener is not None:
            try:
                self._hotkey_listener.stop()
                logger.debug("hotkey_listener: stopped existing listener")
            except Exception as e:
                logger.error("hotkey_listener: stop failed, error_type=%s", type(e).__name__)

        # Start new listener with fresh config
        try:
            self._start_hotkey_listener()
            logger.info("hotkey_listener: restart complete")
        except Exception as e:
            logger.error("hotkey_listener: restart failed, error_type=%s", type(e).__name__)
            self._overlay.show_error("Failed to restart hotkey listener")

    def _on_settings_changed(self) -> None:
        """Handle settings changed callback from settings window.

        Restarts the hotkey listener and invalidates the transcriber
        to apply new settings immediately. The transcriber will be
        recreated on next transcription with new settings (vocabulary,
        model name, device, etc.).
        """
        logger.info("tray_app: settings changed, applying")
        self._restart_hotkey_listener()

        # Invalidate transcriber so it's recreated with new settings
        with self._transcriber_lock:
            if self._transcriber is not None:
                logger.info("tray_app: invalidating transcriber for settings reload")
                self._transcriber = None

        # Reset model preload state so model reloads on next use
        self._model_preload_complete.clear()

    def run(self) -> None:
        """Run the tray application.

        This method blocks until the user exits. The tray icon runs in
        the main thread, with the hotkey listener in a background thread.
        """
        import pystray

        # Get initial mode for tooltip
        mode = self._mode_manager.current_mode
        initial_title = f"{APP_DISPLAY_NAME} - {mode.name} Mode"

        # Create the icon
        self._icon = pystray.Icon(
            name="localwispr",
            icon=create_icon_image(TrayState.IDLE),
            title=initial_title,
            menu=self._create_menu(),
        )

        # Set up periodic queue processing using the icon's run callback
        def setup(icon: pystray.Icon) -> None:
            """Setup callback - runs after icon is visible."""
            icon.visible = True

            # Start the floating overlay widget
            self._overlay.start()

            # Start model preloading in background (non-blocking)
            preload_thread = threading.Thread(
                target=self._preload_model_async,
                daemon=True,
            )
            preload_thread.start()
            logger.info("model_preload: thread started")

            # Start the hotkey listener
            try:
                self._start_hotkey_listener()
            except Exception as e:
                logger.error("hotkey_listener: failed to start, error_type=%s", type(e).__name__)
                self._overlay.show_error("Failed to start hotkey listener")

            # Start queue polling in a background thread
            poll_thread = threading.Thread(
                target=self._poll_queue_loop,
                daemon=True,
            )
            poll_thread.start()

        logger.info("tray_app: starting")

        # Run the icon (blocks until stop)
        self._icon.run(setup=setup)

    def _poll_queue_loop(self) -> None:
        """Background thread that polls the update queue.

        Runs until stop event is set.
        """
        while not self._stop_event.is_set():
            self._process_queue()
            self._stop_event.wait(timeout=self.QUEUE_POLL_INTERVAL)

    def stop(self) -> None:
        """Stop the tray application from an external thread.

        Safe to call from any thread.
        """
        self._shutdown()
