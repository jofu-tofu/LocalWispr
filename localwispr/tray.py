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
from enum import Enum, auto
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Union

from PIL import Image, ImageDraw

from localwispr.pipeline import PipelineResult, RecordingPipeline

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
        "wave": "#4A90D9",     # Blue wave (same as idle)
        "letter": "#FFFFFF",   # White L
        "background": None,    # Transparent
    },
    TrayState.TRANSCRIBING: {
        "wave": "#4A90D9",     # Blue wave (same as idle)
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
        "wave": "#FF8C00",     # Orange wave (same as idle)
        "letter": "#FFFFFF",   # White L
        "background": None,    # Transparent
    },
    TrayState.TRANSCRIBING: {
        "wave": "#FF8C00",     # Orange wave (same as idle)
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

        # Thread-safe queue for state updates and transcription results
        # Queue accepts TrayState or ("result", PipelineResult, gen) tuples
        self._update_queue: queue.Queue[
            Union[TrayState, tuple[str, PipelineResult | None, int]]
        ] = queue.Queue()

        # Stop event for coordinating shutdown
        self._stop_event = threading.Event()

        # pystray icon reference
        self._icon: "pystray.Icon | None" = None

        # Background threads to join on shutdown
        self._background_threads: list[threading.Thread] = []

        # Hotkey listener (lazy loaded)
        self._hotkey_listener = None

        # Track current transcription generation for UI safety
        self._current_transcription_gen: int = 0

        # Initialize mode manager with notification callback
        from localwispr.modes import get_mode_manager

        self._mode_manager = get_mode_manager(
            auto_reset=False,
            on_mode_change=self._on_mode_changed,
        )

        # Floating overlay widget for visual feedback
        from localwispr.overlay import OverlayWidget

        self._overlay = OverlayWidget(
            audio_level_callback=self._get_current_audio_level,
            model_name_callback=self._get_model_name,
        )

        # Initialize recording pipeline (single source of truth for recording/transcription)
        self._pipeline = RecordingPipeline(
            mode_manager=self._mode_manager,
            on_error=lambda msg: self._overlay.show_error(msg),
        )

        # Register settings invalidation handlers
        self._register_settings_handlers()

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
        """Process pending state updates and transcription results from the queue.

        Non-blocking - processes all available updates.
        Handles both TrayState updates and result tuples from async transcription.
        """
        while True:
            try:
                item = self._update_queue.get_nowait()
                if isinstance(item, TrayState):
                    # Direct state update
                    self._set_state(item)
                elif isinstance(item, tuple):
                    # Transcription result tuple: ("result"|"complete", payload, gen)
                    msg_type, payload, gen = item
                    # Check generation before processing
                    if gen != self._current_transcription_gen:
                        continue  # Stale, skip
                    if msg_type == "result":
                        self._handle_transcription_result(payload)
                    elif msg_type == "complete":
                        self._finish_transcription()
            except queue.Empty:
                break

    def _create_menu(self) -> "pystray.Menu":
        """Create the tray context menu.

        Returns:
            pystray Menu object.
        """
        import pystray

        from localwispr.modes import get_all_modes

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

    def _make_mode_callback(self, mode_type):
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

    def _make_mode_check(self, mode_type):
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
        4. Shutdown pipeline (waits for in-flight transcription)
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

        # Shutdown pipeline (waits for in-flight transcription)
        try:
            self._pipeline.shutdown(timeout=5.0)
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
        Delegates to RecordingPipeline for actual recording.
        """
        from localwispr.config import get_config
        from localwispr.feedback import play_start_beep

        logger.info("recording: start")

        config = get_config()
        mute = config["hotkeys"].get("mute_system", False)

        success = self._pipeline.start_recording(mute_system=mute)
        if success:
            self.update_state(TrayState.RECORDING)
            self._overlay.show_recording()

            # Play audio feedback if enabled
            if config["hotkeys"]["audio_feedback"]:
                play_start_beep()

            logger.info("recording: started")
        else:
            # Reset hotkey listener on error
            if self._hotkey_listener is not None:
                self._hotkey_listener.on_transcription_complete()
            self._overlay.show_error("Failed to start recording")

    def _on_record_stop(self) -> None:
        """Callback when recording should stop and transcription should begin.

        Called by the hotkey listener when the hotkey chord is released.
        Delegates to RecordingPipeline for async transcription with callbacks.
        """
        from localwispr.config import get_config

        logger.info("recording: stop, starting async transcription")

        # Update state and show overlay
        self.update_state(TrayState.TRANSCRIBING)
        self._overlay.show_transcribing()

        # Reset hotkey listener immediately - user can start new recording
        if self._hotkey_listener is not None:
            self._hotkey_listener.on_transcription_complete()

        # Capture config values at stop time (not in callback)
        config = get_config()
        mute = config["hotkeys"].get("mute_system", False)

        # Async transcription - callbacks fire on background thread
        # Note: Callbacks receive generation as a parameter from pipeline,
        # avoiding closure bug where gen wasn't defined at lambda creation time
        gen = self._pipeline.stop_and_transcribe_async(
            mute_system=mute,
            on_result=lambda r, g: self._queue_result_callback(r, g),
            on_complete=lambda g: self._queue_complete_callback(g),
        )
        self._current_transcription_gen = gen

    def _queue_result_callback(
        self, result: PipelineResult, generation: int
    ) -> None:
        """Queue result handling to UI thread via update queue.

        Args:
            result: Transcription result from pipeline.
            generation: Generation ID to detect stale results.
        """
        self._update_queue.put(("result", result, generation))

    def _queue_complete_callback(self, generation: int) -> None:
        """Queue completion handling to UI thread.

        Args:
            generation: Generation ID to detect stale results.
        """
        self._update_queue.put(("complete", None, generation))

    def _handle_transcription_result(self, result: PipelineResult) -> None:
        """Handle transcription result on UI thread.

        Args:
            result: Transcription result from pipeline.
        """
        if result.success:
            self._output_result(result)
        else:
            self._overlay.show_error(result.error or "Transcription failed")

    def _output_result(self, result: PipelineResult) -> None:
        """Output the transcription result.

        Args:
            result: Pipeline result with transcription text.
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

        if not success:
            logger.warning("output: paste failed, text is in clipboard")

        logger.info(
            "pipeline: output complete, audio_duration_s=%.2f, inference_ms=%d",
            result.audio_duration,
            int(result.inference_time * 1000),
        )

    def _finish_transcription(self) -> None:
        """Finish transcription and reset state."""
        self.update_state(TrayState.IDLE)
        self._overlay.hide()

    def _get_current_audio_level(self) -> float:
        """Get current audio level from pipeline for overlay visualization.

        Returns:
            Audio RMS level (0.0-1.0), or 0.0 if not recording.
        """
        return self._pipeline.get_rms_level()

    def _get_model_name(self) -> str:
        """Get current Whisper model name for overlay display.

        Returns:
            Model name (e.g., "large-v3"), or empty string if not loaded.
        """
        return self._pipeline.get_model_name()

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

    def _register_settings_handlers(self) -> None:
        """Register handlers with the SettingsManager.

        Called during initialization to set up automatic invalidation
        when settings change.
        """
        from localwispr.settings_manager import (
            InvalidationFlags,
            get_settings_manager,
        )

        manager = get_settings_manager()
        manager.register_handler(
            InvalidationFlags.HOTKEY_LISTENER,
            self._restart_hotkey_listener,
        )
        manager.register_handler(
            InvalidationFlags.TRANSCRIBER,
            self._pipeline.invalidate_transcriber,
        )
        manager.register_handler(
            InvalidationFlags.MODEL_PRELOAD,
            self._pipeline.clear_model_preload,
        )
        logger.debug("tray_app: settings handlers registered")

    def _on_settings_changed(self) -> None:
        """Handle settings changed callback from settings window.

        Note: Most work is now done by SettingsManager handlers.
        This callback is kept for any additional logic that doesn't
        fit the invalidation pattern.
        """
        logger.info("tray_app: settings changed callback invoked")

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

            # Start model preloading via pipeline (non-blocking)
            self._pipeline.preload_model_async()
            logger.info("model_preload: delegated to pipeline")

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
