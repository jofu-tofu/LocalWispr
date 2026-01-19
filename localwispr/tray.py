"""System tray application module for LocalWispr.

This module implements the system tray UI using pystray, providing:
- Custom "L through sound wave" branded icon with state colors
- Tray menu with recording mode, about, and exit options
- Thread-safe state management for external callers
- Integration with hotkey listener in background thread
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
from enum import Enum, auto
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from PIL import Image, ImageDraw

if TYPE_CHECKING:
    import pystray


# Configure module logger with rotation
def _setup_logger() -> logging.Logger:
    """Set up the LocalWispr logger with rotation.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("localwispr")

    # Only set up handlers if not already configured
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # File handler with rotation (10MB max, keep 2 backups)
        log_file = Path.cwd() / "localwispr.log"
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


# Icon colors for each state
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


def create_icon_image(state: TrayState = TrayState.IDLE, size: int = 64) -> Image.Image:
    """Create the LocalWispr icon: "L" through a sound wave.

    The icon shows a stylized letter "L" with sound wave lines emanating
    from the right side, representing speech-to-text functionality.

    Args:
        state: Current tray state for color selection.
        size: Icon size in pixels (square).

    Returns:
        PIL Image of the icon.
    """
    colors = STATE_COLORS[state]

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

        return pystray.Menu(
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
            pystray.MenuItem("About LocalWispr", self._on_about),
            pystray.MenuItem("Exit", self._on_exit),
        )

    def _is_mode_ptt(self) -> bool:
        """Check if current mode is push-to-talk.

        Returns:
            True if push-to-talk mode.
        """
        from localwispr.config import load_config

        config = load_config()
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

    def _on_about(self, icon: "pystray.Icon", item: "pystray.MenuItem") -> None:
        """Handle About menu item."""
        from localwispr.notifications import show_notification

        show_notification(
            title="About LocalWispr",
            message="LocalWispr v0.1.0\nLocal speech-to-text with Whisper\nPrivacy-first, no cloud APIs",
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
        3. Join threads with timeout
        4. Stop the tray icon
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
        from localwispr.config import load_config
        from localwispr.feedback import play_start_beep
        from localwispr.notifications import show_recording_started

        logger.info("recording: start")
        start_time = time.time()

        try:
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

            # Update state and show notification
            self.update_state(TrayState.RECORDING)
            show_recording_started()

            # Play audio feedback if enabled
            config = load_config()
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

            from localwispr.notifications import show_error

            show_error("Failed to start recording")

    def _on_record_stop(self) -> None:
        """Callback when recording should stop and transcription should begin.

        Called by the hotkey listener when the hotkey chord is released.
        """
        from localwispr.config import load_config
        from localwispr.notifications import (
            show_complete,
            show_clipboard_only,
            show_error,
            show_transcribing,
        )
        from localwispr.output import output_transcription

        logger.info("recording: stop, starting transcription")
        start_time = time.time()

        try:
            # Update state and show notification
            self.update_state(TrayState.TRANSCRIBING)
            show_transcribing()

            # Get audio from recorder
            with self._recorder_lock:
                if self._recorder is None or not self._recorder.is_recording:
                    logger.warning("recording: not recording, skipping transcription")
                    self._finish_transcription()
                    return
                audio = self._recorder.get_whisper_audio()

            # Check if we have audio
            audio_duration = len(audio) / 16000.0
            if audio_duration < 0.1:
                logger.warning("recording: no audio captured")
                show_error("No audio captured")
                self._finish_transcription()
                return

            logger.debug("recording: audio captured, duration_s=%.2f", audio_duration)

            # Initialize transcriber on first use
            if self._transcriber is None:
                from localwispr.transcribe import WhisperTranscriber

                logger.info("transcriber: initializing")
                self._transcriber = WhisperTranscriber()
                # Force model load
                _ = self._transcriber.model
                logger.info("transcriber: model loaded")

            # Initialize context detector on first use
            if self._detector is None:
                from localwispr.context import ContextDetector

                self._detector = ContextDetector()

            # Transcribe with context detection
            from localwispr.transcribe import transcribe_with_context

            result = transcribe_with_context(audio, self._transcriber, self._detector)

            inference_time_ms = int(result.inference_time * 1000)
            logger.info(
                "transcription: complete, duration_ms=%d, was_retranscribed=%s",
                inference_time_ms,
                result.was_retranscribed,
            )

            # Output transcription
            config = load_config()
            auto_paste = config["output"]["auto_paste"]
            paste_delay_ms = config["output"]["paste_delay_ms"]

            success = output_transcription(
                text=result.text,
                auto_paste=auto_paste,
                paste_delay_ms=paste_delay_ms,
                play_feedback=config["hotkeys"]["audio_feedback"],
            )

            if success:
                if auto_paste:
                    show_complete()
                else:
                    show_clipboard_only()
            else:
                show_error("Paste failed - text is in clipboard")

            total_time_ms = int((time.time() - start_time) * 1000)
            logger.info("pipeline: complete, total_ms=%d", total_time_ms)

        except Exception as e:
            logger.error("transcription: failed, error_type=%s", type(e).__name__)
            show_error("Transcription failed")

        finally:
            self._finish_transcription()

    def _finish_transcription(self) -> None:
        """Finish transcription and reset state."""
        self.update_state(TrayState.IDLE)
        if self._hotkey_listener is not None:
            self._hotkey_listener.on_transcription_complete()

    def _start_hotkey_listener(self) -> None:
        """Start the hotkey listener in a background thread."""
        from localwispr.config import load_config
        from localwispr.hotkeys import HotkeyListener, HotkeyMode

        config = load_config()
        hotkey_config = config["hotkeys"]

        # Determine mode
        if hotkey_config["mode"] == "toggle":
            mode = HotkeyMode.TOGGLE
        else:
            mode = HotkeyMode.PUSH_TO_TALK

        # Create listener
        self._hotkey_listener = HotkeyListener(
            on_record_start=self._on_record_start,
            on_record_stop=self._on_record_stop,
            mode=mode,
        )

        # Start listener
        self._hotkey_listener.start()
        logger.info("hotkey_listener: started, mode=%s", mode.name)

    def run(self) -> None:
        """Run the tray application.

        This method blocks until the user exits. The tray icon runs in
        the main thread, with the hotkey listener in a background thread.
        """
        import pystray

        # Create the icon
        self._icon = pystray.Icon(
            name="localwispr",
            icon=create_icon_image(TrayState.IDLE),
            title="LocalWispr",
            menu=self._create_menu(),
        )

        # Set up periodic queue processing using the icon's run callback
        def setup(icon: pystray.Icon) -> None:
            """Setup callback - runs after icon is visible."""
            icon.visible = True

            # Start the hotkey listener
            try:
                self._start_hotkey_listener()
            except Exception as e:
                logger.error("hotkey_listener: failed to start, error_type=%s", type(e).__name__)
                from localwispr.notifications import show_error

                show_error(f"Failed to start hotkey listener")

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
