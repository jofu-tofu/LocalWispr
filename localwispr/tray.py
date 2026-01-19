"""System tray application module for LocalWispr.

This module implements the system tray UI using pystray, providing:
- Custom "L through sound wave" branded icon with state colors
- Tray menu with recording mode, about, and exit options
- Thread-safe state management for external callers
- Integration with hotkey listener in background thread
- Clean shutdown sequence to prevent orphaned threads
"""

from __future__ import annotations

import queue
import threading
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable

from PIL import Image, ImageDraw

if TYPE_CHECKING:
    import pystray


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
    listener for recording functionality.

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
        self._icon: pystray.Icon | None = None

        # Background threads to join on shutdown
        self._background_threads: list[threading.Thread] = []

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
            self._state = new_state

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
        # TODO: Read from config when settings integration is added
        return True  # Default to PTT for V1

    def _on_mode_ptt(self, icon: "pystray.Icon", item: "pystray.MenuItem") -> None:
        """Handle push-to-talk mode selection."""
        # TODO: Save to config when settings integration is added
        pass

    def _on_mode_toggle(self, icon: "pystray.Icon", item: "pystray.MenuItem") -> None:
        """Handle toggle mode selection."""
        # TODO: Save to config when settings integration is added
        pass

    def _on_about(self, icon: "pystray.Icon", item: "pystray.MenuItem") -> None:
        """Handle About menu item."""
        # Show a simple notification for now
        # TODO: Could open an about dialog in V2
        if self._icon is not None:
            try:
                self._icon.notify(
                    "LocalWispr v0.1.0\n"
                    "Local speech-to-text with Whisper\n"
                    "Privacy-first, no cloud APIs",
                    "About LocalWispr",
                )
            except Exception:
                pass  # Notifications may not be supported on all platforms

    def _on_exit(self, icon: "pystray.Icon", item: "pystray.MenuItem") -> None:
        """Handle Exit menu item.

        Initiates clean shutdown sequence.
        """
        self._shutdown()

    def _shutdown(self) -> None:
        """Perform clean shutdown.

        Shutdown sequence:
        1. Signal stop to all background threads
        2. Join threads with timeout
        3. Stop the tray icon
        """
        # Signal stop to all threads
        self._stop_event.set()

        # Join background threads with timeout
        for thread in self._background_threads:
            if thread.is_alive():
                thread.join(timeout=self.THREAD_JOIN_TIMEOUT)
                if thread.is_alive():
                    # Thread didn't stop in time - log warning but continue
                    pass  # Can't log here safely, just continue shutdown

        # Stop the tray icon
        if self._icon is not None:
            self._icon.stop()

    def run(self) -> None:
        """Run the tray application.

        This method blocks until the user exits. The tray icon runs in
        the main thread, with a periodic callback to process state updates.
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
            # Start queue polling in a background thread
            poll_thread = threading.Thread(
                target=self._poll_queue_loop,
                daemon=True,
            )
            poll_thread.start()

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
