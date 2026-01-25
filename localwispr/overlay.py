"""Floating overlay widget for LocalWispr.

Provides visual feedback via a floating overlay instead of toast notifications:
- Recording state with model name and audio level visualization (waveform bars)
- Transcribing state with model name and rotating spinner animation
- Consistent teal color scheme across all states
- Error feedback with brief flash

Thread Safety:
    The overlay runs in its own thread with a dedicated Tkinter event loop.
    All public methods are thread-safe via queue-based communication.

Privacy Note:
    The overlay NEVER displays transcription content, only state indicators.
"""

from __future__ import annotations

import ctypes
import logging
import math
import queue
import threading
import tkinter as tk
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    pass

# Import build variant detection
from localwispr import IS_TEST_BUILD

# App display name varies by build variant
APP_DISPLAY_NAME = "LocalWispr [TEST]" if IS_TEST_BUILD else "LocalWispr"

logger = logging.getLogger("localwispr")


class OverlayState(Enum):
    """Visual states for the overlay."""

    HIDDEN = auto()
    RECORDING = auto()
    TRANSCRIBING = auto()


# Single overlay color (smoky charcoal)
OVERLAY_COLOR = "#2D3436"

# Overlay dimensions (reduced ~50%)
PILL_WIDTH = 130
PILL_HEIGHT = 38
PILL_RADIUS = 19
PADDING_BOTTOM = 40

# Waveform bar settings (scaled down)
NUM_BARS = 9
BAR_WIDTH = 2
BAR_GAP = 2
BAR_MIN_HEIGHT = 2  # Dot-like when silent
BAR_MAX_HEIGHT = 26
BAR_COLOR = "#E8E8E8"  # Slightly softer white

# Spinner settings
SPINNER_RADIUS = 10
SPINNER_WIDTH = 3
SPINNER_ARC_EXTENT = 270
SPINNER_SPEED = 360  # Degrees per second

# Model name display
MODEL_FONT = ("Segoe UI", 9)
MODEL_COLOR = "#FFFFFF"
MODEL_LEFT_MARGIN = 12

# Animation settings
POLL_INTERVAL_MS = 50
EMA_ALPHA = 0.3  # Smoothing factor for audio level


class OverlayWidget:
    """Floating overlay widget for visual recording feedback.

    This widget runs in a dedicated thread and provides thread-safe
    methods to show/hide and change states.

    Usage:
        overlay = OverlayWidget(audio_level_callback=get_level_func)
        overlay.start()
        overlay.show_recording()
        # ... recording happens ...
        overlay.show_transcribing()
        # ... transcription happens ...
        overlay.hide()
        overlay.stop()
    """

    def __init__(
        self,
        audio_level_callback: Callable[[], float] | None = None,
        model_name_callback: Callable[[], str] | None = None,
    ) -> None:
        """Initialize the overlay widget.

        Args:
            audio_level_callback: Function that returns current audio level (0.0-1.0).
                Called during recording to animate waveform bars.
            model_name_callback: Function that returns current model name (e.g., "large-v3").
                Called when drawing to display model name on overlay.
        """
        # Enable DPI awareness for crisp rendering (once per process)
        if not hasattr(OverlayWidget, '_dpi_awareness_set'):
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
                OverlayWidget._dpi_awareness_set = True
            except (AttributeError, OSError, WindowsError):
                pass  # Not on Windows, older Windows, or already set

        self._audio_level_callback = audio_level_callback
        self._model_name_callback = model_name_callback
        self._command_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._root: tk.Tk | None = None
        self._canvas: tk.Canvas | None = None
        self._running = False
        self._state = OverlayState.HIDDEN
        self._smoothed_level = 0.0
        self._bar_heights: list[float] = [BAR_MIN_HEIGHT] * NUM_BARS
        self._spinner_angle = 0.0
        self._restart_count = 0
        self._max_restarts = 3

    def start(self) -> None:
        """Start the overlay thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._running = True
        self._restart_count = 0
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.debug("overlay: started")

    def stop(self) -> None:
        """Stop the overlay and clean up."""
        self._running = False
        self._command_queue.put(("quit", None))

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        logger.debug("overlay: stopped")

    def show_recording(self) -> None:
        """Show the recording state with waveform visualization."""
        self._command_queue.put(("show_recording", None))

    def show_transcribing(self) -> None:
        """Show the transcribing state with loading animation."""
        self._command_queue.put(("show_transcribing", None))

    def show_error(self, message: str) -> None:
        """Flash an error indicator briefly.

        Args:
            message: Error message (logged but not displayed).
        """
        self._command_queue.put(("show_error", message))

    def hide(self) -> None:
        """Hide the overlay."""
        self._command_queue.put(("hide", None))

    def _run_loop(self) -> None:
        """Main loop running in dedicated thread."""
        try:
            self._root = tk.Tk()
            self._setup_window()
            self._create_canvas()
            self._schedule_queue_processing()
            self._root.mainloop()
        except Exception as e:
            logger.warning("overlay: thread error, error_type=%s", type(e).__name__)
            self._handle_thread_crash()

    def _handle_thread_crash(self) -> None:
        """Handle overlay thread crash with auto-restart."""
        self._restart_count += 1
        if self._restart_count <= self._max_restarts and self._running:
            logger.info("overlay: restarting, attempt=%d", self._restart_count)
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
        else:
            logger.warning("overlay: max restarts reached, giving up")
            self._running = False

    def _setup_window(self) -> None:
        """Configure the overlay window."""
        if self._root is None:
            return

        self._root.title(f"{APP_DISPLAY_NAME} Overlay")
        self._root.overrideredirect(True)  # No window decorations
        self._root.attributes("-topmost", True)  # Always on top
        self._root.attributes("-alpha", 0.85)  # Glassier look

        # Transparent background color
        self._transparent_color = "#010101"
        self._root.configure(bg=self._transparent_color)
        self._root.attributes("-transparentcolor", self._transparent_color)

        # Position at bottom-center
        self._root.update_idletasks()
        screen_width = self._root.winfo_screenwidth()
        screen_height = self._root.winfo_screenheight()
        x = (screen_width - PILL_WIDTH) // 2
        y = screen_height - PILL_HEIGHT - PADDING_BOTTOM
        self._root.geometry(f"{PILL_WIDTH}x{PILL_HEIGHT}+{x}+{y}")

        # Start hidden
        self._root.withdraw()

    def _create_canvas(self) -> None:
        """Create the drawing canvas."""
        if self._root is None:
            return

        self._canvas = tk.Canvas(
            self._root,
            width=PILL_WIDTH,
            height=PILL_HEIGHT,
            bg=self._transparent_color,
            highlightthickness=0,
        )
        self._canvas.pack()

    def _schedule_queue_processing(self) -> None:
        """Schedule periodic queue processing."""
        if self._root is None or not self._running:
            return

        self._process_queue()
        self._root.after(POLL_INTERVAL_MS, self._schedule_queue_processing)

    def _process_queue(self) -> None:
        """Process commands from other threads."""
        try:
            while True:
                cmd, args = self._command_queue.get_nowait()
                self._handle_command(cmd, args)
        except queue.Empty:
            pass

        # Update animation based on current state
        if self._state == OverlayState.RECORDING:
            self._update_recording_animation()
        elif self._state == OverlayState.TRANSCRIBING:
            self._update_transcribing_animation()

    def _handle_command(self, cmd: str, args: Any) -> None:
        """Handle a command from the queue."""
        if cmd == "show_recording":
            self._do_show_recording()
        elif cmd == "show_transcribing":
            self._do_show_transcribing()
        elif cmd == "show_error":
            self._do_show_error(args)
        elif cmd == "hide":
            self._do_hide()
        elif cmd == "quit":
            self._do_quit()

    def _do_show_recording(self) -> None:
        """Show recording state."""
        if self._root is None:
            return

        self._state = OverlayState.RECORDING
        self._smoothed_level = 0.0
        self._bar_heights = [BAR_MIN_HEIGHT] * NUM_BARS
        self._draw_recording()
        self._root.deiconify()
        logger.debug("overlay: showing recording state")

    def _do_show_transcribing(self) -> None:
        """Show transcribing state."""
        if self._root is None:
            return

        self._state = OverlayState.TRANSCRIBING
        self._spinner_angle = 0.0
        self._draw_transcribing()
        self._root.deiconify()
        logger.debug("overlay: showing transcribing state")

    def _do_show_error(self, message: str) -> None:
        """Show error flash briefly."""
        if self._root is None or self._canvas is None:
            return

        logger.debug("overlay: showing error, message=%s", message[:50])

        # Draw red pill briefly
        self._canvas.delete("all")
        self._draw_pill("#8B0000")  # Dark red/maroon
        self._root.deiconify()

        # Schedule hide after 500ms
        self._root.after(500, self._do_hide)

    def _do_hide(self) -> None:
        """Hide the overlay."""
        if self._root is None:
            return

        self._state = OverlayState.HIDDEN
        self._root.withdraw()
        logger.debug("overlay: hidden")

    def _do_quit(self) -> None:
        """Quit the Tkinter mainloop."""
        if self._root is not None:
            self._root.quit()
            self._root.destroy()
            self._root = None

    def _draw_pill(self, color: str) -> None:
        """Draw a pill-shaped background.

        Args:
            color: Fill color for the pill.
        """
        if self._canvas is None:
            return

        # Draw rounded rectangle (pill shape)
        x1, y1 = 0, 0
        x2, y2 = PILL_WIDTH, PILL_HEIGHT

        # Create pill using arcs and rectangles
        self._canvas.create_arc(
            x1, y1, x1 + PILL_HEIGHT, y2,
            start=90, extent=180, fill=color, outline=color,
        )
        self._canvas.create_arc(
            x2 - PILL_HEIGHT, y1, x2, y2,
            start=270, extent=180, fill=color, outline=color,
        )
        self._canvas.create_rectangle(
            x1 + PILL_RADIUS, y1, x2 - PILL_RADIUS, y2,
            fill=color, outline=color,
        )

    def _get_model_name(self) -> str:
        """Get the current model name from callback.

        Returns:
            Model name string, or empty string if unavailable.
        """
        if self._model_name_callback is not None:
            try:
                return self._model_name_callback()
            except Exception:
                pass
        return ""

    def _draw_model_name(self) -> float:
        """Draw the model name on the left side of the pill.

        Returns:
            The x position after the model name (for positioning other elements).
        """
        if self._canvas is None:
            return MODEL_LEFT_MARGIN

        model_name = self._get_model_name()
        if not model_name:
            return MODEL_LEFT_MARGIN

        center_y = PILL_HEIGHT // 2

        # Draw model name text
        text_id = self._canvas.create_text(
            MODEL_LEFT_MARGIN,
            center_y,
            text=model_name,
            font=MODEL_FONT,
            fill=MODEL_COLOR,
            anchor="w",
        )

        # Get text bounding box to calculate width
        bbox = self._canvas.bbox(text_id)
        if bbox:
            return bbox[2] + 8  # Right edge of text + margin
        return MODEL_LEFT_MARGIN + 50  # Fallback

    def _draw_recording(self) -> None:
        """Draw the recording state with model name and waveform bars."""
        if self._canvas is None:
            return

        self._canvas.delete("all")
        self._draw_pill(OVERLAY_COLOR)

        # Draw model name on left
        content_start_x = self._draw_model_name()

        # Calculate waveform area (right of model name, centered in remaining space)
        total_bar_width = NUM_BARS * BAR_WIDTH + (NUM_BARS - 1) * BAR_GAP
        available_width = PILL_WIDTH - content_start_x - PILL_RADIUS
        start_x = content_start_x + (available_width - total_bar_width) // 2
        center_y = PILL_HEIGHT // 2

        for i, height in enumerate(self._bar_heights):
            x = start_x + i * (BAR_WIDTH + BAR_GAP)
            y1 = center_y - height / 2
            y2 = center_y + height / 2

            # Draw bar
            self._canvas.create_rectangle(
                x, y1, x + BAR_WIDTH, y2,
                fill=BAR_COLOR, outline=BAR_COLOR,
            )

    def _draw_transcribing(self) -> None:
        """Draw the transcribing state with model name and rotating spinner."""
        if self._canvas is None:
            return

        self._canvas.delete("all")
        self._draw_pill(OVERLAY_COLOR)

        # Draw model name on left
        content_start_x = self._draw_model_name()

        # Calculate spinner position (centered in remaining space)
        available_width = PILL_WIDTH - content_start_x - PILL_RADIUS
        spinner_center_x = content_start_x + available_width // 2
        spinner_center_y = PILL_HEIGHT // 2

        # Draw rotating arc spinner
        # Calculate arc start angle (rotates over time)
        start_angle = self._spinner_angle

        # Draw the arc (270-degree arc that rotates)
        self._canvas.create_arc(
            spinner_center_x - SPINNER_RADIUS,
            spinner_center_y - SPINNER_RADIUS,
            spinner_center_x + SPINNER_RADIUS,
            spinner_center_y + SPINNER_RADIUS,
            start=start_angle,
            extent=SPINNER_ARC_EXTENT,
            style="arc",
            outline=MODEL_COLOR,
            width=SPINNER_WIDTH,
        )

    def _update_recording_animation(self) -> None:
        """Update waveform bars based on audio level."""
        if self._canvas is None:
            return

        # Get current audio level
        level = 0.0
        if self._audio_level_callback is not None:
            try:
                level = self._audio_level_callback()
            except Exception:
                level = 0.0

        # Apply EMA smoothing
        self._smoothed_level = (
            EMA_ALPHA * level + (1 - EMA_ALPHA) * self._smoothed_level
        )

        # Apply logarithmic scaling for better perception
        log_level = math.log10(self._smoothed_level * 9 + 1) if self._smoothed_level > 0 else 0

        # Calculate bar heights with center emphasis
        for i in range(NUM_BARS):
            # Distance from center (0 at center, 1 at edges)
            center_idx = (NUM_BARS - 1) / 2
            center_dist = abs(i - center_idx) / center_idx if center_idx > 0 else 0
            # Weight: 1.0 at center, 0.6 at edges
            weight = 1.0 - 0.4 * center_dist
            target_height = BAR_MIN_HEIGHT + (BAR_MAX_HEIGHT - BAR_MIN_HEIGHT) * log_level * weight

            # Smooth transition to target
            self._bar_heights[i] = (
                EMA_ALPHA * target_height + (1 - EMA_ALPHA) * self._bar_heights[i]
            )

        self._draw_recording()

    def _update_transcribing_animation(self) -> None:
        """Update spinning arc animation."""
        if self._canvas is None:
            return

        # Advance spinner angle (SPINNER_SPEED degrees per second)
        self._spinner_angle = (
            self._spinner_angle + SPINNER_SPEED * POLL_INTERVAL_MS / 1000.0
        ) % 360.0
        self._draw_transcribing()
