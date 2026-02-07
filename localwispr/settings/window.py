"""Settings window for LocalWispr.

This module provides a Tkinter implementation of the SettingsViewProtocol.
It's a temporary GUI that will be replaced when migrating to a new framework.

Tabs:
    - General: Recording mode, hotkeys, audio feedback, output settings
    - Model: Whisper model selection, device, compute type
    - Vocabulary: Custom vocabulary/hotwords for better transcription
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Any, Callable

from localwispr.settings.window_models import ModelManagerMixin
from localwispr.settings.window_tabs import TabsMixin

if TYPE_CHECKING:
    from localwispr.settings.model import SettingsSnapshot, ValidationResult

logger = logging.getLogger(__name__)


# Model options
MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
DEVICES = ["auto", "cuda", "cpu"]
COMPUTE_TYPES = ["auto", "float16", "int8", "float32"]

# Language options (most common, with auto-detect)
LANGUAGES = [
    ("Auto-detect", "auto"),
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Dutch", "nl"),
    ("Russian", "ru"),
    ("Chinese", "zh"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
]


class TkinterSettingsView(TabsMixin, ModelManagerMixin):
    """Tkinter implementation of SettingsViewProtocol.

    Provides a tabbed interface for configuring:
    - Recording mode and hotkeys
    - Audio feedback settings
    - Output/paste settings
    - Whisper model settings
    - Custom vocabulary

    Explicit Save button approach - no auto-save.
    """

    # Window dimensions
    WINDOW_WIDTH = 450
    WINDOW_HEIGHT = 650

    # Base window title
    WINDOW_TITLE = "LocalWispr Settings"

    def __init__(self) -> None:
        """Initialize the settings view."""
        self._window: tk.Toplevel | None = None
        self._root: tk.Tk | None = None

        # Callbacks (set by controller)
        self.on_save_requested: Callable[[], None] | None = None
        self.on_cancel_requested: Callable[[], None] | None = None
        self.on_setting_changed: Callable[[str, Any], None] | None = None

        # Tkinter variables for bindings
        self._vars: dict[str, tk.Variable] = {}

        # Dirty state
        self._is_dirty = False

        # Save button reference for enabling/disabling
        self._save_button: ttk.Button | None = None

        # Vocabulary listbox reference
        self._vocab_listbox: tk.Listbox | None = None
        self._vocab_entry: ttk.Entry | None = None

        # Model Manager UI components
        self._model_tree: ttk.Treeview | None = None
        self._download_progress: ttk.Progressbar | None = None
        self._progress_text: ttk.Label | None = None
        self._cpu_recommendation: ttk.Label | None = None
        self._is_downloading = False
        self._downloading_model: str | None = None

    def populate(self, settings: "SettingsSnapshot") -> None:
        """Populate the view with settings values.

        Args:
            settings: Settings snapshot to display.
        """
        # Always store settings for reference (needed by collect())
        self._settings = settings

        if self._root is None:
            return  # UI not ready, will be populated in _create_and_run_window

        # Populate variables (traces are set up after this)
        self._vars["mode"].set(settings.hotkey_mode)
        self._vars["audio_feedback"].set(settings.audio_feedback)
        self._vars["mute_system"].set(settings.mute_system)
        self._vars["auto_paste"].set(settings.auto_paste)
        self._vars["paste_delay_ms"].set(settings.paste_delay_ms)
        self._vars["model_name"].set(settings.model_name)
        self._vars["device"].set(settings.model_device)
        self._vars["compute_type"].set(settings.model_compute_type)
        self._vars["language"].set(settings.model_language)
        self._vars["streaming_enabled"].set(settings.streaming_enabled)

        # Populate vocabulary list
        if self._vocab_listbox is not None:
            self._vocab_listbox.delete(0, tk.END)
            for word in settings.vocabulary_words:
                self._vocab_listbox.insert(tk.END, word)

        # Update hotkey display
        hotkey_str = " + ".join(mod.capitalize() for mod in settings.hotkey_modifiers)
        if hasattr(self, "_hotkey_label"):
            self._hotkey_label.config(text=f"Current: {hotkey_str}")

        # Update model manager display
        self._refresh_model_tree()
        self._update_cpu_recommendation()

        logger.debug("settings_view: populated with %s settings", len(self._vars))

    def collect(self) -> "SettingsSnapshot":
        """Collect current UI state into a settings snapshot.

        Returns:
            SettingsSnapshot reflecting current UI values.
        """
        from localwispr.settings.model import SettingsSnapshot

        # Get vocabulary words from listbox
        vocab_words: tuple[str, ...] = ()
        if self._vocab_listbox is not None:
            vocab_words = tuple(self._vocab_listbox.get(0, tk.END))

        return SettingsSnapshot(
            # Model
            model_name=self._vars["model_name"].get(),
            model_device=self._vars["device"].get(),
            model_compute_type=self._vars["compute_type"].get(),
            model_language=self._vars["language"].get(),
            # Hotkeys (modifiers from original, mode from UI)
            hotkey_mode=self._vars["mode"].get(),
            hotkey_modifiers=self._settings.hotkey_modifiers,  # Keep existing
            audio_feedback=self._vars["audio_feedback"].get(),
            mute_system=self._vars["mute_system"].get(),
            # Output
            auto_paste=self._vars["auto_paste"].get(),
            paste_delay_ms=self._vars["paste_delay_ms"].get(),
            # Vocabulary
            vocabulary_words=vocab_words,
            # Streaming
            streaming_enabled=self._vars["streaming_enabled"].get(),
            streaming_min_silence_ms=self._settings.streaming_min_silence_ms,
            streaming_max_segment_duration=self._settings.streaming_max_segment_duration,
            streaming_min_segment_duration=self._settings.streaming_min_segment_duration,
            streaming_overlap_ms=self._settings.streaming_overlap_ms,
        )

    def set_dirty(self, is_dirty: bool) -> None:
        """Update the dirty state indicator.

        Args:
            is_dirty: True if there are unsaved changes.
        """
        self._is_dirty = is_dirty

        # Update window title
        if self._window is not None:
            title = self.WINDOW_TITLE
            if is_dirty:
                title = f"* {title}"
            self._window.title(title)

        # Update Save button state
        if self._save_button is not None:
            if is_dirty:
                self._save_button.config(state=tk.NORMAL)
            else:
                self._save_button.config(state=tk.DISABLED)

    def show_validation_errors(self, result: "ValidationResult") -> None:
        """Display validation errors to the user.

        Args:
            result: Validation result with errors dict.
        """
        # Build error message
        lines = ["Cannot save settings due to the following errors:\n"]
        for field, error in result.errors.items():
            # Clean up field name for display
            display_field = field.replace("_", " ").title()
            lines.append(f"- {display_field}: {error}")

        message = "\n".join(lines)

        # Show error dialog
        if self._root is not None:
            try:
                messagebox.showerror("Validation Error", message)
            except tk.TclError:
                pass

    def confirm_discard_changes(self) -> bool:
        """Ask user to confirm discarding unsaved changes.

        Returns:
            True if user confirms discard, False to cancel close.
        """
        if self._root is None:
            return True

        try:
            return messagebox.askyesno(
                "Unsaved Changes",
                "You have unsaved changes.\n\nDiscard changes and close?",
                icon=messagebox.WARNING,
            )
        except tk.TclError:
            return True

    def close(self) -> None:
        """Close the settings window."""
        if self._window is not None:
            self._window.destroy()
        if self._root is not None:
            self._root.quit()
            self._root.destroy()
        self._window = None
        self._root = None
        logger.debug("settings_view: closed")

    def show(self) -> None:
        """Show the settings window.

        Creates the window if needed and starts the event loop.
        """
        # Run in a separate thread to avoid blocking
        thread = threading.Thread(target=self._create_and_run_window, daemon=True)
        thread.start()

    def _create_and_run_window(self) -> None:
        """Create and run the settings window."""
        # Create root window
        self._root = tk.Tk()
        self._root.withdraw()  # Hide root

        # Create toplevel window
        self._window = tk.Toplevel(self._root)
        self._window.title(self.WINDOW_TITLE)
        self._window.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}")
        self._window.resizable(False, False)

        # Center window on screen
        self._window.update_idletasks()
        x = (self._window.winfo_screenwidth() // 2) - (self.WINDOW_WIDTH // 2)
        y = (self._window.winfo_screenheight() // 2) - (self.WINDOW_HEIGHT // 2)
        self._window.geometry(f"+{x}+{y}")

        # Make window modal-like (stay on top)
        self._window.attributes("-topmost", True)

        # Create main frame with padding
        main_frame = ttk.Frame(self._window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create notebook (tabbed interface)
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Initialize variables before creating tabs
        self._init_variables()

        # Create tabs
        self._create_general_tab(notebook)
        self._create_model_tab(notebook)
        self._create_vocabulary_tab(notebook)

        # Create button frame with Save and Cancel buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        # Cancel button (left side)
        ttk.Button(
            button_frame,
            text="Cancel",
            command=self._on_cancel_click,
        ).pack(side=tk.LEFT)

        # Save button (right side) - initially disabled
        self._save_button = ttk.Button(
            button_frame,
            text="Save",
            command=self._on_save_click,
            state=tk.DISABLED,
        )
        self._save_button.pack(side=tk.RIGHT)

        # Handle window close
        self._window.protocol("WM_DELETE_WINDOW", self._on_cancel_click)

        # Populate with initial settings (set by controller before show())
        if hasattr(self, "_settings"):
            self.populate(self._settings)

        # Set up change traces AFTER populate to avoid false dirty state
        self._setup_change_traces()

        logger.info("settings_view: window created")

        # Run the window's event loop
        self._root.mainloop()

    def _init_variables(self) -> None:
        """Initialize Tkinter variables before creating widgets."""
        self._vars["mode"] = tk.StringVar(master=self._root, value="push-to-talk")
        self._vars["audio_feedback"] = tk.BooleanVar(master=self._root, value=True)
        self._vars["mute_system"] = tk.BooleanVar(master=self._root, value=False)
        self._vars["auto_paste"] = tk.BooleanVar(master=self._root, value=True)
        self._vars["paste_delay_ms"] = tk.IntVar(master=self._root, value=50)
        self._vars["model_name"] = tk.StringVar(master=self._root, value="large-v3")
        self._vars["device"] = tk.StringVar(master=self._root, value="cuda")
        self._vars["compute_type"] = tk.StringVar(master=self._root, value="float16")
        self._vars["language"] = tk.StringVar(master=self._root, value="auto")
        self._vars["streaming_enabled"] = tk.BooleanVar(master=self._root, value=False)

    def _setup_change_traces(self) -> None:
        """Set up trace callbacks on all tkinter variables for change detection."""
        var_names = [
            "mode",
            "audio_feedback",
            "mute_system",
            "auto_paste",
            "paste_delay_ms",
            "model_name",
            "device",
            "compute_type",
            "language",
            "streaming_enabled",
        ]

        for var_name in var_names:
            if var_name in self._vars:
                self._vars[var_name].trace_add("write", self._on_var_change)

        logger.debug("settings_view: change traces set up for %d variables", len(var_names))

    def _on_var_change(self, *args) -> None:
        """Handle variable change - notify controller."""
        if self.on_setting_changed is not None:
            # Get the variable name from the trace callback args
            var_name = args[0] if args else "unknown"
            try:
                self.on_setting_changed(var_name, None)
            except Exception as e:
                logger.error("settings_view: on_setting_changed callback failed: %s", e)

    def _on_save_click(self) -> None:
        """Handle Save button click."""
        if self.on_save_requested is not None:
            try:
                self.on_save_requested()
            except Exception as e:
                logger.error("settings_view: on_save_requested callback failed: %s", e)

    def _on_cancel_click(self) -> None:
        """Handle Cancel button click or window close."""
        if self.on_cancel_requested is not None:
            try:
                self.on_cancel_requested()
            except Exception as e:
                logger.error("settings_view: on_cancel_requested callback failed: %s", e)
        else:
            # No controller attached, just close
            self.close()

    def _create_scrollable_tab(self, notebook: ttk.Notebook, title: str) -> ttk.Frame:
        """Create a scrollable tab with canvas and inner frame.

        Args:
            notebook: Parent notebook widget.
            title: Tab title.

        Returns:
            The inner frame to pack widgets into.
        """
        # Outer frame holds canvas + scrollbar
        outer = ttk.Frame(notebook)
        notebook.add(outer, text=title)

        # Canvas for scrolling
        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)

        # Inner frame for actual content
        inner = ttk.Frame(canvas, padding="10")

        # Create window in canvas
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")

        # Configure scrolling
        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Make inner frame match canvas width
            canvas.itemconfig(canvas_window, width=canvas.winfo_width())

        inner.bind("<Configure>", on_configure)

        def on_canvas_configure(event):
            # Update inner frame width when canvas resizes
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind("<Configure>", on_canvas_configure)

        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Enable mousewheel scrolling when mouse is over this canvas
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)

        def unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", bind_mousewheel)
        canvas.bind("<Leave>", unbind_mousewheel)

        return inner


# Convenience alias for backward compatibility
SettingsWindow = TkinterSettingsView


def open_settings(on_settings_changed: Callable[[], None] | None = None) -> None:
    """Open the settings window.

    Convenience function that creates the controller and view, then opens settings.

    Args:
        on_settings_changed: Optional callback invoked after settings are saved.
                             Used to notify the app to apply changes.
    """
    from localwispr.settings.controller import SettingsController

    view = TkinterSettingsView()
    controller = SettingsController(view, on_settings_applied=on_settings_changed)
    controller.open()
