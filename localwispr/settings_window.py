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

if TYPE_CHECKING:
    from localwispr.settings_model import SettingsSnapshot, ValidationResult

logger = logging.getLogger(__name__)


# Model options
MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
DEVICES = ["cuda", "cpu"]
COMPUTE_TYPES = ["float16", "int8", "float32"]

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


class TkinterSettingsView:
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
    WINDOW_HEIGHT = 580

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

        logger.debug("settings_view: populated with %s settings", len(self._vars))

    def collect(self) -> "SettingsSnapshot":
        """Collect current UI state into a settings snapshot.

        Returns:
            SettingsSnapshot reflecting current UI values.
        """
        from localwispr.settings_model import SettingsSnapshot

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

        # Set up change traces on all variables
        self._setup_change_traces()

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

    def _create_general_tab(self, notebook: ttk.Notebook) -> None:
        """Create the General settings tab.

        Args:
            notebook: Parent notebook widget.
        """
        tab = self._create_scrollable_tab(notebook, "General")

        # Recording Mode section
        mode_frame = ttk.LabelFrame(tab, text="Recording Mode", padding="10")
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Radiobutton(
            mode_frame,
            text="Push-to-Talk (hold to record)",
            variable=self._vars["mode"],
            value="push-to-talk",
        ).pack(anchor=tk.W)

        ttk.Radiobutton(
            mode_frame,
            text="Toggle (press to start/stop)",
            variable=self._vars["mode"],
            value="toggle",
        ).pack(anchor=tk.W)

        # Hotkey display (read-only for now)
        hotkey_frame = ttk.LabelFrame(tab, text="Hotkey", padding="10")
        hotkey_frame.pack(fill=tk.X, pady=(0, 10))

        self._hotkey_label = ttk.Label(
            hotkey_frame,
            text="Current: (loading...)",
            font=("TkDefaultFont", 10, "bold"),
        )
        self._hotkey_label.pack(anchor=tk.W)

        ttk.Label(
            hotkey_frame,
            text="(Edit config.toml to change hotkey)",
            foreground="gray",
        ).pack(anchor=tk.W)

        # Audio Feedback section
        audio_frame = ttk.LabelFrame(tab, text="Audio", padding="10")
        audio_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Checkbutton(
            audio_frame,
            text="Play sounds when recording starts/stops",
            variable=self._vars["audio_feedback"],
            onvalue=True,
            offvalue=False,
        ).pack(anchor=tk.W)

        ttk.Checkbutton(
            audio_frame,
            text="Mute system audio during recording",
            variable=self._vars["mute_system"],
            onvalue=True,
            offvalue=False,
        ).pack(anchor=tk.W, pady=(5, 0))

        # Output section
        output_frame = ttk.LabelFrame(tab, text="Output", padding="10")
        output_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Checkbutton(
            output_frame,
            text="Auto-paste after transcription",
            variable=self._vars["auto_paste"],
            onvalue=True,
            offvalue=False,
        ).pack(anchor=tk.W)

        # Paste delay
        delay_frame = ttk.Frame(output_frame)
        delay_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(delay_frame, text="Paste delay:").pack(side=tk.LEFT)

        delay_spinbox = ttk.Spinbox(
            delay_frame,
            from_=0,
            to=500,
            increment=10,
            width=6,
            textvariable=self._vars["paste_delay_ms"],
        )
        delay_spinbox.pack(side=tk.LEFT, padx=(5, 0))

        ttk.Label(delay_frame, text="ms").pack(side=tk.LEFT, padx=(2, 0))

        # Advanced section (Streaming)
        advanced_frame = ttk.LabelFrame(tab, text="Advanced", padding="10")
        advanced_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Checkbutton(
            advanced_frame,
            text="Enable streaming transcription",
            variable=self._vars["streaming_enabled"],
            onvalue=True,
            offvalue=False,
        ).pack(anchor=tk.W)

        ttk.Label(
            advanced_frame,
            text="Processes audio in chunks during recording. "
            "Faster for long recordings (2+ minutes).",
            foreground="gray",
            justify=tk.LEFT,
            wraplength=380,
        ).pack(anchor=tk.W, pady=(2, 0))

    def _create_model_tab(self, notebook: ttk.Notebook) -> None:
        """Create the Model settings tab.

        Args:
            notebook: Parent notebook widget.
        """
        tab = self._create_scrollable_tab(notebook, "Model")

        # Model selection
        model_frame = ttk.LabelFrame(tab, text="Whisper Model", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(model_frame, text="Model size:").pack(anchor=tk.W)

        model_combo = ttk.Combobox(
            model_frame,
            textvariable=self._vars["model_name"],
            values=MODEL_SIZES,
            state="readonly",
            width=20,
        )
        model_combo.pack(anchor=tk.W, pady=(2, 5))

        # Model size descriptions
        size_info = ttk.Label(
            model_frame,
            text="tiny: Fastest, lowest accuracy\n"
            "base/small: Good balance\n"
            "medium: Better accuracy\n"
            "large-v3: Best accuracy (recommended)",
            foreground="gray",
            justify=tk.LEFT,
            wraplength=380,
        )
        size_info.pack(anchor=tk.W)

        # Device selection
        device_frame = ttk.LabelFrame(tab, text="Device", padding="10")
        device_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(device_frame, text="Processing device:").pack(anchor=tk.W)

        device_combo = ttk.Combobox(
            device_frame,
            textvariable=self._vars["device"],
            values=DEVICES,
            state="readonly",
            width=20,
        )
        device_combo.pack(anchor=tk.W, pady=(2, 5))

        ttk.Label(
            device_frame,
            text="cuda: GPU (faster, requires NVIDIA GPU)\n"
            "cpu: CPU (slower, works everywhere)",
            foreground="gray",
            justify=tk.LEFT,
            wraplength=380,
        ).pack(anchor=tk.W)

        # Compute type selection
        compute_frame = ttk.LabelFrame(tab, text="Compute Type", padding="10")
        compute_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(compute_frame, text="Precision:").pack(anchor=tk.W)

        compute_combo = ttk.Combobox(
            compute_frame,
            textvariable=self._vars["compute_type"],
            values=COMPUTE_TYPES,
            state="readonly",
            width=20,
        )
        compute_combo.pack(anchor=tk.W, pady=(2, 5))

        ttk.Label(
            compute_frame,
            text="float16: Best for GPU (fast)\n"
            "int8: Best for CPU (smaller memory)\n"
            "float32: Maximum precision (slow)",
            foreground="gray",
            justify=tk.LEFT,
            wraplength=380,
        ).pack(anchor=tk.W)

        # Language selection
        lang_frame = ttk.LabelFrame(tab, text="Language", padding="10")
        lang_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(lang_frame, text="Transcription language:").pack(anchor=tk.W)

        lang_combo = ttk.Combobox(
            lang_frame,
            textvariable=self._vars["language"],
            values=[code for _, code in LANGUAGES],
            state="readonly",
            width=20,
        )
        lang_combo.pack(anchor=tk.W, pady=(2, 5))

        # Show language name mapping
        lang_names = ", ".join(f"{code}={name}" for name, code in LANGUAGES[:6])
        ttk.Label(
            lang_frame,
            text=f"Codes: {lang_names}...",
            foreground="gray",
            wraplength=380,
        ).pack(anchor=tk.W)

    def _create_vocabulary_tab(self, notebook: ttk.Notebook) -> None:
        """Create the Vocabulary settings tab.

        Args:
            notebook: Parent notebook widget.
        """
        tab = ttk.Frame(notebook, padding="10")
        notebook.add(tab, text="Vocabulary")

        # Description
        ttk.Label(
            tab,
            text="Add custom words to improve transcription accuracy.\n"
            "Useful for technical terms, names, and acronyms.",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 10))

        # Vocabulary list frame
        list_frame = ttk.Frame(tab)
        list_frame.pack(fill=tk.BOTH, expand=True)

        # Listbox with scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._vocab_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            height=12,
            selectmode=tk.EXTENDED,
        )
        self._vocab_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self._vocab_listbox.yview)

        # Add word frame
        add_frame = ttk.Frame(tab)
        add_frame.pack(fill=tk.X, pady=(10, 0))

        self._vocab_entry = ttk.Entry(add_frame)
        self._vocab_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._vocab_entry.bind("<Return>", lambda e: self._add_vocab_word())

        ttk.Button(
            add_frame,
            text="Add",
            command=self._add_vocab_word,
            width=8,
        ).pack(side=tk.LEFT, padx=(5, 0))

        # Remove button
        ttk.Button(
            tab,
            text="Remove Selected",
            command=self._remove_vocab_words,
        ).pack(anchor=tk.E, pady=(5, 0))

    def _validate_vocab_word(self, word: str) -> tuple[bool, str]:
        """Validate a vocabulary word for TOML safety.

        Args:
            word: The word to validate.

        Returns:
            Tuple of (is_valid, cleaned_word_or_error_message).
        """
        word = word.strip()

        if not word:
            return False, "Word cannot be empty"

        if len(word) > 100:
            return False, "Word too long (max 100 characters)"

        # TOML-unsafe characters that would corrupt config
        unsafe_chars = {
            '"': 'double quote',
            '\\': 'backslash',
            '\n': 'newline',
            '\r': 'carriage return',
            '\t': 'tab',
        }
        for char, name in unsafe_chars.items():
            if char in word:
                return False, f"Word contains invalid character: {name}"

        return True, word

    def _add_vocab_word(self) -> None:
        """Add a word to the vocabulary list."""
        if self._vocab_entry is None or self._vocab_listbox is None:
            return

        word = self._vocab_entry.get()

        # Validate the word
        is_valid, result = self._validate_vocab_word(word)
        if not is_valid:
            if result != "Word cannot be empty":  # Don't show error for empty input
                messagebox.showwarning("Invalid Word", result)
            return

        word = result  # Use cleaned word

        # Check for duplicates
        existing = list(self._vocab_listbox.get(0, tk.END))
        if word in existing:
            messagebox.showinfo("Duplicate", f"'{word}' is already in the vocabulary list.")
            self._vocab_entry.delete(0, tk.END)
            return

        self._vocab_listbox.insert(tk.END, word)
        self._vocab_entry.delete(0, tk.END)

        # Notify controller of change
        if self.on_setting_changed is not None:
            self.on_setting_changed("vocabulary_words", None)

    def _remove_vocab_words(self) -> None:
        """Remove selected words from the vocabulary list."""
        if self._vocab_listbox is None:
            return

        # Get selected indices in reverse order to avoid index shifting
        selected = list(self._vocab_listbox.curselection())
        if selected:
            for index in reversed(selected):
                self._vocab_listbox.delete(index)

            # Notify controller of change
            if self.on_setting_changed is not None:
                self.on_setting_changed("vocabulary_words", None)


# Convenience alias for backward compatibility
SettingsWindow = TkinterSettingsView


def open_settings(on_settings_changed: Callable[[], None] | None = None) -> None:
    """Open the settings window.

    Convenience function that creates the controller and view, then opens settings.

    Args:
        on_settings_changed: Optional callback invoked after settings are saved.
                             Used to notify the app to apply changes.
    """
    from localwispr.settings_controller import SettingsController

    view = TkinterSettingsView()
    controller = SettingsController(view, on_settings_applied=on_settings_changed)
    controller.open()
