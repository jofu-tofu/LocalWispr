"""Settings window for LocalWispr.

This module provides a graphical settings interface using Tkinter,
allowing users to configure LocalWispr without editing config.toml directly.

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
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from localwispr.config import Config

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


class SettingsWindow:
    """Settings window for LocalWispr configuration.

    Provides a tabbed interface for configuring:
    - Recording mode and hotkeys
    - Audio feedback settings
    - Output/paste settings
    - Whisper model settings
    - Custom vocabulary

    Auto-saves on every change with 500ms debounce.

    Example:
        >>> window = SettingsWindow()
        >>> window.show()
    """

    # Window dimensions
    WINDOW_WIDTH = 450
    WINDOW_HEIGHT = 500

    # Debounce delay for auto-save (milliseconds)
    SAVE_DEBOUNCE_MS = 500

    def __init__(
        self,
        on_settings_changed: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the settings window.

        Args:
            on_settings_changed: Optional callback invoked after settings are saved.
                                 Used to notify the app to apply changes (e.g., restart hotkey listener).
        """
        self._window: tk.Toplevel | None = None
        self._root: tk.Tk | None = None

        # Callback for settings changes
        self._on_settings_changed = on_settings_changed

        # Config reference (loaded when window opens)
        self._config: Config | None = None

        # Tkinter variables for bindings
        self._vars: dict[str, tk.Variable] = {}

        # Debounced auto-save state
        self._save_timer: threading.Timer | None = None
        self._save_lock = threading.Lock()

    def show(self) -> None:
        """Show the settings window.

        Creates and displays the settings window. If already open,
        brings the existing window to focus.
        """
        # Run in a separate thread to avoid blocking
        thread = threading.Thread(target=self._create_window, daemon=True)
        thread.start()

    def _schedule_save(self, *args) -> None:
        """Schedule a debounced save (500ms after last change).

        Called by tkinter variable trace callbacks. Multiple rapid changes
        will only trigger one save after the debounce period.
        """
        with self._save_lock:
            if self._save_timer is not None:
                self._save_timer.cancel()
            self._save_timer = threading.Timer(
                self.SAVE_DEBOUNCE_MS / 1000.0,
                self._do_save,
            )
            self._save_timer.start()

    def _do_save(self) -> None:
        """Actually perform the save (called by debounce timer)."""
        from localwispr.config import reload_config, save_config

        # Build updated config from current UI state
        try:
            updated_config: Config = {
                "model": {
                    "name": self._vars["model_name"].get(),
                    "device": self._vars["device"].get(),
                    "compute_type": self._vars["compute_type"].get(),
                    "language": self._vars["language"].get(),
                },
                "hotkeys": {
                    "mode": self._vars["mode"].get(),
                    "modifiers": self._config["hotkeys"]["modifiers"],  # Keep existing
                    "audio_feedback": self._vars["audio_feedback"].get(),
                    "mute_system": self._vars["mute_system"].get(),
                },
                "context": self._config.get("context", {}),  # Keep existing
                "output": {
                    "auto_paste": self._vars["auto_paste"].get(),
                    "paste_delay_ms": self._vars["paste_delay_ms"].get(),
                },
                "vocabulary": {
                    "words": list(self._vocab_listbox.get(0, tk.END)),
                },
            }

            logger.debug("settings_window: auto-saving config=%s", updated_config)

            # Save to file
            save_config(updated_config)
            # Refresh the cached config so other modules see the new values
            reload_config()
            logger.info("settings_window: config auto-saved successfully")

            # Notify caller that settings changed
            if self._on_settings_changed is not None:
                try:
                    logger.debug("settings_window: invoking on_settings_changed callback")
                    self._on_settings_changed()
                except Exception as e:
                    logger.error("settings_window: on_settings_changed callback failed: %s", e)

        except Exception as e:
            logger.error("settings_window: auto-save failed, error=%s", e)

    def _setup_auto_save_traces(self) -> None:
        """Set up trace callbacks on all tkinter variables for auto-save."""
        # Variables to watch for changes
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
        ]

        for var_name in var_names:
            if var_name in self._vars:
                self._vars[var_name].trace_add("write", self._schedule_save)

        logger.debug("settings_window: auto-save traces set up for %d variables", len(var_names))

    def _create_window(self) -> None:
        """Create and run the settings window."""
        import os
        import sys
        from pathlib import Path
        from localwispr.config import load_config, _get_config_path

        # Comprehensive diagnostic - write to Desktop for guaranteed visibility
        diagnostic = []
        diagnostic.append("=== LocalWispr Settings Diagnostic ===")
        diagnostic.append(f"sys.frozen: {getattr(sys, 'frozen', False)}")
        diagnostic.append(f"sys.executable: {sys.executable}")
        diagnostic.append(f"os.getcwd(): {os.getcwd()}")

        config_path = _get_config_path()
        diagnostic.append(f"config_path: {config_path}")
        diagnostic.append(f"config_path.exists(): {config_path.exists()}")
        diagnostic.append(f"config_path.resolve(): {config_path.resolve()}")

        if config_path.exists():
            raw_content = config_path.read_text()
            diagnostic.append(f"\n=== Raw config.toml content ===\n{raw_content}")
            diagnostic.append(f"\n=== Key value checks ===")
            diagnostic.append(f"'toggle' in content: {'toggle' in raw_content}")
            diagnostic.append(f"'push-to-talk' in content: {'push-to-talk' in raw_content}")
            diagnostic.append(f"'audio_feedback = false' in content: {'audio_feedback = false' in raw_content.lower()}")
            diagnostic.append(f"'audio_feedback = true' in content: {'audio_feedback = true' in raw_content.lower()}")
        else:
            diagnostic.append("\n!!! CONFIG FILE DOES NOT EXIST !!!")
            parent = config_path.parent
            if parent.exists():
                diagnostic.append(f"Files in {parent}:")
                for f in parent.iterdir():
                    diagnostic.append(f"  - {f.name}")

        # Write to Desktop (guaranteed findable)
        desktop_file = Path.home() / "Desktop" / "localwispr_diagnostic.txt"
        try:
            desktop_file.write_text("\n".join(diagnostic))
        except Exception as e:
            diagnostic.append(f"Failed to write to desktop: {e}")

        # Load current config fresh from disk
        self._config = load_config()

        # Write loaded config to debug file
        try:
            with open(desktop_file, "a") as f:
                f.write(f"\n=== Loaded Config ===\n")
                f.write(f"hotkeys: {self._config.get('hotkeys')}\n")
                f.write(f"output: {self._config.get('output')}\n")
                f.write(f"audio_feedback type: {type(self._config['hotkeys']['audio_feedback'])}\n")
                f.write(f"audio_feedback value: {self._config['hotkeys']['audio_feedback']}\n")
                f.write(f"auto_paste type: {type(self._config['output']['auto_paste'])}\n")
                f.write(f"auto_paste value: {self._config['output']['auto_paste']}\n")
        except Exception:
            pass

        logger.debug("settings_window: loaded config hotkeys=%s", self._config.get("hotkeys"))
        logger.debug("settings_window: loaded config output=%s", self._config.get("output"))
        logger.debug("settings_window: loaded config model=%s", self._config.get("model"))

        # Create root window
        self._root = tk.Tk()
        self._root.withdraw()  # Hide root

        # Create toplevel window
        self._window = tk.Toplevel(self._root)
        self._window.title("LocalWispr Settings")
        self._window.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}")
        self._window.resizable(False, False)

        # Set window icon (optional, may fail on some systems)
        try:
            # Use a simple approach - no icon
            pass
        except Exception:
            pass

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

        # Create tabs
        self._create_general_tab(notebook)
        self._create_model_tab(notebook)
        self._create_vocabulary_tab(notebook)

        # Set up auto-save trace callbacks on all variables
        self._setup_auto_save_traces()

        # Create button frame with Close button only (auto-save handles persistence)
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        ttk.Button(
            button_frame,
            text="Close",
            command=self._on_close,
        ).pack(side=tk.RIGHT)

        # Handle window close
        self._window.protocol("WM_DELETE_WINDOW", self._on_close)

        # Run the window's event loop
        self._root.mainloop()

    def _create_general_tab(self, notebook: ttk.Notebook) -> None:
        """Create the General settings tab.

        Args:
            notebook: Parent notebook widget.
        """
        tab = ttk.Frame(notebook, padding="10")
        notebook.add(tab, text="General")

        # Recording Mode section
        mode_frame = ttk.LabelFrame(tab, text="Recording Mode", padding="10")
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        self._vars["mode"] = tk.StringVar(
            master=self._root,
            value=self._config["hotkeys"]["mode"]
        )

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

        modifiers = self._config["hotkeys"]["modifiers"]
        hotkey_str = " + ".join(mod.capitalize() for mod in modifiers)

        ttk.Label(
            hotkey_frame,
            text=f"Current: {hotkey_str}",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(anchor=tk.W)

        ttk.Label(
            hotkey_frame,
            text="(Edit config.toml to change hotkey)",
            foreground="gray",
        ).pack(anchor=tk.W)

        # Audio Feedback section
        audio_frame = ttk.LabelFrame(tab, text="Audio", padding="10")
        audio_frame.pack(fill=tk.X, pady=(0, 10))

        self._vars["audio_feedback"] = tk.BooleanVar(
            master=self._root,
            value=self._config["hotkeys"]["audio_feedback"]
        )

        ttk.Checkbutton(
            audio_frame,
            text="Play sounds when recording starts/stops",
            variable=self._vars["audio_feedback"],
            onvalue=True,
            offvalue=False,
        ).pack(anchor=tk.W)

        self._vars["mute_system"] = tk.BooleanVar(
            master=self._root,
            value=self._config["hotkeys"].get("mute_system", False)
        )

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

        self._vars["auto_paste"] = tk.BooleanVar(
            master=self._root,
            value=self._config["output"]["auto_paste"]
        )

        ttk.Checkbutton(
            output_frame,
            text="Auto-paste after transcription",
            variable=self._vars["auto_paste"],
            onvalue=True,
            offvalue=False,
        ).pack(anchor=tk.W)

        # Debug: Log the actual tkinter variable values after creation
        try:
            import sys
            from pathlib import Path
            if getattr(sys, 'frozen', False):
                debug_file = Path(sys.executable).parent / "settings_debug.txt"
            else:
                debug_file = Path.cwd() / "settings_debug.txt"
            with open(debug_file, "a") as f:
                f.write(f"\n=== Tkinter Variables After Creation ===\n")
                f.write(f"audio_feedback BooleanVar.get(): {self._vars['audio_feedback'].get()}\n")
                f.write(f"mute_system BooleanVar.get(): {self._vars['mute_system'].get()}\n")
                f.write(f"auto_paste BooleanVar.get(): {self._vars['auto_paste'].get()}\n")
                f.write(f"mode StringVar.get(): {self._vars['mode'].get()}\n")
        except Exception:
            pass

        # Paste delay
        delay_frame = ttk.Frame(output_frame)
        delay_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(delay_frame, text="Paste delay:").pack(side=tk.LEFT)

        self._vars["paste_delay_ms"] = tk.IntVar(
            master=self._root,
            value=self._config["output"]["paste_delay_ms"]
        )

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

    def _create_model_tab(self, notebook: ttk.Notebook) -> None:
        """Create the Model settings tab.

        Args:
            notebook: Parent notebook widget.
        """
        tab = ttk.Frame(notebook, padding="10")
        notebook.add(tab, text="Model")

        # Model selection
        model_frame = ttk.LabelFrame(tab, text="Whisper Model", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 10))

        self._vars["model_name"] = tk.StringVar(
            master=self._root,
            value=self._config["model"]["name"]
        )

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
        )
        size_info.pack(anchor=tk.W)

        # Device selection
        device_frame = ttk.LabelFrame(tab, text="Device", padding="10")
        device_frame.pack(fill=tk.X, pady=(0, 10))

        self._vars["device"] = tk.StringVar(
            master=self._root,
            value=self._config["model"]["device"]
        )

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
        ).pack(anchor=tk.W)

        # Compute type selection
        compute_frame = ttk.LabelFrame(tab, text="Compute Type", padding="10")
        compute_frame.pack(fill=tk.X, pady=(0, 10))

        self._vars["compute_type"] = tk.StringVar(
            master=self._root,
            value=self._config["model"]["compute_type"]
        )

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
        ).pack(anchor=tk.W)

        # Language selection
        lang_frame = ttk.LabelFrame(tab, text="Language", padding="10")
        lang_frame.pack(fill=tk.X, pady=(0, 10))

        # Get current language or default to auto
        current_lang = self._config.get("model", {}).get("language", "auto")
        self._vars["language"] = tk.StringVar(master=self._root, value=current_lang)

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

        # Load existing vocabulary
        vocab = self._config.get("vocabulary", {}).get("words", [])
        for word in vocab:
            self._vocab_listbox.insert(tk.END, word)

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

    def _add_vocab_word(self) -> None:
        """Add a word to the vocabulary list."""
        word = self._vocab_entry.get().strip()
        if word:
            # Check for duplicates
            existing = list(self._vocab_listbox.get(0, tk.END))
            if word not in existing:
                self._vocab_listbox.insert(tk.END, word)
                # Trigger auto-save after adding word
                self._schedule_save()
            self._vocab_entry.delete(0, tk.END)

    def _remove_vocab_words(self) -> None:
        """Remove selected words from the vocabulary list."""
        # Get selected indices in reverse order to avoid index shifting
        selected = list(self._vocab_listbox.curselection())
        if selected:
            for index in reversed(selected):
                self._vocab_listbox.delete(index)
            # Trigger auto-save after removing words
            self._schedule_save()

    def _on_close(self) -> None:
        """Handle Close button click or window close.

        Cancels any pending save timer and performs a final immediate save
        to ensure all changes are persisted before closing.
        """
        # Cancel any pending save timer
        with self._save_lock:
            if self._save_timer is not None:
                self._save_timer.cancel()
                self._save_timer = None

        # Perform a final immediate save to ensure everything is persisted
        self._do_save()

        self._close()

    def _close(self) -> None:
        """Close the settings window."""
        if self._window is not None:
            self._window.destroy()
        if self._root is not None:
            self._root.quit()
            self._root.destroy()
        self._window = None
        self._root = None


def open_settings(on_settings_changed: Callable[[], None] | None = None) -> None:
    """Open the settings window.

    Convenience function to create and show the settings window.

    Args:
        on_settings_changed: Optional callback invoked after settings are saved.
                             Used to notify the app to apply changes (e.g., restart hotkey listener).
    """
    window = SettingsWindow(on_settings_changed=on_settings_changed)
    window.show()
