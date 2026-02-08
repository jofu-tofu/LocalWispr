"""Tab creation methods for the settings window.

Mixin class providing General and Vocabulary tab builders
for TkinterSettingsView.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import messagebox, ttk

from pynput import keyboard
from pynput.keyboard import Key

logger = logging.getLogger(__name__)

# Modifier key mapping: pynput key -> canonical name
_KEY_TO_MODIFIER: dict[Key, str] = {
    Key.cmd: "win", Key.cmd_l: "win", Key.cmd_r: "win",
    Key.ctrl: "ctrl", Key.ctrl_l: "ctrl", Key.ctrl_r: "ctrl",
    Key.shift: "shift", Key.shift_l: "shift", Key.shift_r: "shift",
    Key.alt: "alt", Key.alt_l: "alt", Key.alt_r: "alt", Key.alt_gr: "alt",
}


class TabsMixin:
    """Tab creation methods for TkinterSettingsView.

    Provides the General tab (recording mode, hotkeys, audio, output, streaming)
    and Vocabulary tab (custom word list management).
    """

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

        # Hotkey section
        hotkey_frame = ttk.LabelFrame(tab, text="Hotkeys", padding="10")
        hotkey_frame.pack(fill=tk.X, pady=(0, 10))

        # --- Recording Hotkey ---
        ttk.Label(hotkey_frame, text="Recording Hotkey:").pack(anchor=tk.W)

        self._hotkey_label = ttk.Label(
            hotkey_frame,
            text="Current: (loading...)",
            font=("TkDefaultFont", 10, "bold"),
        )
        self._hotkey_label.pack(anchor=tk.W)

        self._record_btn = ttk.Button(
            hotkey_frame, text="Record", width=8,
            command=lambda: self._toggle_record_btn("recording"),
        )
        self._record_btn.pack(anchor=tk.W, pady=(2, 8))

        # --- Mode Cycle Hotkey ---
        ttk.Label(hotkey_frame, text="Mode Cycle Hotkey:").pack(anchor=tk.W)

        self._mode_cycle_label = ttk.Label(
            hotkey_frame,
            text="Current: (loading...)",
            font=("TkDefaultFont", 10, "bold"),
        )
        self._mode_cycle_label.pack(anchor=tk.W)

        self._mc_record_btn = ttk.Button(
            hotkey_frame, text="Record", width=8,
            command=lambda: self._toggle_record_btn("mode_cycle"),
        )
        self._mc_record_btn.pack(anchor=tk.W, pady=(2, 0))

        # Initialize recording state
        self._recording_target: str | None = None
        self._recorded_keys: set[str] = set()
        self._original_modifiers: tuple[str, ...] = ()
        self._key_listener: keyboard.Listener | None = None

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

    # --- Key Recorder Methods ---

    def _toggle_record_btn(self, target: str) -> None:
        """Toggle between Record and Stop for a hotkey button.

        Args:
            target: "recording" for recording hotkey, "mode_cycle" for mode cycle.
        """
        if self._recording_target == target:
            # Currently recording this target — stop
            self._stop_recording(target)
        else:
            # Not recording this target — start
            self._start_recording(target)

    def _start_recording(self, target: str) -> None:
        """Start recording a hotkey combination.

        Args:
            target: "recording" for recording hotkey, "mode_cycle" for mode cycle.
        """
        # Cancel any existing recording first
        if self._recording_target is not None:
            self._cancel_recording()

        self._recording_target = target

        # Save original modifiers for restore on cancel
        if target == "recording":
            self._original_modifiers = tuple(
                m.strip() for m in self._vars["hotkey_modifiers"].get().split(",") if m.strip()
            )
        else:
            self._original_modifiers = tuple(
                m.strip() for m in self._vars["mode_cycle_modifiers"].get().split(",") if m.strip()
            )

        self._recorded_keys = set()

        # Update UI: button becomes "Stop", show prompt
        if target == "recording":
            self._record_btn.config(text="Stop")
            self._hotkey_label.config(text="Press keys...")
        else:
            self._mc_record_btn.config(text="Stop")
            self._mode_cycle_label.config(text="Press keys...")

        # Suppress global hotkeys while recording keybinds
        from localwispr.hotkeys import suppress_hotkeys
        suppress_hotkeys()

        # Start pynput listener
        self._key_listener = keyboard.Listener(on_press=self._on_record_key_press)
        self._key_listener.start()

        logger.debug("keybind_recorder: started recording for %s", target)

    def _stop_recording(self, target: str) -> None:
        """Stop recording and validate the result.

        Args:
            target: "recording" or "mode_cycle".
        """
        if self._recording_target != target:
            return

        recorded = self._recorded_keys.copy()

        # Validate: minimum 2 modifiers
        if len(recorded) < 2:
            messagebox.showwarning(
                "Invalid Hotkey",
                "At least 2 modifier keys required.\nKeep recording or cancel.",
            )
            return

        # Valid — save to StringVar and update label
        modifiers_str = ",".join(sorted(recorded))
        label_str = " + ".join(mod.capitalize() for mod in sorted(recorded))

        if target == "recording":
            self._vars["hotkey_modifiers"].set(modifiers_str)
            self._hotkey_label.config(text=f"Current: {label_str}")
        else:
            self._vars["mode_cycle_modifiers"].set(modifiers_str)
            self._mode_cycle_label.config(text=f"Current: {label_str}")

        self._finish_recording()

        # Notify controller of change
        if self.on_setting_changed is not None:
            self.on_setting_changed(f"{target}_modifiers", None)

        logger.debug("keybind_recorder: saved %s = %s", target, modifiers_str)

    def _cancel_recording(self) -> None:
        """Cancel recording and restore original modifiers."""
        if self._recording_target is None:
            return

        target = self._recording_target
        original_str = " + ".join(mod.capitalize() for mod in self._original_modifiers)

        if target == "recording":
            self._hotkey_label.config(text=f"Current: {original_str}")
        else:
            self._mode_cycle_label.config(text=f"Current: {original_str}")

        self._finish_recording()
        logger.debug("keybind_recorder: cancelled recording for %s", target)

    def _finish_recording(self) -> None:
        """Clean up recording state (shared by stop and cancel)."""
        # Re-enable global hotkeys
        from localwispr.hotkeys import unsuppress_hotkeys
        unsuppress_hotkeys()

        # Stop pynput listener
        if self._key_listener is not None:
            try:
                self._key_listener.stop()
            except Exception:
                pass
            self._key_listener = None

        target = self._recording_target
        self._recording_target = None
        self._recorded_keys = set()

        # Reset button text back to "Record"
        if target == "recording":
            self._record_btn.config(text="Record")
        elif target == "mode_cycle":
            self._mc_record_btn.config(text="Record")

    def _on_record_key_press(self, key: Key | keyboard.KeyCode | None) -> None:
        """Handle key press during recording (runs on pynput thread).

        Args:
            key: The key that was pressed.
        """
        if key is None:
            return

        # Map to modifier name
        modifier = _KEY_TO_MODIFIER.get(key)
        if modifier is None:
            # Also check vk for Windows key
            if hasattr(key, 'vk') and key.vk in (0x5B, 0x5C):
                modifier = "win"
            else:
                return

        self._recorded_keys.add(modifier)

        # Schedule UI update on main thread
        if self._root is not None:
            self._root.after(0, self._update_recording_label)

    def _update_recording_label(self) -> None:
        """Update the label to show currently recorded keys (main thread)."""
        if self._recording_target is None:
            return

        if self._recorded_keys:
            label_str = " + ".join(mod.capitalize() for mod in sorted(self._recorded_keys))
        else:
            label_str = "Press keys..."

        if self._recording_target == "recording":
            self._hotkey_label.config(text=label_str)
        else:
            self._mode_cycle_label.config(text=label_str)

    def _on_tab_changed(self, event=None) -> None:
        """Handle notebook tab change — cancel active recording."""
        if self._recording_target is not None:
            self._cancel_recording()

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
