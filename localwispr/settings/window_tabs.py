"""Tab creation methods for the settings window.

Mixin class providing General and Vocabulary tab builders
for TkinterSettingsView.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import messagebox, ttk

logger = logging.getLogger(__name__)


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
