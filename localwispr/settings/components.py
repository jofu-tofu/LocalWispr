"""Reusable UI components for LocalWispr.

This module provides helper functions for creating consistent Tkinter UI elements:
- Labeled frames with standard padding
- Radio button groups
- Checkbox groups
- Combobox dropdowns with labels

These helpers reduce code repetition in settings_window.py and ensure
consistent styling across the application.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Sequence


def create_labeled_frame(
    parent: tk.Widget,
    text: str,
    padding: str = "10",
    fill: str = tk.X,
    pady: tuple[int, int] = (0, 10),
) -> ttk.LabelFrame:
    """Create a labeled frame with standard styling.

    Args:
        parent: Parent widget.
        text: Frame label text.
        padding: Internal padding (default "10").
        fill: Fill direction (default X).
        pady: Vertical padding tuple (default (0, 10)).

    Returns:
        Configured LabelFrame widget.
    """
    frame = ttk.LabelFrame(parent, text=text, padding=padding)
    frame.pack(fill=fill, pady=pady)
    return frame


def create_radio_group(
    parent: tk.Widget,
    variable: tk.Variable,
    options: Sequence[tuple[str, Any]],
    anchor: str = tk.W,
    pady: int = 0,
) -> list[ttk.Radiobutton]:
    """Create a group of radio buttons.

    Args:
        parent: Parent widget.
        variable: Tkinter variable to bind to.
        options: List of (label, value) tuples.
        anchor: Widget anchor (default W for left-align).
        pady: Vertical padding between buttons (default 0).

    Returns:
        List of created Radiobutton widgets.
    """
    buttons = []
    for i, (label, value) in enumerate(options):
        button = ttk.Radiobutton(
            parent,
            text=label,
            variable=variable,
            value=value,
        )
        button.pack(anchor=anchor, pady=(pady if i > 0 else 0, 0))
        buttons.append(button)
    return buttons


def create_checkbox(
    parent: tk.Widget,
    text: str,
    variable: tk.BooleanVar,
    anchor: str = tk.W,
    pady: tuple[int, int] = (0, 0),
) -> ttk.Checkbutton:
    """Create a checkbox with standard styling.

    Args:
        parent: Parent widget.
        text: Checkbox label text.
        variable: BooleanVar to bind to.
        anchor: Widget anchor (default W for left-align).
        pady: Vertical padding tuple.

    Returns:
        Configured Checkbutton widget.
    """
    checkbox = ttk.Checkbutton(
        parent,
        text=text,
        variable=variable,
        onvalue=True,
        offvalue=False,
    )
    checkbox.pack(anchor=anchor, pady=pady)
    return checkbox


def create_combobox(
    parent: tk.Widget,
    variable: tk.Variable,
    values: Sequence[str],
    label: str | None = None,
    description: str | None = None,
    width: int = 20,
    state: str = "readonly",
) -> ttk.Combobox:
    """Create a combobox with optional label and description.

    Args:
        parent: Parent widget.
        variable: Tkinter variable to bind to.
        values: List of option values.
        label: Optional label above the combobox.
        description: Optional gray description text below.
        width: Combobox width (default 20).
        state: Widget state (default "readonly").

    Returns:
        Configured Combobox widget.
    """
    if label:
        ttk.Label(parent, text=label).pack(anchor=tk.W)

    combo = ttk.Combobox(
        parent,
        textvariable=variable,
        values=list(values),
        state=state,
        width=width,
    )
    combo.pack(anchor=tk.W, pady=(2, 5))

    if description:
        ttk.Label(
            parent,
            text=description,
            foreground="gray",
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

    return combo


def create_spinbox_row(
    parent: tk.Widget,
    label: str,
    variable: tk.IntVar,
    from_: int,
    to: int,
    increment: int = 1,
    suffix: str = "",
    width: int = 6,
) -> ttk.Spinbox:
    """Create a spinbox with label and optional suffix.

    Args:
        parent: Parent widget.
        label: Label text before spinbox.
        variable: IntVar to bind to.
        from_: Minimum value.
        to: Maximum value.
        increment: Step increment (default 1).
        suffix: Text after spinbox (e.g., "ms").
        width: Spinbox width (default 6).

    Returns:
        Configured Spinbox widget.
    """
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, pady=(5, 0))

    ttk.Label(frame, text=label).pack(side=tk.LEFT)

    spinbox = ttk.Spinbox(
        frame,
        from_=from_,
        to=to,
        increment=increment,
        width=width,
        textvariable=variable,
    )
    spinbox.pack(side=tk.LEFT, padx=(5, 0))

    if suffix:
        ttk.Label(frame, text=suffix).pack(side=tk.LEFT, padx=(2, 0))

    return spinbox


def create_info_label(
    parent: tk.Widget,
    text: str,
    foreground: str = "gray",
    justify: str = tk.LEFT,
    anchor: str = tk.W,
) -> ttk.Label:
    """Create an informational label with gray text.

    Args:
        parent: Parent widget.
        text: Label text.
        foreground: Text color (default "gray").
        justify: Text justification (default LEFT).
        anchor: Widget anchor (default W).

    Returns:
        Configured Label widget.
    """
    label = ttk.Label(
        parent,
        text=text,
        foreground=foreground,
        justify=justify,
    )
    label.pack(anchor=anchor)
    return label


def create_button_row(
    parent: tk.Widget,
    buttons: Sequence[tuple[str, Callable[[], None]]],
    side: str = tk.RIGHT,
    padx: int = 5,
) -> list[ttk.Button]:
    """Create a row of buttons.

    Args:
        parent: Parent widget.
        buttons: List of (label, callback) tuples.
        side: Pack side (default RIGHT).
        padx: Horizontal padding between buttons (default 5).

    Returns:
        List of created Button widgets.
    """
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X)

    created_buttons = []
    for i, (label, callback) in enumerate(buttons):
        button = ttk.Button(
            frame,
            text=label,
            command=callback,
        )
        button.pack(side=side, padx=(padx if i > 0 else 0, 0))
        created_buttons.append(button)

    return created_buttons


__all__ = [
    "create_labeled_frame",
    "create_radio_group",
    "create_checkbox",
    "create_combobox",
    "create_spinbox_row",
    "create_info_label",
    "create_button_row",
]
