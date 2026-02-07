"""First-run wizard for LocalWispr.

Shows a model selection and download dialog on first run when no model
is installed. Guides users to select an appropriate model size based on
their system capabilities.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable

from localwispr.transcribe.model_manager import (
    MODEL_SIZES_MB,
    download_model,
    get_model_path,
    is_model_downloaded,
)

logger = logging.getLogger(__name__)

# Model options with descriptions
MODEL_OPTIONS = [
    {
        "name": "base",
        "display": "Base (Recommended)",
        "size_mb": 145,
        "description": "Good accuracy, fast. Best for most users.",
        "recommended": True,
    },
    {
        "name": "tiny",
        "display": "Tiny",
        "size_mb": 75,
        "description": "Fastest, basic accuracy. Good for quick start.",
        "recommended": False,
    },
    {
        "name": "small",
        "display": "Small",
        "size_mb": 465,
        "description": "Better accuracy, moderate speed.",
        "recommended": False,
    },
    {
        "name": "medium",
        "display": "Medium",
        "size_mb": 1500,
        "description": "Great accuracy, slower. For accuracy-focused users.",
        "recommended": False,
    },
    {
        "name": "large-v3",
        "display": "Large v3",
        "size_mb": 3000,
        "description": "Best accuracy, slowest. Requires good hardware.",
        "recommended": False,
    },
]


def get_user_config_path() -> Path:
    """Get the path to the user's config file.

    Returns:
        Path to user config file.
    """
    from localwispr.config import get_user_config_path as _get_config_path

    return _get_config_path()


def is_first_run() -> bool:
    """Check if this is the first run (no model is downloaded).

    We check if any model is downloaded, not just if config exists,
    because users may have config but no models (e.g., after reinstall).

    Returns:
        True if no whisper model is downloaded.
    """
    # Check if any of our supported models are downloaded
    for option in MODEL_OPTIONS:
        if is_model_downloaded(option["name"]):
            logger.debug("first_run: found model %s", option["name"])
            return False

    logger.info("first_run: no models found, showing wizard")
    return True


class FirstRunWizard:
    """First-run wizard window for model selection and download."""

    def __init__(self, on_complete: Callable[[str], None]) -> None:
        """Initialize the wizard.

        Args:
            on_complete: Callback called with selected model name when wizard completes.
        """
        self._on_complete = on_complete
        self._selected_model = tk.StringVar(value="base")
        self._download_thread: threading.Thread | None = None
        self._cancelled = False

        # Create main window
        self._root = tk.Tk()
        self._root.title("LocalWispr Setup")
        self._root.geometry("500x450")
        self._root.resizable(False, False)

        # Center window on screen
        self._root.update_idletasks()
        width = self._root.winfo_width()
        height = self._root.winfo_height()
        x = (self._root.winfo_screenwidth() // 2) - (width // 2)
        y = (self._root.winfo_screenheight() // 2) - (height // 2)
        self._root.geometry(f"+{x}+{y}")

        # Handle window close
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create the wizard UI."""
        # Main container with padding
        main_frame = ttk.Frame(self._root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Welcome to LocalWispr",
            font=("Segoe UI", 16, "bold"),
        )
        title_label.pack(pady=(0, 10))

        # Description
        desc_label = ttk.Label(
            main_frame,
            text="Select a speech recognition model to download.\n"
            "You can change this later in settings.",
            font=("Segoe UI", 10),
            justify=tk.CENTER,
        )
        desc_label.pack(pady=(0, 20))

        # Model selection frame
        model_frame = ttk.LabelFrame(main_frame, text="Choose Model", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 15))

        for option in MODEL_OPTIONS:
            frame = ttk.Frame(model_frame)
            frame.pack(fill=tk.X, pady=2)

            # Radio button
            radio = ttk.Radiobutton(
                frame,
                text=f"{option['display']} ({option['size_mb']} MB)",
                variable=self._selected_model,
                value=option["name"],
            )
            radio.pack(side=tk.LEFT)

            # Description label
            desc = ttk.Label(
                frame,
                text=f" - {option['description']}",
                font=("Segoe UI", 9),
                foreground="gray",
            )
            desc.pack(side=tk.LEFT)

        # Progress section (initially hidden)
        self._progress_frame = ttk.Frame(main_frame)
        self._progress_frame.pack(fill=tk.X, pady=(0, 15))

        self._progress_label = ttk.Label(
            self._progress_frame,
            text="",
            font=("Segoe UI", 10),
        )
        self._progress_label.pack(pady=(0, 5))

        self._progress_bar = ttk.Progressbar(
            self._progress_frame,
            mode="determinate",
            length=400,
        )
        self._progress_bar.pack(fill=tk.X)

        # Hide progress initially
        self._progress_frame.pack_forget()

        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        self._download_button = ttk.Button(
            button_frame,
            text="Download & Continue",
            command=self._start_download,
        )
        self._download_button.pack(side=tk.RIGHT)

        self._cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=self._on_close,
        )
        self._cancel_button.pack(side=tk.RIGHT, padx=(0, 10))

        # Status label at bottom
        self._status_label = ttk.Label(
            main_frame,
            text="",
            font=("Segoe UI", 9),
            foreground="gray",
        )
        self._status_label.pack(side=tk.BOTTOM, pady=(10, 0))

    def _start_download(self) -> None:
        """Start downloading the selected model."""
        model_name = self._selected_model.get()
        model_size = MODEL_SIZES_MB.get(model_name, 0)

        # Check if already downloaded
        if is_model_downloaded(model_name):
            self._on_download_complete(model_name)
            return

        # Show progress UI
        self._progress_frame.pack(fill=tk.X, pady=(0, 15))
        self._progress_label.config(
            text=f"Downloading {model_name} model ({model_size} MB)..."
        )
        self._progress_bar["value"] = 0

        # Disable buttons
        self._download_button.config(state=tk.DISABLED)

        # Start download in background
        self._download_thread = threading.Thread(
            target=self._download_worker,
            args=(model_name,),
            daemon=True,
        )
        self._download_thread.start()

    def _download_worker(self, model_name: str) -> None:
        """Background worker for downloading model.

        Args:
            model_name: Name of the model to download.
        """
        try:

            def progress_callback(downloaded: int, total: int) -> None:
                if self._cancelled:
                    raise InterruptedError("Download cancelled")

                if total > 0:
                    percent = (downloaded / total) * 100
                    self._root.after(
                        0,
                        lambda: self._update_progress(downloaded, total, percent),
                    )

            download_model(model_name, progress_callback)

            if not self._cancelled:
                self._root.after(0, lambda: self._on_download_complete(model_name))

        except InterruptedError:
            logger.info("first_run: download cancelled")
        except Exception as e:
            logger.error("first_run: download failed: %s", e)
            if not self._cancelled:
                self._root.after(0, lambda: self._on_download_error(str(e)))

    def _update_progress(self, downloaded: int, total: int, percent: float) -> None:
        """Update the progress bar and label.

        Args:
            downloaded: Bytes downloaded.
            total: Total bytes.
            percent: Percentage complete.
        """
        self._progress_bar["value"] = percent

        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        self._progress_label.config(
            text=f"Downloading... {downloaded_mb:.1f} / {total_mb:.1f} MB ({percent:.0f}%)"
        )

    def _on_download_complete(self, model_name: str) -> None:
        """Called when download completes successfully.

        Args:
            model_name: Name of the downloaded model.
        """
        self._progress_label.config(text="Download complete!")
        self._progress_bar["value"] = 100
        self._status_label.config(text=f"Model saved to: {get_model_path(model_name)}")

        # Close wizard and call completion callback
        self._root.after(500, lambda: self._finish(model_name))

    def _on_download_error(self, error: str) -> None:
        """Called when download fails.

        Args:
            error: Error message.
        """
        self._progress_label.config(text=f"Download failed: {error}")
        self._progress_bar["value"] = 0
        self._download_button.config(state=tk.NORMAL)

    def _finish(self, model_name: str) -> None:
        """Finish the wizard and call completion callback.

        Args:
            model_name: Name of the selected model.
        """
        self._root.destroy()
        self._on_complete(model_name)

    def _on_close(self) -> None:
        """Handle window close."""
        self._cancelled = True
        self._root.destroy()

    def run(self) -> None:
        """Run the wizard dialog."""
        self._root.mainloop()


def show_first_run_wizard(on_complete: Callable[[str], None]) -> None:
    """Show the first-run wizard dialog.

    Args:
        on_complete: Callback called with selected model name when wizard completes.
    """
    wizard = FirstRunWizard(on_complete)
    wizard.run()


def ensure_model_available(model_name: str | None = None) -> str | None:
    """Ensure a model is available, showing wizard if needed.

    If no model is specified, checks if any model is available.
    Shows the first-run wizard if no model is downloaded.

    Args:
        model_name: Specific model name to check, or None to check any.

    Returns:
        Name of available model, or None if user cancelled wizard.
    """
    # Check if specific model is requested and available
    if model_name and is_model_downloaded(model_name):
        return model_name

    # Check if any model is available
    if not is_first_run():
        # Find first available model
        for option in MODEL_OPTIONS:
            if is_model_downloaded(option["name"]):
                return option["name"]

    # Show wizard and get result
    result: list[str | None] = [None]

    def on_complete(selected: str) -> None:
        result[0] = selected

    show_first_run_wizard(on_complete)
    return result[0]
