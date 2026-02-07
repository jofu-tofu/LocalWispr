"""Model manager UI methods for the settings window.

Mixin class providing model download, delete, and status display
functionality for TkinterSettingsView.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ModelManagerMixin:
    """Model manager UI methods for TkinterSettingsView.

    Provides the Model tab with model download/delete functionality,
    progress tracking, and CPU recommendation display.
    """

    def _create_model_tab(self, notebook: ttk.Notebook) -> None:
        """Create the Model settings tab.

        Args:
            notebook: Parent notebook widget.
        """
        tab = self._create_scrollable_tab(notebook, "Model")

        # Model Manager section (at the top)
        manager_frame = ttk.LabelFrame(tab, text="Model Manager", padding="10")
        manager_frame.pack(fill=tk.X, pady=(0, 10))

        # Treeview for models
        tree_frame = ttk.Frame(manager_frame)
        tree_frame.pack(fill=tk.X)

        # Create treeview with columns
        columns = ("size", "status")
        self._model_tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="tree headings",
            height=6,
            selectmode="browse",
        )

        # Configure tree column (model name)
        self._model_tree.heading("#0", text="Model")
        self._model_tree.column("#0", width=100, anchor="w")

        # Configure data columns
        self._model_tree.heading("size", text="Size")
        self._model_tree.heading("status", text="Status")
        self._model_tree.column("size", width=80, anchor="center")
        self._model_tree.column("status", width=110, anchor="center")

        # Scrollbar for treeview
        tree_scroll = ttk.Scrollbar(
            tree_frame, orient="vertical", command=self._model_tree.yview
        )
        self._model_tree.configure(yscrollcommand=tree_scroll.set)

        self._model_tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Button frame for download/delete
        btn_frame = ttk.Frame(manager_frame)
        btn_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(
            btn_frame,
            text="Download",
            command=self._on_download_selected,
            width=12,
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            btn_frame,
            text="Delete",
            command=self._on_delete_selected,
            width=12,
        ).pack(side=tk.LEFT)

        # Progress bar (hidden initially)
        self._download_progress = ttk.Progressbar(
            manager_frame,
            mode="determinate",
            length=300,
        )

        # Progress text (hidden initially)
        self._progress_text = ttk.Label(
            manager_frame,
            text="",
            foreground="gray",
            font=("TkDefaultFont", 8),
        )

        # CPU recommendation tip (shown when device is auto or cpu)
        self._cpu_recommendation = ttk.Label(
            manager_frame,
            text="Tip: For CPU, 'small' model (465 MB) is recommended for faster performance.",
            foreground="#666666",
            wraplength=380,
            font=("TkDefaultFont", 8),
        )
        # Initially hidden, shown based on device selection

        # Set up trace to update recommendation when device changes
        self._vars["device"].trace_add("write", self._on_model_or_device_change)

        # Model selection
        model_frame = ttk.LabelFrame(tab, text="Whisper Model", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(model_frame, text="Model size:").pack(anchor=tk.W)

        from localwispr.settings.window import MODEL_SIZES

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

        from localwispr.settings.window import DEVICES

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
            text="auto: Auto-detect (recommended)\n"
            "cuda: GPU (faster, requires NVIDIA GPU)\n"
            "cpu: CPU (slower, works everywhere)",
            foreground="gray",
            justify=tk.LEFT,
            wraplength=380,
        ).pack(anchor=tk.W)

        # Compute type selection
        compute_frame = ttk.LabelFrame(tab, text="Compute Type", padding="10")
        compute_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(compute_frame, text="Precision:").pack(anchor=tk.W)

        from localwispr.settings.window import COMPUTE_TYPES

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
            text="auto: Auto-select based on device (recommended)\n"
            "float16: Best for GPU (fast)\n"
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

        from localwispr.settings.window import LANGUAGES

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

    def _on_model_or_device_change(self, *args) -> None:
        """Handle device change - update CPU recommendation."""
        self._update_cpu_recommendation()

    def _refresh_model_tree(self) -> None:
        """Refresh the model manager treeview with current status."""
        if self._model_tree is None:
            return

        # Don't refresh while downloading
        if self._is_downloading:
            return

        try:
            from localwispr.transcribe.model_manager import get_all_models_status

            # Clear existing items
            for item in self._model_tree.get_children():
                self._model_tree.delete(item)

            # Add all models
            models = get_all_models_status()
            for model in models:
                # Format size
                if model["downloaded"]:
                    # Show actual disk size
                    disk_mb = model["disk_size"] / (1024 * 1024)
                    size_str = f"{disk_mb:.0f} MB"
                    status_str = "Downloaded"
                else:
                    size_str = f"~{model['size_mb']} MB"
                    status_str = "Not downloaded"

                self._model_tree.insert(
                    "",
                    tk.END,
                    iid=model["name"],
                    text=model["name"],
                    values=(size_str, status_str),
                    tags=("downloaded" if model["downloaded"] else "not_downloaded",),
                )

            # Style downloaded vs not downloaded
            self._model_tree.tag_configure("downloaded", foreground="green")
            self._model_tree.tag_configure("not_downloaded", foreground="orange")

            # Select currently configured model
            current_model = self._vars["model_name"].get()
            if self._model_tree.exists(current_model):
                self._model_tree.selection_set(current_model)
                self._model_tree.see(current_model)

        except Exception as e:
            logger.error("settings_view: failed to refresh model tree: %s", e)

    def _update_cpu_recommendation(self) -> None:
        """Show/hide CPU model recommendation based on device selection."""
        if self._cpu_recommendation is None:
            return

        device = self._vars["device"].get()
        if device in ("auto", "cpu"):
            # Show the recommendation
            self._cpu_recommendation.pack(anchor=tk.W, pady=(5, 0))
        else:
            # Hide the recommendation
            self._cpu_recommendation.pack_forget()

    def _on_download_selected(self) -> None:
        """Handle Download button click for selected model."""
        logger.info("settings_view: Download button clicked")
        print("DEBUG: Download button clicked")  # Console output for test build

        if self._is_downloading:
            logger.info("settings_view: Already downloading, ignoring click")
            print("DEBUG: Already downloading")
            return

        if self._model_tree is None:
            logger.error("settings_view: model_tree is None")
            print("DEBUG: model_tree is None")
            return

        # Get selected model
        selection = self._model_tree.selection()
        logger.info("settings_view: selection = %s", selection)
        print(f"DEBUG: selection = {selection}")

        if not selection:
            messagebox.showinfo("No Selection", "Please select a model to download.")
            return

        model_name = selection[0]
        logger.info("settings_view: selected model = %s", model_name)
        print(f"DEBUG: selected model = {model_name}")

        # Check if already downloaded
        from localwispr.transcribe.model_manager import is_model_downloaded

        if is_model_downloaded(model_name):
            messagebox.showinfo(
                "Already Downloaded",
                f"Model '{model_name}' is already downloaded.",
            )
            return

        logger.info("settings_view: starting download for model %s", model_name)
        print(f"DEBUG: Starting download for {model_name}")

        # Update UI state
        self._is_downloading = True
        self._downloading_model = model_name
        if self._download_progress is not None:
            self._download_progress.pack(anchor=tk.W, pady=(5, 0))
            self._download_progress["value"] = 0
        if self._progress_text is not None:
            self._progress_text.pack(anchor=tk.W)
            self._progress_text.config(text=f"Downloading {model_name}...")

        # Start download in background thread
        thread = threading.Thread(
            target=self._download_thread,
            args=(model_name,),
            daemon=True,
        )
        thread.start()

    def _on_delete_selected(self) -> None:
        """Handle Delete button click for selected model."""
        logger.info("settings_view: Delete button clicked")
        print("DEBUG: Delete button clicked")

        if self._is_downloading or self._model_tree is None:
            return

        # Get selected model
        selection = self._model_tree.selection()
        logger.info("settings_view: selection = %s", selection)
        print(f"DEBUG: selection = {selection}")

        if not selection:
            messagebox.showinfo("No Selection", "Please select a model to delete.")
            return

        model_name = selection[0]
        logger.info("settings_view: selected model = %s", model_name)
        print(f"DEBUG: selected model = {model_name}")

        # Check if downloaded
        from localwispr.transcribe.model_manager import is_model_downloaded

        if not is_model_downloaded(model_name):
            messagebox.showinfo(
                "Not Downloaded",
                f"Model '{model_name}' is not downloaded.",
            )
            return

        # Confirm deletion
        if not messagebox.askyesno(
            "Confirm Delete",
            f"Delete model '{model_name}'?\n\n"
            "This will free up disk space. You can download it again "
            "anytime using the Download button.",
        ):
            return

        # Delete the model
        from localwispr.transcribe.model_manager import delete_model

        logger.info("settings_view: deleting model %s", model_name)

        if delete_model(model_name):
            messagebox.showinfo(
                "Model Deleted",
                f"Model '{model_name}' has been deleted.",
            )
            self._refresh_model_tree()
        else:
            messagebox.showerror(
                "Delete Failed",
                f"Failed to delete model '{model_name}'.\n\n"
                "The model files may be in use or you may not have permission.",
            )

    def _download_thread(self, model_name: str) -> None:
        """Download model in background thread.

        Args:
            model_name: Name of the model to download.
        """
        print(f"DEBUG: Download thread started for {model_name}")
        logger.info("settings_view: download thread started for %s", model_name)

        try:
            from localwispr.transcribe.download_progress import download_model_with_progress

            def on_progress(current: int, total: int) -> None:
                """Progress callback - marshal to UI thread."""
                if self._root is not None:
                    self._root.after(0, lambda: self._update_download_progress(current, total))

            print(f"DEBUG: Calling download_model_with_progress for {model_name}")
            download_model_with_progress(model_name, on_progress)
            print(f"DEBUG: Download completed for {model_name}")

            # Success - notify UI thread
            if self._root is not None:
                self._root.after(0, self._on_download_complete)

        except Exception as e:
            error_msg = str(e)
            print(f"DEBUG: Download failed for {model_name}: {error_msg}")
            logger.error(
                "settings_view: download failed for %s: %s",
                model_name,
                error_msg,
            )
            # Error - notify UI thread (capture error_msg by value)
            if self._root is not None:
                self._root.after(0, lambda msg=error_msg: self._on_download_error(msg))

    def _update_download_progress(self, current: int, total: int) -> None:
        """Update download progress display.

        Args:
            current: Bytes downloaded so far.
            total: Total bytes to download.
        """
        if self._download_progress is None or self._progress_text is None:
            return

        if total > 0:
            percent = (current / total) * 100
            self._download_progress["value"] = percent

            # Format sizes in MB
            current_mb = current / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            self._progress_text.config(
                text=f"{current_mb:.1f} / {total_mb:.1f} MB ({percent:.0f}%)"
            )

    def _on_download_complete(self) -> None:
        """Handle successful download completion."""
        model_name = self._downloading_model
        self._is_downloading = False
        self._downloading_model = None

        # Hide progress UI
        if self._download_progress is not None:
            self._download_progress.pack_forget()
        if self._progress_text is not None:
            self._progress_text.pack_forget()

        # Refresh the model tree
        self._refresh_model_tree()

        # Show success message
        try:
            messagebox.showinfo(
                "Download Complete",
                f"Model '{model_name}' downloaded successfully!\n\n"
                "You can now use LocalWispr for transcription.",
            )
        except tk.TclError:
            pass

        logger.info("settings_view: model download complete for %s", model_name)

    def _on_download_error(self, error: str) -> None:
        """Handle download failure.

        Args:
            error: Error message to display.
        """
        model_name = self._downloading_model
        self._is_downloading = False
        self._downloading_model = None

        # Hide progress UI
        if self._download_progress is not None:
            self._download_progress.pack_forget()
        if self._progress_text is not None:
            self._progress_text.pack_forget()

        # Show error message
        try:
            messagebox.showerror(
                "Download Failed",
                f"Failed to download model '{model_name}':\n\n{error}\n\n"
                "Please check your internet connection and try again.",
            )
        except tk.TclError:
            pass

        logger.error("settings_view: model download failed for %s", model_name)
