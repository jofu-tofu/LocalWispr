"""Download with progress reporting for LocalWispr.

This module provides model download functionality with progress callbacks,
delegating to model_manager.download_model() for the actual download.
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


class DownloadProgressCallback:
    """Progress callback wrapper for model downloads.

    Tracks download progress and reports to a callback function.
    """

    def __init__(self, callback: Callable[[int, int], None] | None = None) -> None:
        """Initialize the progress callback.

        Args:
            callback: Function called with (bytes_downloaded, total_bytes)
        """
        self._callback = callback
        self._total_bytes = 0
        self._downloaded_bytes = 0

    def __call__(self, progress: int, total: int) -> None:
        """Handle progress update.

        Args:
            progress: Bytes downloaded so far
            total: Total bytes to download
        """
        self._downloaded_bytes = progress
        self._total_bytes = total

        if self._callback is not None and total > 0:
            self._callback(progress, total)


def download_model_with_progress(
    model_name: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> str:
    """Download a Whisper model with progress reporting.

    Delegates to model_manager.download_model() which handles the actual
    download via urllib with progress callbacks.

    Args:
        model_name: Model name (e.g., "large-v3", "small")
        progress_callback: Optional callback called with (bytes_downloaded, total_bytes)

    Returns:
        Path to the downloaded model file as a string.

    Raises:
        Exception: If download fails.
    """
    from localwispr.transcribe.model_manager import download_model

    result_path = download_model(model_name, progress_callback)
    return str(result_path)
