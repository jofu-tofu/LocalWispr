"""Download with progress reporting for LocalWispr.

This module provides model download functionality with progress callbacks,
using huggingface_hub for reliable model downloads.
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


class DownloadProgressCallback:
    """Progress callback wrapper for huggingface_hub downloads.

    Tracks download progress across multiple files and reports
    aggregate progress to a callback function.
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
        """Handle progress update from huggingface_hub.

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

    Uses huggingface_hub's snapshot_download to download all model files.
    Progress is reported via the callback function.

    Args:
        model_name: Model name (e.g., "large-v3", "small")
        progress_callback: Optional callback called with (bytes_downloaded, total_bytes)

    Returns:
        Path to the downloaded model directory.

    Raises:
        Exception: If download fails.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

    from localwispr.transcribe.model_manager import get_repo_id

    repo_id = get_repo_id(model_name)
    logger.info("model_download: starting download for %s (%s)", model_name, repo_id)

    # Create progress tracker
    tracker = DownloadProgressCallback(progress_callback)

    try:
        # Download the model snapshot
        # Note: huggingface_hub doesn't expose fine-grained progress callbacks
        # in snapshot_download, so we use tqdm_class for progress tracking
        from tqdm import tqdm

        class ProgressTqdm(tqdm):
            """Custom tqdm that reports to our callback."""

            def __init__(self, *args, **kwargs):
                # Filter out unknown kwargs that huggingface_hub passes
                known_kwargs = {
                    'iterable', 'desc', 'total', 'leave', 'file', 'ncols',
                    'mininterval', 'maxinterval', 'miniters', 'ascii', 'disable',
                    'unit', 'unit_scale', 'dynamic_ncols', 'smoothing', 'bar_format',
                    'initial', 'position', 'postfix', 'unit_divisor', 'write_bytes',
                    'lock_args', 'nrows', 'colour', 'delay', 'gui', 'kwargs'
                }
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in known_kwargs}
                super().__init__(*args, **filtered_kwargs)

            def update(self, n=1):
                super().update(n)
                if tracker._callback and self.total:
                    tracker(self.n, self.total)

        model_path = snapshot_download(
            repo_id,
            tqdm_class=ProgressTqdm,
        )

        logger.info("model_download: complete, path=%s", model_path)
        return model_path

    except ImportError:
        # tqdm not available, download without progress
        logger.warning("model_download: tqdm not available, downloading without progress")
        model_path = snapshot_download(repo_id)
        logger.info("model_download: complete, path=%s", model_path)
        return model_path

    except Exception as e:
        logger.error(
            "model_download: failed for %s, error_type=%s, error=%s",
            model_name,
            type(e).__name__,
            str(e),
        )
        raise
