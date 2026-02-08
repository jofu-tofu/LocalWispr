"""Model download and cache management for LocalWispr.

This module provides functionality to:
1. Check if a model is already downloaded (cached)
2. Get model size information for UI display
3. Download models from HuggingFace
4. Support both faster-whisper (CTranslate2) and pywhispercpp (GGML) model formats
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def get_active_backend() -> str:
    """Detect which transcription backend is active.

    Returns:
        "faster-whisper" if available, otherwise "pywhispercpp".
    """
    try:
        from faster_whisper import WhisperModel  # noqa: F401

        logger.debug("get_active_backend: faster-whisper available")
        return "faster-whisper"
    except ImportError:
        logger.debug("get_active_backend: falling back to pywhispercpp")
        return "pywhispercpp"

# GGML model names to HuggingFace file names
GGML_MODELS = {
    "tiny": "ggml-tiny.bin",
    "tiny.en": "ggml-tiny.en.bin",
    "base": "ggml-base.bin",
    "base.en": "ggml-base.en.bin",
    "small": "ggml-small.bin",
    "small.en": "ggml-small.en.bin",
    "medium": "ggml-medium.bin",
    "medium.en": "ggml-medium.en.bin",
    "large-v1": "ggml-large-v1.bin",
    "large-v2": "ggml-large-v2.bin",
    "large-v3": "ggml-large-v3.bin",
    "large-v3-turbo": "ggml-large-v3-turbo.bin",
}

# Approximate model sizes in megabytes (for UI display)
MODEL_SIZES_MB = {
    "tiny": 75,
    "tiny.en": 75,
    "base": 145,
    "base.en": 145,
    "small": 465,
    "small.en": 465,
    "medium": 1500,
    "medium.en": 1500,
    "large-v1": 3000,
    "large-v2": 3000,
    "large-v3": 3000,
    "large-v3-turbo": 1600,
}

# Base URL for GGML models on HuggingFace
HUGGINGFACE_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"


def get_models_dir() -> Path:
    """Get the LocalWispr models directory.

    Uses pywhispercpp's default model storage location when available,
    otherwise falls back to LocalWispr-specific directory.

    Returns:
        Path to the models directory.
    """
    # Check if pywhispercpp has a models directory we can use
    try:
        from pywhispercpp.constants import MODELS_DIR as PYWHISPERCPP_MODELS_DIR

        return Path(PYWHISPERCPP_MODELS_DIR)
    except ImportError:
        pass

    # Fall back to platform-specific user data directory
    if os.name == "nt":  # Windows
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:  # Linux/macOS
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))

    models_dir = base / "LocalWispr" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_model_filename(model_name: str) -> str:
    """Get the GGML filename for a model name.

    Args:
        model_name: Model name (e.g., "large-v3", "small")

    Returns:
        Filename (e.g., "ggml-large-v3.bin")
    """
    return GGML_MODELS.get(model_name, f"ggml-{model_name}.bin")


def get_model_path(model_name: str) -> Path:
    """Get the full path to a model file.

    Args:
        model_name: Model name (e.g., "large-v3", "small")

    Returns:
        Full path to the model file.
    """
    return get_models_dir() / get_model_filename(model_name)


def is_model_downloaded(model_name: str) -> bool:
    """Check if a model is available for the active backend.

    For faster-whisper: checks HuggingFace Hub cache for CTranslate2 format.
    For pywhispercpp: checks local GGML model file.

    Args:
        model_name: Model name (e.g., "large-v3")

    Returns:
        True if the model is available, False otherwise.
    """
    backend = get_active_backend()
    logger.debug("is_model_downloaded: model=%s backend=%s", model_name, backend)

    if backend == "faster-whisper":
        exists = _is_model_downloaded_faster_whisper(model_name)
        if exists:
            logger.debug("model_cache: %s found in faster-whisper cache", model_name)
            return True
        # Also check GGML as fallback info
        ggml_exists = _is_model_downloaded_ggml(model_name)
        if ggml_exists:
            logger.debug(
                "model_cache: %s found as GGML but not in faster-whisper cache "
                "(will auto-download on first use)",
                model_name,
            )
            return True
        return False

    # pywhispercpp backend
    return _is_model_downloaded_ggml(model_name)


def _is_model_downloaded_faster_whisper(model_name: str) -> bool:
    """Check if a faster-whisper (CTranslate2) model is cached.

    faster-whisper auto-downloads models from HuggingFace Hub on first use.
    This checks if the model is already in the HF cache.

    Args:
        model_name: Model name (e.g., "large-v3")

    Returns:
        True if the model exists in HuggingFace cache.
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        repo_id = f"Systran/faster-whisper-{model_name}"
        # Check for the model.bin file which is always present
        result = try_to_load_from_cache(repo_id, "model.bin")
        return result is not None and isinstance(result, str)
    except ImportError:
        logger.debug("model_cache: huggingface_hub not available for cache check")
        return False
    except Exception as e:
        logger.debug("model_cache: faster-whisper cache check failed: %s", e)
        return False


def _is_model_downloaded_ggml(model_name: str) -> bool:
    """Check if a GGML model file exists in the cache.

    Args:
        model_name: Model name (e.g., "large-v3")

    Returns:
        True if the GGML model file exists.
    """
    model_path = get_model_path(model_name)
    exists = model_path.exists()
    logger.debug("model_cache: %s ggml exists=%s path=%s", model_name, exists, model_path)
    return exists


def get_model_download_url(model_name: str) -> str:
    """Get the download URL for a model.

    Args:
        model_name: Model name (e.g., "large-v3")

    Returns:
        URL to download the model from.
    """
    filename = get_model_filename(model_name)
    return f"{HUGGINGFACE_BASE_URL}/{filename}"


def download_model(
    model_name: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Download a model using the active backend's format.

    For faster-whisper: downloads CTranslate2 format from HuggingFace Hub.
    For pywhispercpp: downloads GGML format from HuggingFace.

    Args:
        model_name: Model name (e.g., "large-v3")
        progress_callback: Optional callback(downloaded_bytes, total_bytes)

    Returns:
        Path to the downloaded model.

    Raises:
        ValueError: If model name is not recognized.
    """
    if model_name not in GGML_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(GGML_MODELS.keys())}"
        )

    backend = get_active_backend()

    if backend == "faster-whisper":
        return _download_model_faster_whisper(model_name, progress_callback)
    else:
        return _download_model_ggml(model_name, progress_callback)


def _download_model_faster_whisper(
    model_name: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Download a model in CTranslate2 format for faster-whisper.

    faster-whisper auto-downloads on first model load, but this provides
    explicit download with progress tracking for the UI.

    Args:
        model_name: Model name (e.g., "large-v3")
        progress_callback: Optional callback(downloaded_bytes, total_bytes)

    Returns:
        Path to the cached model directory.
    """
    from huggingface_hub import snapshot_download

    repo_id = f"Systran/faster-whisper-{model_name}"
    logger.info("model_download: starting faster-whisper %s from %s", model_name, repo_id)

    # snapshot_download returns the local path to the cached model
    local_dir = snapshot_download(repo_id)

    # Signal completion to progress callback
    if progress_callback:
        progress_callback(1, 1)

    logger.info("model_download: completed faster-whisper %s at %s", model_name, local_dir)
    return Path(local_dir)


def _download_model_ggml(
    model_name: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Download a model in GGML format for pywhispercpp.

    Args:
        model_name: Model name (e.g., "large-v3")
        progress_callback: Optional callback(downloaded_bytes, total_bytes)

    Returns:
        Path to the downloaded model file.
    """
    url = get_model_download_url(model_name)
    model_path = get_model_path(model_name)

    # Ensure directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("model_download: starting GGML %s from %s", model_name, url)

    # Download with progress tracking
    temp_path = model_path.with_suffix(".tmp")

    try:
        request = urllib.request.Request(url)
        request.add_header("User-Agent", "LocalWispr/0.1.0")

        with urllib.request.urlopen(request) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 8192

            with open(temp_path, "wb") as f:
                while True:
                    block = response.read(block_size)
                    if not block:
                        break
                    f.write(block)
                    downloaded += len(block)

                    if progress_callback:
                        progress_callback(downloaded, total_size)

        # Rename temp file to final location
        temp_path.rename(model_path)
        logger.info("model_download: completed GGML %s at %s", model_name, model_path)
        return model_path

    except Exception as e:
        # Clean up partial download
        if temp_path.exists():
            temp_path.unlink()
        logger.error("model_download: failed GGML %s: %s", model_name, e)
        raise


def get_model_status(model_name: str) -> dict:
    """Get comprehensive status information for a model.

    Args:
        model_name: Model name (e.g., "large-v3")

    Returns:
        Dictionary containing:
        - downloaded: bool - whether model files exist
        - size_mb: int - approximate download size in MB
        - path: str - full path to model file
        - url: str - download URL
    """
    return {
        "downloaded": is_model_downloaded(model_name),
        "size_mb": MODEL_SIZES_MB.get(model_name, 0),
        "path": str(get_model_path(model_name)),
        "url": get_model_download_url(model_name),
    }


def get_recommended_model_for_cpu() -> str:
    """Get the recommended model name for CPU inference.

    Returns:
        Recommended model name for CPU users ("small").
    """
    return "small"


def get_model_disk_size(model_name: str) -> int:
    """Get the actual disk size of a downloaded model in bytes.

    For faster-whisper: returns size of the cached model directory.
    For pywhispercpp: returns size of the GGML model file.

    Args:
        model_name: Model name (e.g., "large-v3")

    Returns:
        Size in bytes, or 0 if model is not downloaded.
    """
    backend = get_active_backend()

    if backend == "faster-whisper":
        try:
            from huggingface_hub import try_to_load_from_cache

            repo_id = f"Systran/faster-whisper-{model_name}"
            result = try_to_load_from_cache(repo_id, "model.bin")
            if result and isinstance(result, str):
                model_bin = Path(result)
                if model_bin.exists():
                    return model_bin.stat().st_size
        except Exception:
            pass
        return 0

    # pywhispercpp: check GGML file
    model_path = get_model_path(model_name)

    if not model_path.exists():
        return 0

    try:
        return model_path.stat().st_size
    except OSError as e:
        logger.warning("model_cache: error getting size for %s: %s", model_name, e)
        return 0


def delete_model(model_name: str) -> bool:
    """Delete a model from the cache.

    Deletes from the active backend's cache. For faster-whisper, uses
    HuggingFace Hub cache deletion. For pywhispercpp, deletes the GGML file.

    Args:
        model_name: Model name (e.g., "large-v3")

    Returns:
        True if model was deleted, False if it wasn't found or deletion failed.
    """
    backend = get_active_backend()
    deleted = False

    if backend == "faster-whisper":
        try:
            from huggingface_hub import scan_cache_dir

            repo_id = f"Systran/faster-whisper-{model_name}"
            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                if repo.repo_id == repo_id:
                    for revision in repo.revisions:
                        cache_info.delete_revisions(revision.commit_hash).execute()
                        deleted = True
                    break
            if deleted:
                logger.info("model_cache: deleted faster-whisper model %s", model_name)
                return True
        except Exception as e:
            logger.warning("model_cache: failed to delete faster-whisper %s: %s", model_name, e)

    # pywhispercpp / fallback: delete GGML file
    model_path = get_model_path(model_name)

    if not model_path.exists():
        if not deleted:
            logger.debug("model_cache: cannot delete %s, not found", model_name)
            return False
        return True

    try:
        model_path.unlink()
        logger.info("model_cache: deleted GGML model %s from %s", model_name, model_path)
        return True
    except OSError as e:
        logger.error("model_cache: failed to delete %s: %s", model_name, e)
        return False


def get_all_models_status() -> list[dict]:
    """Get status information for all available models.

    Returns:
        List of dictionaries, each containing:
        - name: str - model name
        - downloaded: bool - whether model files exist
        - size_mb: int - approximate download size in MB
        - disk_size: int - actual disk size in bytes (0 if not downloaded)
    """
    models = []
    for model_name in GGML_MODELS.keys():
        downloaded = is_model_downloaded(model_name)
        models.append({
            "name": model_name,
            "downloaded": downloaded,
            "size_mb": MODEL_SIZES_MB.get(model_name, 0),
            "disk_size": get_model_disk_size(model_name) if downloaded else 0,
        })
    return models


def get_available_model_names() -> list[str]:
    """Get list of all available model names.

    Returns:
        List of model names (e.g., ["tiny", "base", "small", ...])
    """
    return list(GGML_MODELS.keys())
