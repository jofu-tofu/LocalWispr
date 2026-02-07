"""Model download and cache management for LocalWispr.

This module provides functionality to:
1. Check if a model is already downloaded (cached)
2. Get model size information for UI display
3. Download models from HuggingFace whisper.cpp repository
4. Support GGML format models for whisper.cpp
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

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
    """Check if a model file exists in the cache.

    Args:
        model_name: Model name (e.g., "large-v3")

    Returns:
        True if the model file exists, False otherwise.
    """
    model_path = get_model_path(model_name)
    exists = model_path.exists()
    logger.debug("model_cache: %s exists=%s path=%s", model_name, exists, model_path)
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
    """Download a model from HuggingFace.

    Args:
        model_name: Model name (e.g., "large-v3")
        progress_callback: Optional callback(downloaded_bytes, total_bytes)

    Returns:
        Path to the downloaded model file.

    Raises:
        ValueError: If model name is not recognized.
        urllib.error.URLError: If download fails.
    """
    if model_name not in GGML_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(GGML_MODELS.keys())}"
        )

    url = get_model_download_url(model_name)
    model_path = get_model_path(model_name)

    # Ensure directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("model_download: starting %s from %s", model_name, url)

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
        logger.info("model_download: completed %s at %s", model_name, model_path)
        return model_path

    except Exception as e:
        # Clean up partial download
        if temp_path.exists():
            temp_path.unlink()
        logger.error("model_download: failed %s: %s", model_name, e)
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

    Args:
        model_name: Model name (e.g., "large-v3")

    Returns:
        Size in bytes, or 0 if model is not downloaded.
    """
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

    Args:
        model_name: Model name (e.g., "large-v3")

    Returns:
        True if model was deleted, False if it wasn't found or deletion failed.
    """
    model_path = get_model_path(model_name)

    if not model_path.exists():
        logger.debug("model_cache: cannot delete %s, not found", model_name)
        return False

    try:
        model_path.unlink()
        logger.info("model_cache: deleted model %s from %s", model_name, model_path)
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
