"""Device auto-detection and resolution for LocalWispr.

This module provides device auto-detection functionality that:
1. Resolves "auto" device to actual device (cuda or cpu)
2. Falls back to CPU when CUDA is unavailable
3. Returns the number of threads to use for CPU inference
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def resolve_device(config_device: str, config_compute: str) -> tuple[str, str]:
    """Resolve configured device to actual device with appropriate compute type.

    Handles three device configurations:
    - "auto": Auto-detect CUDA, fallback to CPU if unavailable
    - "cuda": Use CUDA if available, fallback to CPU with warning if not
    - "cpu": Use CPU directly

    Note: compute_type is kept for config compatibility but whisper.cpp
    handles precision automatically based on model and device.

    Args:
        config_device: "auto", "cuda", or "cpu"
        config_compute: "auto", "float16", "int8", or "float32" (for compatibility)

    Returns:
        Tuple of (actual_device, actual_compute_type)
    """
    from localwispr.transcribe.gpu import check_cuda_available

    # Resolve device
    if config_device == "auto":
        if check_cuda_available():
            actual_device = "cuda"
            logger.info("device: auto-detected CUDA")
        else:
            actual_device = "cpu"
            logger.info("device: CUDA not available, using CPU")
    elif config_device == "cuda":
        if check_cuda_available():
            actual_device = "cuda"
        else:
            actual_device = "cpu"
            logger.warning(
                "device: CUDA requested but not available, falling back to CPU"
            )
    else:
        actual_device = "cpu"

    # Resolve compute_type
    # For faster-whisper (CTranslate2), compute_type is meaningful
    # For pywhispercpp (whisper.cpp), it uses GGML quantization internally
    if config_compute == "auto":
        if actual_device == "cuda":
            actual_compute = "float16"
        else:
            actual_compute = "int8"
        logger.debug(
            "compute_type: auto-resolved to %s for device %s",
            actual_compute,
            actual_device,
        )
    else:
        actual_compute = config_compute

    return actual_device, actual_compute


def get_optimal_threads() -> int:
    """Get the optimal number of threads for CPU inference.

    Returns:
        Number of threads to use (defaults to CPU count or 4).
    """
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 4
    # Use all cores but leave one for system responsiveness
    return max(1, cpu_count - 1)


def get_device_info() -> dict:
    """Get information about the resolved device configuration.

    Returns:
        Dictionary containing:
        - cuda_available: bool
        - resolved_device: str ("cuda" or "cpu")
        - optimal_threads: int (for CPU inference)
    """
    from localwispr.transcribe.gpu import check_cuda_available

    cuda_available = check_cuda_available()

    return {
        "cuda_available": cuda_available,
        "resolved_device": "cuda" if cuda_available else "cpu",
        "optimal_threads": get_optimal_threads(),
    }
