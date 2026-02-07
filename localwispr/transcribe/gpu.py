"""GPU detection and verification utilities for LocalWispr."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def check_cuda_available() -> bool:
    """Check if CUDA is available for inference.

    Returns:
        True if CUDA is available, False otherwise.
    """
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        logger.warning("gpu: torch not available for CUDA detection")
        return False


def get_gpu_info() -> dict:
    """Get information about available GPU(s).

    Returns:
        Dictionary containing GPU information:
        - cuda_available: bool
        - device_count: int
        - devices: list of device info dicts
    """
    cuda_available = check_cuda_available()

    info = {
        "cuda_available": cuda_available,
        "device_count": 0,
        "devices": [],
    }

    if cuda_available:
        try:
            import torch

            info["device_count"] = torch.cuda.device_count()
            for i in range(info["device_count"]):
                device_props = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "index": i,
                    "name": device_props.name,
                    "total_memory_gb": round(
                        device_props.total_memory / (1024**3), 2
                    ),
                    "compute_capability": f"{device_props.major}.{device_props.minor}",
                })
        except Exception as e:
            logger.warning("gpu: error getting GPU details: %s", e)
            info["device_count"] = 1
            info["devices"].append({
                "index": 0,
                "name": "CUDA Device (details unavailable)",
                "total_memory_gb": None,
                "compute_capability": None,
            })

    return info


def verify_whisper_gpu() -> dict:
    """Verify that whisper.cpp can use the GPU for inference.

    Returns:
        Dictionary containing verification results:
        - success: bool
        - cuda_available: bool
        - error: str or None
    """
    result = {
        "success": False,
        "cuda_available": False,
        "error": None,
    }

    try:
        cuda_available = check_cuda_available()
        result["cuda_available"] = cuda_available

        if not cuda_available:
            result["error"] = "CUDA is not available"
            return result

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def print_gpu_status() -> None:
    """Print a formatted GPU status report."""
    print("=" * 60)
    print("LocalWispr GPU Status Report")
    print("=" * 60)

    gpu_info = get_gpu_info()
    whisper_status = verify_whisper_gpu()

    print(f"\nCUDA Available: {gpu_info['cuda_available']}")

    if gpu_info['cuda_available']:
        print(f"Device Count: {gpu_info['device_count']}")

        if gpu_info['devices']:
            print("\nDetected GPU(s):")
            for device in gpu_info['devices']:
                print(f"  [{device['index']}] {device['name']}")
                if device['total_memory_gb']:
                    print(f"      Memory: {device['total_memory_gb']} GB")
                if device['compute_capability']:
                    print(f"      Compute Capability: {device['compute_capability']}")

        print(f"\nGPU Verification: {'PASSED' if whisper_status['success'] else 'FAILED'}")
    else:
        print("\nGPU Verification: FAILED - CUDA not available")
        if whisper_status['error']:
            print(f"Error: {whisper_status['error']}")

    print("=" * 60)


def check_gpu() -> dict:
    """Convenience function for quick GPU status check.

    Returns:
        Dictionary with cuda_available, gpu_name, vram_gb, error keys.
    """
    info = get_gpu_info()
    result = {
        "cuda_available": info["cuda_available"],
        "gpu_name": None,
        "vram_gb": None,
        "error": None,
    }

    if info["cuda_available"] and info["devices"]:
        device = info["devices"][0]
        result["gpu_name"] = device["name"]
        result["vram_gb"] = device["total_memory_gb"]

    if not info["cuda_available"]:
        result["error"] = "CUDA not available"

    return result


if __name__ == "__main__":
    print_gpu_status()
