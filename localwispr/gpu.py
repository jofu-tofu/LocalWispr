"""GPU detection and verification utilities for LocalWispr."""

from __future__ import annotations

import ctranslate2


def check_cuda_available() -> bool:
    """Check if CUDA is available for inference.

    Returns:
        True if CUDA is available, False otherwise.
    """
    supported_devices = ctranslate2.get_supported_compute_types("cuda")
    return len(supported_devices) > 0


def get_gpu_info() -> dict:
    """Get information about available GPU(s).

    Returns:
        Dictionary containing GPU information:
        - cuda_available: bool
        - device_count: int
        - devices: list of device info dicts
        - supported_compute_types: list of supported compute types for CUDA
    """
    cuda_available = check_cuda_available()

    info = {
        "cuda_available": cuda_available,
        "device_count": 0,
        "devices": [],
        "supported_compute_types": [],
    }

    if cuda_available:
        info["supported_compute_types"] = list(
            ctranslate2.get_supported_compute_types("cuda")
        )

        # Try to get device info via torch if available
        try:
            import torch
            if torch.cuda.is_available():
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
        except ImportError:
            # torch not available, use ctranslate2 detection only
            info["device_count"] = 1  # ctranslate2 detected at least one
            info["devices"].append({
                "index": 0,
                "name": "CUDA Device (install torch for detailed info)",
                "total_memory_gb": None,
                "compute_capability": None,
            })

    return info


def verify_whisper_gpu() -> dict:
    """Verify that faster-whisper can use the GPU for inference.

    Returns:
        Dictionary containing verification results:
        - success: bool
        - cuda_available: bool
        - ctranslate2_version: str
        - recommended_compute_type: str or None
        - error: str or None
    """
    result = {
        "success": False,
        "cuda_available": False,
        "ctranslate2_version": ctranslate2.__version__,
        "recommended_compute_type": None,
        "error": None,
    }

    try:
        cuda_available = check_cuda_available()
        result["cuda_available"] = cuda_available

        if not cuda_available:
            result["error"] = "CUDA is not available"
            return result

        # Get supported compute types
        compute_types = ctranslate2.get_supported_compute_types("cuda")

        # Recommend the best compute type for RTX 4090 (compute 8.9)
        # Priority: float16 > int8_float16 > int8 > float32
        priority = ["float16", "int8_float16", "int8", "float32"]
        for ct in priority:
            if ct in compute_types:
                result["recommended_compute_type"] = ct
                break

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
    print(f"ctranslate2 Version: {whisper_status['ctranslate2_version']}")

    if gpu_info['cuda_available']:
        print(f"Device Count: {gpu_info['device_count']}")
        print(f"Supported Compute Types: {', '.join(gpu_info['supported_compute_types'])}")

        if gpu_info['devices']:
            print("\nDetected GPU(s):")
            for device in gpu_info['devices']:
                print(f"  [{device['index']}] {device['name']}")
                if device['total_memory_gb']:
                    print(f"      Memory: {device['total_memory_gb']} GB")
                if device['compute_capability']:
                    print(f"      Compute Capability: {device['compute_capability']}")

        print(f"\nRecommended Compute Type: {whisper_status['recommended_compute_type']}")
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
