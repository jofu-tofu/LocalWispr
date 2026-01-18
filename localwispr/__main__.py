"""Entry point for LocalWispr."""

import platform

from localwispr import __version__
from localwispr.config import load_config


def print_banner() -> None:
    """Print the startup banner."""
    banner = r"""
    __                    ___       ___
   / /   ____  _________ / / |     / (_)_________  _____
  / /   / __ \/ ___/ __ `/ /| | /| / / / ___/ __ \/ ___/
 / /___/ /_/ / /__/ /_/ / / | |/ |/ / (__  ) /_/ / /
/_____/\____/\___/\__,_/_/  |__/|__/_/____/ .___/_/
                                         /_/
"""
    print(banner)
    print(f"  Version: {__version__}")
    print(f"  Python:  {platform.python_version()}")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print()


def print_config_info(config: dict) -> None:
    """Print configuration information."""
    print("Configuration")
    print("-" * 40)
    print(f"  Model:        {config['model']['name']}")
    print(f"  Device:       {config['model']['device']}")
    print(f"  Compute Type: {config['model']['compute_type']}")
    print()


def print_gpu_info() -> None:
    """Try to print GPU information if gpu module is available."""
    try:
        from localwispr.gpu import check_gpu

        print("GPU Information")
        print("-" * 40)
        gpu_info = check_gpu()
        if gpu_info.get("cuda_available"):
            print(f"  CUDA Available: Yes")
            print(f"  GPU Name:       {gpu_info.get('gpu_name', 'Unknown')}")
            print(f"  VRAM:           {gpu_info.get('vram_gb', 'Unknown')} GB")
        else:
            print(f"  CUDA Available: No")
            if gpu_info.get("error"):
                print(f"  Note: {gpu_info['error']}")
        print()
    except ImportError:
        # gpu.py not yet created (Task 2 running in parallel)
        print("GPU Information")
        print("-" * 40)
        print("  (GPU module not yet available)")
        print()
    except Exception as e:
        print("GPU Information")
        print("-" * 40)
        print(f"  Error checking GPU: {e}")
        print()


def main() -> None:
    """Main entry point for LocalWispr."""
    print_banner()

    # Load and display configuration
    config = load_config()
    print_config_info(config)

    # Try to display GPU info (may not be available yet)
    print_gpu_info()

    # Success message
    print("=" * 40)
    print("LocalWispr initialized successfully!")
    print("=" * 40)


if __name__ == "__main__":
    main()
