"""Entry point for LocalWispr."""

import argparse
import platform
import sys

from localwispr import __version__
from localwispr.config import get_config
from localwispr.tray import TrayApp


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
            print("  CUDA Available: Yes")
            print(f"  GPU Name:       {gpu_info.get('gpu_name', 'Unknown')}")
            print(f"  VRAM:           {gpu_info.get('vram_gb', 'Unknown')} GB")
        else:
            print("  CUDA Available: No")
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


def run_default() -> None:
    """Run the default LocalWispr startup display."""
    print_banner()

    # Load and display configuration
    config = get_config()
    print_config_info(config)

    # Try to display GPU info (may not be available yet)
    print_gpu_info()

    # Success message
    print("=" * 40)
    print("LocalWispr initialized successfully!")
    print("=" * 40)


def run_tray() -> int:
    """Run LocalWispr as a system tray application.

    This is the default mode when running `localwispr` without arguments.
    The application runs in the system tray with hotkey-driven recording.

    Returns:
        Exit code: 0 for success, 1 for error.
    """
    print("Starting LocalWispr in system tray mode...")
    print("Look for the LocalWispr icon in your system tray.")
    print()

    try:
        # Create and run the tray application
        app = TrayApp()

        # The tray app will handle hotkey integration in Task 3
        # For now, it just provides the tray icon and menu
        app.run()
        return 0

    except Exception as e:
        print(f"Error starting tray application: {e}", file=sys.stderr)
        return 1


def main() -> None:
    """Main entry point for LocalWispr."""
    parser = argparse.ArgumentParser(
        prog="localwispr",
        description="LocalWispr - Local speech-to-text with Whisper",
    )
    subparsers = parser.add_subparsers(dest="command")

    # tray subcommand (also the default when no args)
    subparsers.add_parser(
        "tray",
        help="Run as system tray application (default)",
    )

    # info subcommand (shows system info)
    subparsers.add_parser(
        "info",
        help="Show system information and configuration",
    )

    # record-test subcommand
    record_test_parser = subparsers.add_parser(
        "record-test",
        help="Test audio recording functionality",
    )
    record_test_parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Save recorded audio to WAV file",
    )

    # transcribe-test subcommand
    transcribe_test_parser = subparsers.add_parser(
        "transcribe-test",
        help="Test transcription functionality",
    )
    transcribe_test_parser.add_argument(
        "-m", "--model",
        metavar="MODEL",
        help="Model name to use (overrides config)",
    )

    # hotkey-test subcommand
    subparsers.add_parser(
        "hotkey-test",
        help="Test hotkey-triggered recording and transcription",
    )

    # context-test subcommand
    context_test_parser = subparsers.add_parser(
        "context-test",
        help="Test context detection functionality",
    )
    context_test_parser.add_argument(
        "-t", "--text",
        metavar="TEXT",
        help="Text to test keyword-based context detection",
    )

    args = parser.parse_args()

    if args.command == "tray" or args.command is None:
        # Default behavior: run as system tray application
        sys.exit(run_tray())
    elif args.command == "info":
        run_default()
    elif args.command == "record-test":
        from localwispr.cli_tests import record_test

        sys.exit(record_test(args.output))
    elif args.command == "transcribe-test":
        from localwispr.cli_tests import transcribe_test

        sys.exit(transcribe_test(args.model))
    elif args.command == "hotkey-test":
        from localwispr.cli_tests import hotkey_test

        sys.exit(hotkey_test())
    elif args.command == "context-test":
        from localwispr.cli_tests import context_test

        sys.exit(context_test(args.text))


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
