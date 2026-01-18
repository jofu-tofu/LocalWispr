"""Entry point for LocalWispr."""

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from localwispr import __version__
from localwispr.audio import AudioRecorder, AudioRecorderError
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


def record_test(output_file: str | None = None) -> int:
    """Test audio recording functionality.

    Records audio until the user presses Enter, then displays
    information about the captured audio.

    Args:
        output_file: Optional path to save the recorded audio as WAV.

    Returns:
        Exit code: 0 for success, 1 for error.
    """
    print("Audio Recording Test")
    print("=" * 40)

    try:
        recorder = AudioRecorder()
        print(f"Device: {recorder.device_name}")
        print(f"Sample Rate: {recorder.sample_rate} Hz")
        print()
        print("Recording... Press Enter to stop")
        print()

        # Start recording
        recorder.start_recording()

        # Wait for Enter key
        try:
            input()
        except EOFError:
            # Handle non-interactive mode
            pass

        # Stop recording and get audio data
        audio_data = recorder.stop_recording()

        # Calculate statistics
        duration = len(audio_data) / recorder.sample_rate
        peak_level = float(np.max(np.abs(audio_data))) if len(audio_data) > 0 else 0.0

        # Display results
        print("Recording Complete")
        print("-" * 40)
        print(f"  Duration:    {duration:.2f} seconds")
        print(f"  Sample Rate: {recorder.sample_rate} Hz")
        print(f"  Audio Shape: {audio_data.shape}")
        print(f"  Peak Level:  {peak_level:.4f} ({peak_level * 100:.1f}%)")

        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            # Convert float32 [-1, 1] to int16 for WAV
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wavfile.write(output_path, recorder.sample_rate, audio_int16)
            print()
            print(f"Audio saved to: {output_path.absolute()}")

        return 0

    except AudioRecorderError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def run_default() -> None:
    """Run the default LocalWispr startup display."""
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


def main() -> None:
    """Main entry point for LocalWispr."""
    parser = argparse.ArgumentParser(
        prog="localwispr",
        description="LocalWispr - Local speech-to-text with Whisper",
    )
    subparsers = parser.add_subparsers(dest="command")

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

    args = parser.parse_args()

    if args.command == "record-test":
        sys.exit(record_test(args.output))
    else:
        # Default behavior: show startup info
        run_default()


if __name__ == "__main__":
    main()
