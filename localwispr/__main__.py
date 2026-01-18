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
from localwispr.transcribe import WhisperTranscriber


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


def transcribe_test(model_name: str | None = None) -> int:
    """Test transcription functionality.

    Records audio until the user presses Enter, then transcribes it
    using Whisper and displays the results with timing information.

    Args:
        model_name: Optional model name to override config default.

    Returns:
        Exit code: 0 for success, 1 for error.
    """
    print("Transcription Test")
    print("=" * 40)

    # Load config and show model info
    config = load_config()
    actual_model = model_name or config["model"]["name"]
    print(f"Model:        {actual_model}")
    print(f"Device:       {config['model']['device']}")
    print(f"Compute Type: {config['model']['compute_type']}")
    print()

    try:
        # Initialize transcriber (lazy load)
        print("Initializing transcriber...")
        transcriber = WhisperTranscriber(model_name=model_name)

        # Initialize recorder
        recorder = AudioRecorder()
        print(f"Audio Device: {recorder.device_name}")
        print(f"Sample Rate:  {recorder.sample_rate} Hz")
        print()

        # Start recording
        print("Recording... Press Enter to stop and transcribe")
        print()
        recorder.start_recording()

        # Wait for Enter key
        try:
            input()
        except EOFError:
            pass

        # Get audio (stops recording)
        print("Processing...")
        audio = recorder.get_whisper_audio()
        audio_duration = len(audio) / 16000.0

        if audio_duration < 0.1:
            print("No audio captured. Make sure your microphone is working.")
            return 1

        print(f"Audio Duration: {audio_duration:.2f} seconds")
        print()

        # Load model if not loaded (shows progress)
        if not transcriber.is_loaded:
            print(f"Loading model '{transcriber.model_name}'...")
            print("(This may take a moment on first run as the model downloads)")
            # Force model load
            _ = transcriber.model
            print("Model loaded!")
            print()

        # Transcribe
        print("Transcribing...")
        result = transcriber.transcribe(audio)

        # Display results
        print()
        print("Results")
        print("-" * 40)
        print(f"Text: {result.text}")
        print()
        print(f"Audio Duration:  {result.audio_duration:.2f}s")
        print(f"Inference Time:  {result.inference_time:.2f}s")
        print(f"Real-time Factor: {result.inference_time / result.audio_duration:.2f}x")

        # Check latency target
        if result.inference_time < 2.0:
            print(f"Latency Target:  ✅ PASS (<2s)")
        else:
            print(f"Latency Target:  ⚠️  {result.inference_time:.2f}s (target: <2s)")

        return 0

    except AudioRecorderError as e:
        print(f"Audio Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
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

    args = parser.parse_args()

    if args.command == "record-test":
        sys.exit(record_test(args.output))
    elif args.command == "transcribe-test":
        sys.exit(transcribe_test(args.model))
    else:
        # Default behavior: show startup info
        run_default()


if __name__ == "__main__":
    main()
