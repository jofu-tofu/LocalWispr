"""Entry point for LocalWispr."""

import argparse
import platform
import signal
import sys
import threading
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from localwispr import __version__
from localwispr.audio import AudioRecorder, AudioRecorderError
from localwispr.config import load_config
from localwispr.context import ContextDetector
from localwispr.feedback import play_start_beep, play_stop_beep
from localwispr.hotkeys import HotkeyListener, HotkeyListenerError, HotkeyMode
from localwispr.prompts import load_prompt
from localwispr.transcribe import WhisperTranscriber, transcribe_with_context
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
            print("Latency Target:  ✅ PASS (<2s)")
        else:
            print(f"Latency Target:  ⚠️  {result.inference_time:.2f}s (target: <2s)")

        return 0

    except AudioRecorderError as e:
        print(f"Audio Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def context_test(text: str | None = None) -> int:
    """Test context detection functionality.

    Shows current focused window context and optionally tests keyword detection.

    Args:
        text: Optional text to test keyword detection.

    Returns:
        Exit code: 0 for success, 1 for error.
    """
    print("Context Detection Test")
    print("=" * 40)

    try:
        detector = ContextDetector()

        # Test window detection
        print("Window Detection")
        print("-" * 40)
        window_context = detector.detect_from_window()
        print(f"Detected Context: {window_context.value.upper()}")

        # Get window title for debug (without logging it)
        try:
            import pygetwindow as gw
            active_window = gw.getActiveWindow()
            if active_window:
                print("Active Window:    (detected)")  # Don't log title for privacy
            else:
                print("Active Window:    (none detected)")
        except Exception:
            print("Active Window:    (detection unavailable)")

        print()

        # Test text detection if provided
        if text:
            print("Text Detection")
            print("-" * 40)
            print(f"Input Text: {text}")
            text_context = detector.detect_from_text(text)
            print(f"Detected Context: {text_context.value.upper()}")
            print()

            # Show what prompt would be loaded
            print("Prompt Preview")
            print("-" * 40)
            prompt = load_prompt(text_context.value)
            # Show first 100 chars of prompt
            preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
            print(f"Prompt ({text_context.value}): {preview}")

        else:
            print("Text Detection")
            print("-" * 40)
            print("Use --text 'your text here' to test keyword detection")

        print()
        print("Available Contexts: CODING, PLANNING, GENERAL")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def hotkey_test() -> int:
    """Test hotkey functionality with full recording/transcription pipeline.

    Listens for the configured hotkey chord and performs:
    - Recording when hotkey is held/toggled
    - Context-aware transcription with auto-detection
    - Audio feedback beeps (if enabled)

    Returns:
        Exit code: 0 for success, 1 for error.
    """
    print("Hotkey Test")
    print("=" * 40)

    # Load configuration
    config = load_config()
    hotkey_config = config["hotkeys"]

    # Display current hotkey configuration
    mode_str = hotkey_config["mode"]
    modifiers = hotkey_config["modifiers"]
    audio_feedback = hotkey_config["audio_feedback"]

    # Build chord display string
    chord = "+".join(mod.capitalize() for mod in modifiers)

    print(f"Mode:           {mode_str}")
    print(f"Hotkey:         {chord}")
    print(f"Audio Feedback: {'Enabled' if audio_feedback else 'Disabled'}")
    print("Context Aware:  Enabled (auto-detection)")
    print()

    # Privacy notice
    print("Privacy Notice")
    print("-" * 40)
    print("Only modifier keys (Win/Ctrl/Shift) are tracked.")
    print("No alphanumeric or other keystrokes are logged.")
    print("Window titles are used for context detection but not logged.")
    print()

    # Determine hotkey mode
    if mode_str == "toggle":
        mode = HotkeyMode.TOGGLE
        mode_instruction = f"Press {chord} to start recording, press again to stop"
    else:
        mode = HotkeyMode.PUSH_TO_TALK
        mode_instruction = f"Hold {chord} to record, release to transcribe"

    # Initialize components
    try:
        recorder = AudioRecorder()
        print(f"Audio Device: {recorder.device_name}")
    except AudioRecorderError as e:
        print(f"Error: Failed to initialize audio recorder: {e}", file=sys.stderr)
        return 1

    # Initialize context detector
    detector = ContextDetector()

    # Lazy-loaded transcriber (created on first use)
    transcriber: WhisperTranscriber | None = None

    # Event to signal shutdown
    shutdown_event = threading.Event()

    # Lock for thread-safe access to recorder
    state_lock = threading.Lock()

    def on_record_start() -> None:
        """Callback when recording should start."""
        nonlocal recorder

        try:
            with state_lock:
                if recorder.is_recording:
                    return
                recorder.start_recording()

            print("Recording...")
            if audio_feedback:
                play_start_beep()

        except AudioRecorderError as e:
            print(f"Error starting recording: {e}", file=sys.stderr)
            # Reset state on error - need to signal transcription complete
            listener.on_transcription_complete()

    def on_record_stop() -> None:
        """Callback when recording should stop and transcription should begin."""
        nonlocal transcriber, recorder

        print("Transcribing...")

        try:
            # Get audio from recorder (stops recording)
            with state_lock:
                if not recorder.is_recording:
                    listener.on_transcription_complete()
                    return
                audio = recorder.get_whisper_audio()

            # Check if we have audio
            audio_duration = len(audio) / 16000.0
            if audio_duration < 0.1:
                print("No audio captured.")
                listener.on_transcription_complete()
                if audio_feedback:
                    play_stop_beep()
                return

            # Lazy load transcriber
            if transcriber is None:
                print("Loading model (first use)...")
                transcriber = WhisperTranscriber()
                # Force model load
                _ = transcriber.model
                print("Model loaded!")

            # Transcribe with context detection
            result = transcribe_with_context(audio, transcriber, detector)

            # Display result with context info
            print()
            print(f"Result: {result.text}")
            context_name = result.detected_context.value.upper() if result.detected_context else "UNKNOWN"
            retrans_indicator = " (retranscribed)" if result.was_retranscribed else ""
            print(f"  Context: {context_name}{retrans_indicator}")
            print(f"  (Audio: {result.audio_duration:.1f}s, Inference: {result.inference_time:.2f}s)")
            print()

        except Exception as e:
            print(f"Error during transcription: {e}", file=sys.stderr)

        finally:
            # Always signal completion to reset state machine
            listener.on_transcription_complete()
            if audio_feedback:
                play_stop_beep()

    # Create hotkey listener
    listener = HotkeyListener(
        on_record_start=on_record_start,
        on_record_stop=on_record_stop,
        mode=mode,
    )

    # Set up clean Ctrl+C handling
    def signal_handler(signum, frame):
        """Handle Ctrl+C for clean shutdown."""
        print("\nShutting down...")
        shutdown_event.set()

    # Store original handler
    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Start the listener
        listener.start()
        print()
        print(f"Listening for {chord}... Press Ctrl+C to exit")
        print(f"({mode_instruction})")
        print()

        # Wait for shutdown signal
        while not shutdown_event.is_set():
            shutdown_event.wait(timeout=0.5)

        return 0

    except HotkeyListenerError as e:
        print(f"Error: {e}", file=sys.stderr)
        print()
        print("Troubleshooting:", file=sys.stderr)
        print("  1. Check if antivirus/security software is blocking keyboard hooks", file=sys.stderr)
        print("  2. Verify no Group Policy restrictions are in place", file=sys.stderr)
        print("  3. Try running as administrator", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1

    finally:
        # Cleanup
        listener.stop()
        signal.signal(signal.SIGINT, original_handler)


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
        sys.exit(record_test(args.output))
    elif args.command == "transcribe-test":
        sys.exit(transcribe_test(args.model))
    elif args.command == "hotkey-test":
        sys.exit(hotkey_test())
    elif args.command == "context-test":
        sys.exit(context_test(args.text))


if __name__ == "__main__":
    main()
