"""LocalWispr - Local speech-to-text using Whisper."""

__version__ = "0.1.0"

# Configure CUDA DLL paths for Windows before importing ctranslate2
import os
import sys
from pathlib import Path


def _detect_build_variant() -> str:
    """Detect if running as test or stable build.

    Detection logic:
    1. If frozen (PyInstaller), check exe path for "Test" in folder or filename
    2. Otherwise, check BUILD_VARIANT environment variable
    3. Default to "stable"

    Returns:
        "test" or "stable"
    """
    if getattr(sys, "frozen", False):
        exe_path = Path(sys.executable)
        if "Test" in exe_path.parent.name or "Test" in exe_path.stem:
            return "test"
    return os.environ.get("BUILD_VARIANT", "stable")


BUILD_VARIANT = _detect_build_variant()
IS_TEST_BUILD = BUILD_VARIANT == "test"


if sys.platform == "win32":
    # Find nvidia cublas DLL directory in the virtual environment
    try:
        import nvidia.cublas
        # nvidia.cublas is a namespace package, use __path__ instead of __file__
        cublas_paths = list(nvidia.cublas.__path__)
        if cublas_paths:
            cublas_bin = os.path.join(cublas_paths[0], "bin")
            if os.path.isdir(cublas_bin):
                os.add_dll_directory(cublas_bin)
                os.environ["PATH"] = cublas_bin + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass  # nvidia-cublas-cu12 not installed, will use CPU or fail gracefully

from localwispr.audio import AudioRecorder, AudioRecorderError, prepare_for_whisper
from localwispr.hotkeys import (
    HotkeyListener,
    HotkeyListenerError,
    HotkeyMode,
    HotkeyState,
)
from localwispr.transcribe import (
    TranscriptionResult,
    WhisperTranscriber,
    transcribe_recording,
)
from localwispr.volume import (
    get_mute_state,
    mute_system,
    restore_mute_state,
    system_muted,
    unmute_system,
)

__all__ = [
    "AudioRecorder",
    "AudioRecorderError",
    "prepare_for_whisper",
    "HotkeyListener",
    "HotkeyListenerError",
    "HotkeyMode",
    "HotkeyState",
    "TranscriptionResult",
    "WhisperTranscriber",
    "transcribe_recording",
    "get_mute_state",
    "mute_system",
    "restore_mute_state",
    "system_muted",
    "unmute_system",
    "BUILD_VARIANT",
    "IS_TEST_BUILD",
]
