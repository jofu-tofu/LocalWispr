"""LocalWispr - Local speech-to-text using Whisper."""

import os
import sys
from pathlib import Path

__version__ = "0.1.0"


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


def _configure_cuda_dlls() -> None:
    """Configure CUDA DLL paths for Windows before importing ctranslate2.

    This must be called before any imports that depend on ctranslate2.
    """
    if sys.platform == "win32":
        try:
            import nvidia.cublas  # noqa: F401
            # nvidia.cublas is a namespace package, use __path__ instead of __file__
            cublas_paths = list(nvidia.cublas.__path__)
            if cublas_paths:
                cublas_bin = os.path.join(cublas_paths[0], "bin")
                if os.path.isdir(cublas_bin):
                    os.add_dll_directory(cublas_bin)
                    os.environ["PATH"] = cublas_bin + os.pathsep + os.environ.get("PATH", "")
        except ImportError:
            pass  # nvidia-cublas-cu12 not installed, will use CPU or fail gracefully


BUILD_VARIANT = _detect_build_variant()
IS_TEST_BUILD = BUILD_VARIANT == "test"

# Configure CUDA before importing modules that use ctranslate2
_configure_cuda_dlls()

from localwispr.audio import AudioRecorder, AudioRecorderError, prepare_for_whisper  # noqa: E402
from localwispr.hotkeys import (  # noqa: E402
    HotkeyListener,
    HotkeyListenerError,
    HotkeyMode,
    HotkeyState,
)
from localwispr.transcribe import (  # noqa: E402
    TranscriptionResult,
    WhisperTranscriber,
    transcribe_recording,
)
from localwispr.volume import (  # noqa: E402
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
