"""LocalWispr - Local speech-to-text using Whisper."""

__version__ = "0.1.0"

# Configure CUDA DLL paths for Windows before importing ctranslate2
import os
import sys

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
]
