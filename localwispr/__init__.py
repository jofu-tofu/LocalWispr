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


BUILD_VARIANT = _detect_build_variant()
IS_TEST_BUILD = BUILD_VARIANT == "test"

from localwispr.audio import AudioRecorder, AudioRecorderError, prepare_for_whisper  # noqa: E402
from localwispr.hotkeys import (  # noqa: E402
    HotkeyListener,
    HotkeyListenerError,
    HotkeyMode,
    HotkeyState,
)
from localwispr.transcribe.transcriber import (  # noqa: E402
    TranscriptionResult,
    WhisperTranscriber,
    transcribe_recording,
)
from localwispr.audio.volume import (  # noqa: E402
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
