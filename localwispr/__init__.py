"""LocalWispr - Local speech-to-text using Whisper."""

__version__ = "0.1.0"

from localwispr.audio import AudioRecorder, AudioRecorderError, prepare_for_whisper
from localwispr.transcribe import (
    TranscriptionResult,
    WhisperTranscriber,
    transcribe_recording,
)

__all__ = [
    "AudioRecorder",
    "AudioRecorderError",
    "prepare_for_whisper",
    "TranscriptionResult",
    "WhisperTranscriber",
    "transcribe_recording",
]
