"""Audio recording, format conversion, feedback, and volume control."""

from localwispr.audio.recorder import AudioRecorder, AudioRecorderError, list_audio_devices
from localwispr.audio.format import prepare_for_whisper, WHISPER_SAMPLE_RATE, WHISPER_DTYPE
from localwispr.audio.feedback import play_start_beep, play_stop_beep
from localwispr.audio.volume import (
    get_mute_state,
    mute_system,
    unmute_system,
    restore_mute_state,
    system_muted,
)

__all__ = [
    "AudioRecorder",
    "AudioRecorderError",
    "list_audio_devices",
    "prepare_for_whisper",
    "WHISPER_SAMPLE_RATE",
    "WHISPER_DTYPE",
    "play_start_beep",
    "play_stop_beep",
    "get_mute_state",
    "mute_system",
    "unmute_system",
    "restore_mute_state",
    "system_muted",
]
