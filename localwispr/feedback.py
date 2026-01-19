"""Audio feedback module for LocalWispr.

Provides non-blocking audio feedback for recording state changes.
Uses soft sine wave tones played via sounddevice for a pleasant, non-jarring sound.
"""

from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd

# Default volume (0.0 to 1.0) - kept low for subtle feedback
_DEFAULT_VOLUME = 0.15


def _generate_tone(
    frequency: float,
    duration_ms: int,
    sample_rate: int = 44100,
    volume: float = _DEFAULT_VOLUME,
    fade_ms: int = 15,
) -> np.ndarray:
    """Generate a soft sine wave tone with fade in/out.

    Args:
        frequency: Frequency in Hz.
        duration_ms: Duration in milliseconds.
        sample_rate: Audio sample rate.
        volume: Volume level (0.0 to 1.0).
        fade_ms: Fade in/out duration in milliseconds.

    Returns:
        Audio samples as float32 numpy array.
    """
    # Generate time array
    num_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, num_samples, dtype=np.float32)

    # Generate sine wave
    tone = np.sin(2 * np.pi * frequency * t) * volume

    # Apply fade in/out envelope to avoid clicks
    fade_samples = int(sample_rate * fade_ms / 1000)
    if fade_samples > 0 and fade_samples < num_samples // 2:
        # Fade in
        fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
        tone[:fade_samples] *= fade_in
        # Fade out
        fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
        tone[-fade_samples:] *= fade_out

    return tone


def _play_tone_async(frequency: float, duration_ms: int, volume: float = _DEFAULT_VOLUME) -> None:
    """Play a tone asynchronously in a daemon thread.

    Args:
        frequency: Frequency in Hz.
        duration_ms: Duration in milliseconds.
        volume: Volume level (0.0 to 1.0).
    """
    def _play():
        try:
            tone = _generate_tone(frequency, duration_ms, volume=volume)
            sd.play(tone, samplerate=44100)
            sd.wait()
        except Exception:
            # Silently fail - audio feedback is non-critical
            pass

    thread = threading.Thread(target=_play, daemon=True)
    thread.start()


def play_start_beep() -> None:
    """Play a soft tone to indicate recording start.

    A gentle tone (600Hz) for 150ms at 25% volume.
    Non-blocking: plays in a daemon thread.
    """
    _play_tone_async(600, 150, volume=0.25)


def play_stop_beep() -> None:
    """Play a soft tone to indicate recording stop/transcription complete.

    A gentle lower tone (440Hz) for 150ms at 25% volume.
    Non-blocking: plays in a daemon thread.
    """
    _play_tone_async(440, 150, volume=0.25)
