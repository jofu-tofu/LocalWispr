"""Audio format conversion for Whisper compatibility."""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

# Whisper model requirements
WHISPER_SAMPLE_RATE: int = 16000
WHISPER_DTYPE = np.float32


def prepare_for_whisper(
    audio: np.ndarray,
    source_rate: int,
    normalize: bool = True,
    reference_max: float | None = None,
) -> np.ndarray:
    """Convert audio to Whisper-compatible format.

    Converts audio to 16kHz mono float32 normalized to [-1, 1] range,
    which is the format expected by Whisper models.

    Args:
        audio: Input audio as NumPy array. Can be mono (1D) or stereo (2D).
        source_rate: Sample rate of the input audio in Hz.
        normalize: Whether to normalize audio to [-1, 1] range. Default True.
        reference_max: If provided, normalize relative to this value instead
            of the audio's own max. Used for consistent normalization across
            multiple chunks in streaming mode.

    Returns:
        Audio array as float32, mono, resampled to 16kHz, optionally normalized.
    """
    if audio.size == 0:
        return np.array([], dtype=WHISPER_DTYPE)

    # Convert to float32 if needed
    audio = audio.astype(WHISPER_DTYPE)

    # Convert stereo to mono if needed (average channels)
    if audio.ndim == 2:
        # Shape is (samples, channels) - average across channels
        audio = np.mean(audio, axis=1)
    elif audio.ndim > 2:
        raise ValueError(f"Audio must be 1D or 2D, got {audio.ndim}D")

    # Ensure 1D array
    audio = audio.flatten()

    # Resample to 16kHz if needed using polyphase filtering
    # Polyphase is better for speech than FFT-based resampling
    if source_rate != WHISPER_SAMPLE_RATE:
        g = gcd(WHISPER_SAMPLE_RATE, source_rate)
        up = WHISPER_SAMPLE_RATE // g
        down = source_rate // g
        audio = resample_poly(audio, up, down)
        audio = audio.astype(WHISPER_DTYPE)

    # Normalize to [-1, 1] range
    if normalize:
        if reference_max is not None and reference_max > 0:
            # Use provided reference for consistent normalization
            audio = audio / reference_max
        else:
            # Use audio's own max
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

    # Ensure output is float32
    return audio.astype(WHISPER_DTYPE)
