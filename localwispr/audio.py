"""Audio capture module for LocalWispr."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import sounddevice as sd
from scipy import signal

# Whisper model requirements
WHISPER_SAMPLE_RATE: int = 16000
WHISPER_DTYPE = np.float32


def prepare_for_whisper(audio: np.ndarray, source_rate: int) -> np.ndarray:
    """Convert audio to Whisper-compatible format.

    Converts audio to 16kHz mono float32 normalized to [-1, 1] range,
    which is the format expected by Whisper models.

    Args:
        audio: Input audio as NumPy array. Can be mono (1D) or stereo (2D).
        source_rate: Sample rate of the input audio in Hz.

    Returns:
        Audio array as float32, mono, resampled to 16kHz, normalized to [-1, 1].
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

    # Resample to 16kHz if needed
    if source_rate != WHISPER_SAMPLE_RATE:
        # Calculate number of samples in output
        num_samples = int(len(audio) * WHISPER_SAMPLE_RATE / source_rate)
        audio = signal.resample(audio, num_samples)
        audio = audio.astype(WHISPER_DTYPE)

    # Normalize to [-1, 1] range
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Ensure output is float32
    return audio.astype(WHISPER_DTYPE)


class AudioRecorderError(Exception):
    """Exception raised for audio recording errors."""

    pass


class AudioRecorder:
    """Audio recorder for capturing microphone input.

    Captures audio from the system's default input device using sounddevice.
    Audio is stored as chunks and returned as a NumPy array when recording stops.
    """

    def __init__(self, device: int | str | None = None) -> None:
        """Initialize the audio recorder.

        Args:
            device: Audio input device. None uses the system default.
                Can be device index (int) or device name substring (str).
        """
        self._device = device
        self._stream: sd.InputStream | None = None
        self._chunks: list[np.ndarray] = []
        self._recording = False
        self._lock = threading.Lock()
        self._peak_level: float = 0.0
        self._rms_level: float = 0.0

        # Get device info to determine native sample rate
        self._device_info = self._get_device_info()
        self._sample_rate: int = int(self._device_info["default_samplerate"])
        self._channels: int = 1  # Mono recording

    def _get_device_info(self) -> dict[str, Any]:
        """Get information about the input device.

        Returns:
            Device info dictionary from sounddevice.

        Raises:
            AudioRecorderError: If device cannot be found or accessed.
        """
        try:
            if self._device is None:
                # Get default input device
                device_index = sd.default.device[0]
                if device_index is None:
                    device_index = sd.query_devices(kind="input")["index"]
                return sd.query_devices(device_index)
            else:
                return sd.query_devices(self._device)
        except Exception as e:
            raise AudioRecorderError(f"Failed to get device info: {e}") from e

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback function called by sounddevice for each audio chunk.

        Args:
            indata: Input audio data as NumPy array.
            frames: Number of frames in this chunk.
            time_info: Timing information (unused).
            status: Status flags from sounddevice.
        """
        if status:
            # Log status flags (overflow/underflow warnings)
            pass  # Could add logging here if needed

        with self._lock:
            if self._recording:
                # Store a copy of the audio data
                self._chunks.append(indata.copy())

                # Update audio levels
                audio_data = indata.flatten()
                self._peak_level = float(np.max(np.abs(audio_data)))
                self._rms_level = float(np.sqrt(np.mean(audio_data**2)))

    def start_recording(self) -> None:
        """Begin capturing audio from the microphone.

        Raises:
            AudioRecorderError: If recording is already in progress or
                if the audio stream cannot be opened.
        """
        with self._lock:
            if self._recording:
                raise AudioRecorderError("Recording is already in progress")

            self._chunks = []
            self._peak_level = 0.0
            self._rms_level = 0.0
            self._recording = True

        try:
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                device=self._device,
                channels=self._channels,
                dtype=np.float32,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as e:
            with self._lock:
                self._recording = False
            raise AudioRecorderError(f"Failed to start recording: {e}") from e

    def stop_recording(self) -> np.ndarray:
        """Stop recording and return the captured audio data.

        Returns:
            NumPy array of audio data (float32, mono) at the device's
            native sample rate.

        Raises:
            AudioRecorderError: If no recording is in progress.
        """
        with self._lock:
            if not self._recording:
                raise AudioRecorderError("No recording in progress")

            self._recording = False

        # Stop and close the stream
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass  # Ignore errors during cleanup
            self._stream = None

        # Concatenate all chunks into a single array
        with self._lock:
            if not self._chunks:
                return np.array([], dtype=np.float32)

            audio_data = np.concatenate(self._chunks, axis=0)
            self._chunks = []

        # Flatten to 1D array (mono)
        return audio_data.flatten()

    @property
    def is_recording(self) -> bool:
        """Check if recording is currently in progress.

        Returns:
            True if recording, False otherwise.
        """
        with self._lock:
            return self._recording

    @property
    def sample_rate(self) -> int:
        """Get the sample rate being used for recording.

        Returns:
            Sample rate in Hz.
        """
        return self._sample_rate

    @property
    def device_name(self) -> str:
        """Get the name of the audio input device.

        Returns:
            Device name string.
        """
        return str(self._device_info.get("name", "Unknown Device"))

    def get_audio_level(self) -> float:
        """Get the current peak audio level.

        Returns:
            Peak audio level from 0.0 to 1.0.
            Returns 0.0 if not recording.
        """
        with self._lock:
            return self._peak_level

    def get_rms_level(self) -> float:
        """Get the current RMS (average) audio level.

        Returns:
            RMS audio level from 0.0 to 1.0.
            Returns 0.0 if not recording.
        """
        with self._lock:
            return self._rms_level

    def get_whisper_audio(self) -> np.ndarray:
        """Stop recording and return audio ready for Whisper inference.

        Convenience method that stops the recording and converts the captured
        audio to Whisper-compatible format (16kHz mono float32).

        Returns:
            Audio array ready for Whisper: float32, mono, 16kHz, normalized [-1, 1].

        Raises:
            AudioRecorderError: If no recording is in progress.
        """
        # Stop recording and get raw audio
        raw_audio = self.stop_recording()

        # Convert to Whisper format
        return prepare_for_whisper(raw_audio, self._sample_rate)


def list_audio_devices() -> list[dict[str, Any]]:
    """List all available audio input devices.

    Returns:
        List of device info dictionaries with keys:
        - index: Device index
        - name: Device name
        - max_input_channels: Number of input channels
        - default_samplerate: Default sample rate
        - is_default: Whether this is the default input device
    """
    devices = []
    default_input = sd.default.device[0]

    for i, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            devices.append({
                "index": i,
                "name": device["name"],
                "max_input_channels": device["max_input_channels"],
                "default_samplerate": device["default_samplerate"],
                "is_default": i == default_input,
            })

    return devices
