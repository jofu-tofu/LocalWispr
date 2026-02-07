"""Audio recorder for capturing microphone input."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import sounddevice as sd

if TYPE_CHECKING:
    pass  # Future type imports


class AudioRecorderError(Exception):
    """Exception raised for audio recording errors."""

    pass


class AudioRecorder:
    """Audio recorder for capturing microphone input.

    Captures audio from the system's default input device using sounddevice.
    Audio is stored as chunks and returned as a NumPy array when recording stops.

    Supports an optional on_chunk callback for streaming transcription,
    allowing real-time processing of audio during recording.
    """

    def __init__(
        self,
        device: int | str | None = None,
        on_chunk: "Callable[[np.ndarray], None] | None" = None,
    ) -> None:
        """Initialize the audio recorder.

        Args:
            device: Audio input device. None uses the system default.
                Can be device index (int) or device name substring (str).
            on_chunk: Optional callback invoked for each audio chunk during
                recording. Used for streaming transcription. The callback
                receives the raw audio chunk (numpy array) at the device's
                native sample rate.
        """
        self._device = device
        self._on_chunk = on_chunk
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

        # Make a copy for storage and potential callback
        chunk_copy = indata.copy()

        with self._lock:
            if self._recording:
                # Store a copy of the audio data
                self._chunks.append(chunk_copy)

                # Update audio levels
                audio_data = chunk_copy.flatten()
                self._peak_level = float(np.max(np.abs(audio_data)))
                self._rms_level = float(np.sqrt(np.mean(audio_data**2)))

        # Notify streaming callback outside the lock to avoid blocking
        if self._on_chunk is not None and self._recording:
            try:
                self._on_chunk(chunk_copy)
            except Exception:
                pass  # Don't let callback errors disrupt recording

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
        from localwispr.audio.format import prepare_for_whisper

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
