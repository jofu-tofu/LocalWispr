"""Whisper transcription module for LocalWispr."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from localwispr.config import load_config

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


@dataclass
class TranscriptionResult:
    """Result of a transcription operation.

    Attributes:
        text: The transcribed text.
        segments: List of segment dictionaries with timing info.
        inference_time: Time taken for inference in seconds.
        audio_duration: Duration of the input audio in seconds.
    """

    text: str
    segments: list[dict]
    inference_time: float
    audio_duration: float


class WhisperTranscriber:
    """Whisper-based transcriber with lazy model loading.

    Loads the faster-whisper model on first use to avoid startup delay.
    Uses config settings for model name, device, and compute type.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        """Initialize the transcriber.

        Args:
            model_name: Whisper model name (e.g., "large-v3", "medium").
                If None, uses config setting.
            device: Device to run on ("cuda" or "cpu").
                If None, uses config setting.
            compute_type: Compute type for inference ("float16", "int8", etc.).
                If None, uses config setting.
        """
        # Load config for defaults
        config = load_config()
        model_config = config["model"]

        self._model_name = model_name or model_config["name"]
        self._device = device or model_config["device"]
        self._compute_type = compute_type or model_config["compute_type"]

        # Lazy-loaded model
        self._model: WhisperModel | None = None

    def _load_model(self) -> WhisperModel:
        """Load the Whisper model.

        Returns:
            Loaded WhisperModel instance.
        """
        from faster_whisper import WhisperModel

        return WhisperModel(
            self._model_name,
            device=self._device,
            compute_type=self._compute_type,
        )

    @property
    def model(self) -> WhisperModel:
        """Get the Whisper model, loading it if necessary.

        Returns:
            The loaded WhisperModel instance.
        """
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @property
    def model_name(self) -> str:
        """Get the model name being used."""
        return self._model_name

    @property
    def device(self) -> str:
        """Get the device being used."""
        return self._device

    @property
    def compute_type(self) -> str:
        """Get the compute type being used."""
        return self._compute_type

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Audio data as float32 numpy array, mono, 16kHz.
                Should be in range [-1, 1].
            beam_size: Beam size for decoding. Higher is more accurate but slower.
            vad_filter: If True, use voice activity detection to skip silence.

        Returns:
            TranscriptionResult with text and timing information.
        """
        # Calculate audio duration (assuming 16kHz sample rate)
        audio_duration = len(audio) / 16000.0

        # Run inference with timing
        start_time = time.perf_counter()

        segments_iter, info = self.model.transcribe(
            audio,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=False,
        )

        # Collect segments and build full text
        segments = []
        text_parts = []

        for segment in segments_iter:
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            })
            text_parts.append(segment.text)

        inference_time = time.perf_counter() - start_time

        # Join text parts
        text = "".join(text_parts).strip()

        return TranscriptionResult(
            text=text,
            segments=segments,
            inference_time=inference_time,
            audio_duration=audio_duration,
        )
