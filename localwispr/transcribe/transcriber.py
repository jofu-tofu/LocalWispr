"""Whisper transcription module for LocalWispr.

Uses pywhispercpp (whisper.cpp Python bindings) for fast, lightweight transcription.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.io import wavfile

from localwispr.config import get_config
from localwispr.transcribe.context import ContextDetector, ContextType
from localwispr.prompts import load_prompt

if TYPE_CHECKING:
    from pywhispercpp.model import Model as WhisperModel
    from localwispr.audio import AudioRecorder

logger = logging.getLogger(__name__)

# Whisper expects 16kHz audio
WHISPER_SAMPLE_RATE = 16000


@dataclass
class TranscriptionResult:
    """Result of a transcription operation.

    Attributes:
        text: The transcribed text.
        segments: List of segment dictionaries with timing info.
        inference_time: Time taken for inference in seconds.
        audio_duration: Duration of the input audio in seconds.
        detected_context: The context detected for this transcription.
        context_detection_time: Time spent on context detection in seconds.
        was_retranscribed: Whether the audio was retranscribed due to context mismatch.
    """

    text: str
    segments: list[dict]
    inference_time: float
    audio_duration: float
    detected_context: ContextType | None = None
    context_detection_time: float = 0.0
    was_retranscribed: bool = False


class WhisperTranscriber:
    """Whisper-based transcriber with lazy model loading.

    Loads the pywhispercpp model on first use to avoid startup delay.
    Uses config settings for model name and device.
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
            compute_type: Compute type for inference (kept for compatibility).
                If None, uses config setting.
        """
        # Load config for defaults
        config = get_config()
        model_config = config["model"]

        self._model_name = model_name or model_config["name"]

        # Get configured device/compute_type (may be "auto")
        config_device = device or model_config["device"]
        config_compute = compute_type or model_config["compute_type"]

        # Resolve "auto" values to actual device/compute_type
        from localwispr.transcribe.device import resolve_device, get_optimal_threads

        self._device, self._compute_type = resolve_device(config_device, config_compute)
        self._n_threads = get_optimal_threads()

        # Language setting (auto = None for whisper.cpp)
        lang = model_config.get("language", "auto")
        self._language = None if lang == "auto" else lang

        # Vocabulary/hotwords for better recognition
        vocab_config = config.get("vocabulary", {})
        self._hotwords = vocab_config.get("words", [])

        # Lazy-loaded model
        self._model: WhisperModel | None = None

        logger.info(
            "transcriber: initialized model=%s device=%s threads=%d",
            self._model_name,
            self._device,
            self._n_threads,
        )

    def _load_model(self) -> "WhisperModel":
        """Load the Whisper model.

        Returns:
            Loaded pywhispercpp Model instance.
        """
        from pywhispercpp.model import Model
        from localwispr.transcribe.model_manager import get_model_path, is_model_downloaded

        # Check if model is downloaded
        if not is_model_downloaded(self._model_name):
            raise RuntimeError(
                f"Model '{self._model_name}' is not downloaded. "
                "Please download it first using the settings window."
            )

        model_path = get_model_path(self._model_name)
        logger.info("transcriber: loading model from %s", model_path)

        # Create model with appropriate settings
        # pywhispercpp handles GPU automatically if compiled with CUDA support
        model = Model(
            model=str(model_path),
            n_threads=self._n_threads,
        )

        logger.info("transcriber: model loaded successfully")
        return model

    @property
    def model(self) -> "WhisperModel":
        """Get the Whisper model, loading it if necessary.

        Returns:
            The loaded pywhispercpp Model instance.
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
    def language(self) -> str | None:
        """Get the language setting (None = auto-detect)."""
        return self._language

    @property
    def hotwords(self) -> list[str]:
        """Get the vocabulary/hotwords list."""
        return self._hotwords

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None

    def _audio_to_temp_wav(self, audio: np.ndarray) -> str:
        """Write audio to a temporary WAV file for pywhispercpp.

        Args:
            audio: Audio data as float32 numpy array, mono, 16kHz.

        Returns:
            Path to the temporary WAV file.
        """
        # Create a temporary file that won't be auto-deleted
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        # Convert float32 [-1, 1] to int16 for WAV
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(temp_path, WHISPER_SAMPLE_RATE, audio_int16)

        return temp_path

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        beam_size: int = 5,
        vad_filter: bool = True,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Audio data as float32 numpy array, mono, 16kHz.
                Should be in range [-1, 1].
            beam_size: Beam size for decoding. Higher is more accurate but slower.
            vad_filter: If True, use voice activity detection to skip silence.
                (Note: pywhispercpp has its own VAD handling)
            initial_prompt: Optional prompt to guide transcription with domain vocabulary.

        Returns:
            TranscriptionResult with text and timing information.
        """
        # Calculate audio duration (assuming 16kHz sample rate)
        audio_duration = len(audio) / WHISPER_SAMPLE_RATE

        # Write audio to temp file for pywhispercpp
        temp_path = self._audio_to_temp_wav(audio)

        try:
            # Run inference with timing
            start_time = time.perf_counter()

            # Build transcribe kwargs for pywhispercpp
            transcribe_kwargs = {}

            # Add language if specified (None/empty = auto-detect)
            if self._language:
                transcribe_kwargs["language"] = self._language

            # Add initial prompt if specified
            if initial_prompt:
                transcribe_kwargs["initial_prompt"] = initial_prompt

            # Add hotwords to initial prompt if configured
            if self._hotwords and not initial_prompt:
                transcribe_kwargs["initial_prompt"] = " ".join(self._hotwords)
            elif self._hotwords and initial_prompt:
                # Append hotwords to the existing prompt
                transcribe_kwargs["initial_prompt"] = (
                    initial_prompt + " " + " ".join(self._hotwords)
                )

            # Transcribe using pywhispercpp
            segments = self.model.transcribe(temp_path, **transcribe_kwargs)

            inference_time = time.perf_counter() - start_time

            # Collect segments and build full text
            segment_list = []
            text_parts = []

            for segment in segments:
                # pywhispercpp Segment has: t0, t1, text
                # t0 and t1 are in centiseconds (1/100th of a second)
                segment_list.append({
                    "start": segment.t0 / 100.0,
                    "end": segment.t1 / 100.0,
                    "text": segment.text,
                })
                text_parts.append(segment.text)

            # Join text parts
            text = "".join(text_parts).strip()

            return TranscriptionResult(
                text=text,
                segments=segment_list,
                inference_time=inference_time,
                audio_duration=audio_duration,
            )

        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def transcribe_with_context(
    audio: np.ndarray,
    transcriber: WhisperTranscriber,
    detector: ContextDetector | None = None,
    **kwargs,
) -> TranscriptionResult:
    """Transcribe audio with intelligent context detection.

    Performs hybrid pre+post context detection:
    1. Pre-detect context from focused window
    2. Transcribe with context-appropriate prompt
    3. Post-detect context from transcribed text
    4. If contexts differ, retranscribe with post-detected context (max 1 retry)

    Args:
        audio: Audio data as float32 numpy array, mono, 16kHz.
        transcriber: WhisperTranscriber to use.
        detector: ContextDetector to use. If None, creates a new one.
        **kwargs: Additional arguments passed to transcribe().

    Returns:
        TranscriptionResult with detected_context populated.
    """
    if detector is None:
        detector = ContextDetector()

    context_start = time.perf_counter()

    # Step 1: Pre-detect context from focused window
    pre_context = detector.detect_from_window()

    # Step 2: Load prompt for pre-detected context
    pre_prompt = load_prompt(pre_context.value)

    context_detection_time = time.perf_counter() - context_start

    # Step 3: Transcribe with initial_prompt
    result = transcriber.transcribe(audio, initial_prompt=pre_prompt, **kwargs)

    # Step 4: Post-detect context from transcribed text
    post_context = detector.detect_from_text(result.text)

    # Step 5: If contexts differ, retranscribe with post-detected context (MAX 1 RETRY)
    was_retranscribed = False
    if pre_context != post_context and post_context != ContextType.GENERAL:
        # Retranscribe with post-detected context prompt
        post_prompt = load_prompt(post_context.value)
        result = transcriber.transcribe(audio, initial_prompt=post_prompt, **kwargs)
        was_retranscribed = True

    # Step 6: Post-detection context is FINAL
    final_context = post_context if post_context != ContextType.GENERAL else pre_context

    # Update result with context information
    result.detected_context = final_context
    result.context_detection_time = context_detection_time
    result.was_retranscribed = was_retranscribed

    return result


def transcribe_recording(
    recorder: "AudioRecorder",
    transcriber: WhisperTranscriber | None = None,
    *,
    use_context: bool = True,
    detector: ContextDetector | None = None,
    **kwargs,
) -> TranscriptionResult:
    """Convenience function to transcribe a recording.

    Stops the recorder, gets Whisper-ready audio, and transcribes it.

    Args:
        recorder: An AudioRecorder that is currently recording.
        transcriber: WhisperTranscriber to use. If None, creates a new one.
        use_context: If True, use context-aware transcription. Defaults to True.
        detector: ContextDetector to use (only when use_context=True).
        **kwargs: Additional arguments passed to transcribe().

    Returns:
        TranscriptionResult with transcribed text and timing info.
    """
    # Get audio from recorder (stops recording and converts to Whisper format)
    audio = recorder.get_whisper_audio()

    # Create transcriber if not provided
    if transcriber is None:
        transcriber = WhisperTranscriber()

    # Use context-aware transcription or direct transcription
    if use_context:
        return transcribe_with_context(audio, transcriber, detector, **kwargs)
    else:
        return transcriber.transcribe(audio, **kwargs)
