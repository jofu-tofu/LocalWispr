"""Whisper transcription module for LocalWispr."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from localwispr.config import get_config
from localwispr.context import ContextDetector, ContextType
from localwispr.prompts import load_prompt

if TYPE_CHECKING:
    from faster_whisper import WhisperModel
    from localwispr.audio import AudioRecorder


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
        config = get_config()
        model_config = config["model"]

        self._model_name = model_name or model_config["name"]
        self._device = device or model_config["device"]
        self._compute_type = compute_type or model_config["compute_type"]

        # Language setting (auto = None for faster-whisper)
        lang = model_config.get("language", "auto")
        self._language = None if lang == "auto" else lang

        # Vocabulary/hotwords for better recognition
        vocab_config = config.get("vocabulary", {})
        self._hotwords = vocab_config.get("words", [])

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
            initial_prompt: Optional prompt to guide transcription with domain vocabulary.

        Returns:
            TranscriptionResult with text and timing information.
        """
        # Calculate audio duration (assuming 16kHz sample rate)
        audio_duration = len(audio) / 16000.0

        # Run inference with timing
        start_time = time.perf_counter()

        # Build transcribe kwargs
        transcribe_kwargs = {
            "beam_size": beam_size,
            "vad_filter": vad_filter,
            "word_timestamps": False,
            "initial_prompt": initial_prompt,
        }

        # Add language if specified (None = auto-detect)
        if self._language is not None:
            transcribe_kwargs["language"] = self._language

        # Add hotwords if configured (for better recognition of custom vocabulary)
        if self._hotwords:
            transcribe_kwargs["hotwords"] = " ".join(self._hotwords)

        segments_iter, info = self.model.transcribe(audio, **transcribe_kwargs)

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
    recorder: AudioRecorder,
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
