"""Whisper transcription module for LocalWispr.

Supports two backends:
- faster-whisper (CTranslate2): Primary backend with native CUDA/GPU support.
- pywhispercpp (whisper.cpp): CPU fallback.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.io import wavfile

from localwispr.config import get_config
from localwispr.transcribe.context import ContextDetector, ContextType
from localwispr.prompts import load_prompt

if TYPE_CHECKING:
    from localwispr.audio import AudioRecorder

logger = logging.getLogger(__name__)

# Whisper expects 16kHz audio
WHISPER_SAMPLE_RATE = 16000

# Backend detection: prefer faster-whisper (CTranslate2 with GPU support)
_FASTER_WHISPER_AVAILABLE = False
try:
    from faster_whisper import WhisperModel as FasterWhisperModel

    _FASTER_WHISPER_AVAILABLE = True
    logger.debug("transcriber: faster-whisper backend available")
except ImportError:
    logger.debug("transcriber: faster-whisper not available, will use pywhispercpp")


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

    Supports two backends:
    - "faster-whisper": CTranslate2-based, with native CUDA/GPU support.
    - "pywhispercpp": whisper.cpp-based, CPU fallback.

    Loads the model on first use to avoid startup delay.
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
            compute_type: Compute type for inference.
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

        # Select backend
        self._backend = "faster-whisper" if _FASTER_WHISPER_AVAILABLE else "pywhispercpp"

        # Language setting (auto = None)
        lang = model_config.get("language", "auto")
        self._language = None if lang == "auto" else lang

        # Vocabulary/hotwords for better recognition
        vocab_config = config.get("vocabulary", {})
        self._hotwords = vocab_config.get("words", [])

        # Lazy-loaded model
        self._model: Any = None

        logger.info(
            "transcriber: initialized model=%s device=%s backend=%s threads=%d",
            self._model_name,
            self._device,
            self._backend,
            self._n_threads,
        )

    def _load_model(self) -> Any:
        """Load the Whisper model using the active backend.

        Returns:
            Loaded model instance (FasterWhisperModel or pywhispercpp Model).
        """
        from localwispr.transcribe.model_manager import is_model_downloaded

        # Check if model is downloaded
        if not is_model_downloaded(self._model_name):
            raise RuntimeError(
                f"Model '{self._model_name}' is not downloaded. "
                "Please download it first using the settings window."
            )

        if self._backend == "faster-whisper":
            return self._load_faster_whisper_model()
        else:
            return self._load_pywhispercpp_model()

    def _load_faster_whisper_model(self) -> Any:
        """Load model using faster-whisper (CTranslate2) backend.

        Returns:
            Loaded FasterWhisperModel instance.
        """
        from faster_whisper import WhisperModel as FWModel

        # Resolve compute_type for CTranslate2
        compute_type = self._compute_type
        if compute_type in ("default", "auto"):
            compute_type = "float16" if self._device == "cuda" else "int8"

        logger.info(
            "transcriber: loading faster-whisper model=%s device=%s compute_type=%s",
            self._model_name,
            self._device,
            compute_type,
        )

        model = FWModel(
            self._model_name,
            device=self._device,
            compute_type=compute_type,
        )

        logger.info("transcriber: faster-whisper model loaded successfully")
        return model

    def _load_pywhispercpp_model(self) -> Any:
        """Load model using pywhispercpp (whisper.cpp) backend.

        Returns:
            Loaded pywhispercpp Model instance.
        """
        from pywhispercpp.model import Model
        from localwispr.transcribe.model_manager import get_model_path

        model_path = get_model_path(self._model_name)
        logger.info("transcriber: loading pywhispercpp model from %s", model_path)

        model = Model(
            model=str(model_path),
            n_threads=self._n_threads,
        )

        logger.info("transcriber: pywhispercpp model loaded successfully")
        return model

    @property
    def model(self) -> Any:
        """Get the Whisper model, loading it if necessary.

        Returns:
            The loaded model instance.
        """
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @property
    def backend(self) -> str:
        """Get the active backend name."""
        return self._backend

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

    def _build_initial_prompt(self, initial_prompt: str | None) -> str | None:
        """Build the initial prompt including hotwords.

        Args:
            initial_prompt: Optional user-provided prompt.

        Returns:
            Combined prompt string, or None if no prompt/hotwords.
        """
        if self._hotwords and not initial_prompt:
            return " ".join(self._hotwords)
        elif self._hotwords and initial_prompt:
            return initial_prompt + " " + " ".join(self._hotwords)
        return initial_prompt

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
        audio_duration = len(audio) / WHISPER_SAMPLE_RATE
        logger.debug(
            "transcribe: backend=%s shape=%s duration=%.2fs min=%.4f max=%.4f mean_abs=%.4f",
            self._backend,
            audio.shape,
            audio_duration,
            float(np.min(audio)),
            float(np.max(audio)),
            float(np.mean(np.abs(audio))),
        )
        if self._backend == "faster-whisper":
            return self._transcribe_faster_whisper(
                audio, beam_size=beam_size, vad_filter=vad_filter,
                initial_prompt=initial_prompt,
            )
        else:
            return self._transcribe_pywhispercpp(
                audio, initial_prompt=initial_prompt,
            )

    def _transcribe_faster_whisper(
        self,
        audio: np.ndarray,
        *,
        beam_size: int = 5,
        vad_filter: bool = True,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe using faster-whisper (CTranslate2) backend.

        faster-whisper accepts numpy arrays directly — no temp file needed.
        """
        audio_duration = len(audio) / WHISPER_SAMPLE_RATE

        start_time = time.perf_counter()

        prompt = self._build_initial_prompt(initial_prompt)

        # Build kwargs for faster-whisper
        transcribe_kwargs: dict[str, Any] = {
            "beam_size": beam_size,
            "vad_filter": vad_filter,
        }

        if self._language:
            transcribe_kwargs["language"] = self._language

        if prompt:
            transcribe_kwargs["initial_prompt"] = prompt

        if self._hotwords:
            transcribe_kwargs["hotwords"] = " ".join(self._hotwords)

        logger.debug(
            "faster_whisper: audio dtype=%s shape=%s min=%.4f max=%.4f mean_abs=%.4f",
            audio.dtype,
            audio.shape,
            float(np.min(audio)),
            float(np.max(audio)),
            float(np.mean(np.abs(audio))),
        )
        logger.debug("faster_whisper: transcribe_kwargs=%s", transcribe_kwargs)

        # faster-whisper accepts numpy array directly
        try:
            segments_iter, _info = self.model.transcribe(audio, **transcribe_kwargs)
        except RuntimeError as e:
            err_msg = str(e).lower()
            if "cublas" in err_msg or "cudnn" in err_msg or "library" in err_msg:
                logger.error(
                    "faster_whisper: CUDA library missing: %s — GPU inference unavailable", e
                )
            raise

        logger.debug(
            "faster_whisper: info language=%s language_prob=%.3f duration=%.2fs",
            getattr(_info, "language", "?"),
            getattr(_info, "language_probability", 0.0),
            getattr(_info, "duration", 0.0),
        )

        # Collect segments and build full text
        segment_list = []
        text_parts = []

        for segment in segments_iter:
            # faster-whisper Segment has: .start, .end, .text (times in seconds)
            segment_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            })
            text_parts.append(segment.text)
            logger.debug(
                "faster_whisper: segment [%.2f-%.2f] text=%r",
                segment.start,
                segment.end,
                segment.text[:80] if segment.text else "",
            )

        inference_time = time.perf_counter() - start_time

        if not segment_list:
            logger.warning(
                "faster_whisper: ZERO segments returned — audio may be silence or VAD filtered all speech"
            )

        text = "".join(text_parts).strip()

        logger.debug(
            "faster_whisper: result segments=%d text_len=%d inference=%.2fs",
            len(segment_list),
            len(text),
            inference_time,
        )

        return TranscriptionResult(
            text=text,
            segments=segment_list,
            inference_time=inference_time,
            audio_duration=audio_duration,
        )

    def _transcribe_pywhispercpp(
        self,
        audio: np.ndarray,
        *,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe using pywhispercpp (whisper.cpp) backend.

        pywhispercpp requires a WAV file path.
        """
        audio_duration = len(audio) / WHISPER_SAMPLE_RATE

        # Write audio to temp file for pywhispercpp
        temp_path = self._audio_to_temp_wav(audio)

        try:
            start_time = time.perf_counter()

            # Build transcribe kwargs for pywhispercpp
            transcribe_kwargs: dict[str, Any] = {}

            if self._language:
                transcribe_kwargs["language"] = self._language

            prompt = self._build_initial_prompt(initial_prompt)
            if prompt:
                transcribe_kwargs["initial_prompt"] = prompt

            logger.debug(
                "pywhispercpp: audio dtype=%s shape=%s min=%.4f max=%.4f mean_abs=%.4f",
                audio.dtype,
                audio.shape,
                float(np.min(audio)),
                float(np.max(audio)),
                float(np.mean(np.abs(audio))),
            )
            logger.debug("pywhispercpp: transcribe_kwargs=%s", transcribe_kwargs)

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

            if not segment_list:
                logger.warning(
                    "pywhispercpp: ZERO segments returned — audio may be silence"
                )

            text = "".join(text_parts).strip()

            logger.debug(
                "pywhispercpp: result segments=%d text_len=%d inference=%.2fs",
                len(segment_list),
                len(text),
                inference_time,
            )

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
