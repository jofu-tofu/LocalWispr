"""Recording pipeline for LocalWispr.

This module provides the RecordingPipeline class that coordinates the
recording → transcription → output workflow. It encapsulates the recording
state, model management, and transcription logic that was previously
scattered across TrayApp.

Thread Safety:
    The pipeline uses locks for thread-safe access to recorder and transcriber.
    State updates should only be made from a single thread at a time.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from localwispr.audio import AudioRecorder
    from localwispr.context import ContextDetector
    from localwispr.modes import ModeManager
    from localwispr.transcribe import TranscriptionResult, WhisperTranscriber

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from a recording pipeline execution.

    Attributes:
        success: Whether the pipeline completed successfully.
        text: The transcribed text (empty on failure).
        error: Error message if failed.
        audio_duration: Duration of recorded audio in seconds.
        inference_time: Time taken for transcription in seconds.
        was_retranscribed: Whether context-based retranscription occurred.
    """

    success: bool
    text: str = ""
    error: str = ""
    audio_duration: float = 0.0
    inference_time: float = 0.0
    was_retranscribed: bool = False


class RecordingPipeline:
    """Coordinates recording → transcription → output workflow.

    This class encapsulates all the recording pipeline logic:
    - Audio recording with AudioRecorder
    - Model preloading and management
    - Transcription with mode/context support
    - Thread-safe state management

    Example:
        >>> pipeline = RecordingPipeline(mode_manager=mode_manager)
        >>> pipeline.preload_model_async()
        >>> pipeline.start_recording()
        >>> # ... user speaks ...
        >>> result = pipeline.stop_and_transcribe()
        >>> print(result.text)

    Thread Safety:
        - `start_recording()` and `stop_and_transcribe()` are thread-safe
        - `preload_model_async()` runs in a background thread
        - Model access is protected by a lock
    """

    # Model load timeout in seconds
    MODEL_LOAD_TIMEOUT = 60.0

    def __init__(
        self,
        mode_manager: "ModeManager",
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize the recording pipeline.

        Args:
            mode_manager: Mode manager for transcription mode selection.
            on_error: Optional callback for error messages.
        """
        self._mode_manager = mode_manager
        self._on_error = on_error

        # Pipeline components (lazy loaded)
        self._recorder: "AudioRecorder | None" = None
        self._transcriber: "WhisperTranscriber | None" = None
        self._detector: "ContextDetector | None" = None

        # Thread-safe locks
        self._recorder_lock = threading.Lock()
        self._transcriber_lock = threading.Lock()

        # Model preloading state
        self._model_preload_complete = threading.Event()
        self._model_preload_error: Exception | None = None

        # Volume mute state tracking
        self._was_muted_before_recording = False

        logger.info("pipeline: initialized")

    @property
    def is_recording(self) -> bool:
        """Check if currently recording.

        Returns:
            True if recording is in progress.
        """
        with self._recorder_lock:
            return self._recorder is not None and self._recorder.is_recording

    @property
    def is_model_ready(self) -> bool:
        """Check if the transcription model is ready.

        Returns:
            True if model is loaded and ready.
        """
        return self._model_preload_complete.is_set() and self._model_preload_error is None

    def preload_model_async(self) -> None:
        """Start loading the Whisper model in a background thread.

        This should be called early (e.g., at app startup) to preload
        the ~2GB model before the user's first transcription request.
        """
        thread = threading.Thread(
            target=self._preload_model,
            daemon=True,
        )
        thread.start()
        logger.info("pipeline: model preload thread started")

    def _preload_model(self) -> None:
        """Load Whisper model (runs in background thread)."""
        try:
            from localwispr.transcribe import WhisperTranscriber

            logger.info("model_preload: starting")
            transcriber = WhisperTranscriber()
            # Trigger actual model load by accessing the model property
            _ = transcriber.model

            with self._transcriber_lock:
                self._transcriber = transcriber
            logger.info("model_preload: complete")
        except Exception as e:
            logger.error("model_preload: failed, error_type=%s", type(e).__name__)
            self._model_preload_error = e
        finally:
            self._model_preload_complete.set()

    def start_recording(self, mute_system: bool = False) -> bool:
        """Start audio recording.

        Args:
            mute_system: Whether to mute system audio during recording.

        Returns:
            True if recording started successfully, False on error.
        """
        try:
            # Mute system audio if enabled
            if mute_system:
                from localwispr.volume import mute_system as do_mute

                self._was_muted_before_recording = do_mute()
                logger.debug("recording: system muted (was_muted=%s)", self._was_muted_before_recording)

            # Initialize recorder on first use
            with self._recorder_lock:
                if self._recorder is None:
                    from localwispr.audio import AudioRecorder

                    self._recorder = AudioRecorder()
                    logger.debug("recording: recorder initialized")

                if self._recorder.is_recording:
                    logger.warning("recording: already recording")
                    return False

                self._recorder.start_recording()

            logger.info("recording: started")
            return True

        except Exception as e:
            logger.error("recording: start failed, error_type=%s", type(e).__name__)
            if self._on_error:
                self._on_error("Failed to start recording")
            return False

    def stop_and_transcribe(self, mute_system: bool = False) -> PipelineResult:
        """Stop recording and transcribe the audio.

        Args:
            mute_system: Whether system audio was muted (to restore state).

        Returns:
            PipelineResult with transcription text or error.
        """
        start_time = time.time()

        # Restore system audio mute state
        if mute_system:
            self._restore_system_audio()

        try:
            # Get audio from recorder
            audio = self._get_recorded_audio()
            if audio is None:
                return PipelineResult(success=False, error="No audio captured")

            audio_duration = len(audio) / 16000.0

            # Wait for model to be ready
            if not self._wait_for_model():
                return PipelineResult(success=False, error="Model load timeout")

            # Get transcriber
            transcriber = self._get_transcriber()
            if transcriber is None:
                return PipelineResult(success=False, error="Failed to initialize transcriber")

            # Perform transcription
            result = self._perform_transcription(audio, transcriber)

            total_time = time.time() - start_time
            logger.info("pipeline: complete, total_ms=%d", int(total_time * 1000))

            return PipelineResult(
                success=True,
                text=result.text,
                audio_duration=result.audio_duration,
                inference_time=result.inference_time,
                was_retranscribed=result.was_retranscribed,
            )

        except Exception as e:
            logger.error("pipeline: failed, error_type=%s", type(e).__name__)
            return PipelineResult(success=False, error="Transcription failed")

    def _restore_system_audio(self) -> None:
        """Restore system audio mute state after recording."""
        from localwispr.volume import restore_mute_state

        restore_mute_state(self._was_muted_before_recording)
        logger.debug("recording: system mute restored to %s", self._was_muted_before_recording)

    def _get_recorded_audio(self) -> np.ndarray | None:
        """Get recorded audio from the recorder.

        Returns:
            Audio array if successful, None if no audio or not recording.
        """
        with self._recorder_lock:
            if self._recorder is None or not self._recorder.is_recording:
                logger.warning("recording: not recording, skipping transcription")
                return None
            audio = self._recorder.get_whisper_audio()

        # Check if we have audio
        audio_duration = len(audio) / 16000.0
        if audio_duration < 0.1:
            logger.warning("recording: no audio captured")
            return None

        logger.debug("recording: audio captured, duration_s=%.2f", audio_duration)
        return audio

    def _wait_for_model(self) -> bool:
        """Wait for the Whisper model to finish preloading.

        Returns:
            True if model is ready, False on timeout.
        """
        if not self._model_preload_complete.is_set():
            logger.info("transcriber: waiting for model preload")
            if not self._model_preload_complete.wait(timeout=self.MODEL_LOAD_TIMEOUT):
                logger.error("transcriber: preload timeout")
                return False
        return True

    def _get_transcriber(self) -> "WhisperTranscriber | None":
        """Get or create the transcriber instance.

        Returns:
            WhisperTranscriber instance or None on error.
        """
        from localwispr.transcribe import WhisperTranscriber

        with self._transcriber_lock:
            # Check if preload failed
            if self._model_preload_error is not None:
                logger.error("transcriber: preload had error, retrying sync")
                self._model_preload_error = None
                self._transcriber = None

            # Initialize transcriber if preload failed or wasn't done
            if self._transcriber is None:
                try:
                    logger.info("transcriber: initializing (sync fallback)")
                    self._transcriber = WhisperTranscriber()
                    _ = self._transcriber.model
                    logger.info("transcriber: model loaded")
                except Exception as e:
                    logger.error("transcriber: init failed, error_type=%s", type(e).__name__)
                    return None

            return self._transcriber

    def _perform_transcription(
        self,
        audio: np.ndarray,
        transcriber: "WhisperTranscriber",
    ) -> "TranscriptionResult":
        """Perform transcription using the appropriate mode.

        Args:
            audio: Audio array to transcribe.
            transcriber: WhisperTranscriber instance.

        Returns:
            Transcription result.
        """
        from localwispr.modes import get_mode_prompt

        if self._mode_manager.is_manual_override:
            # Use mode's prompt directly
            initial_prompt = get_mode_prompt()
            result = transcriber.transcribe(audio, initial_prompt=initial_prompt)
            logger.debug(
                "transcription: using mode prompt, mode=%s",
                self._mode_manager.current_mode.name,
            )
        else:
            # Use context detection for automatic mode selection
            if self._detector is None:
                from localwispr.context import ContextDetector

                self._detector = ContextDetector()

            from localwispr.transcribe import transcribe_with_context

            result = transcribe_with_context(audio, transcriber, self._detector)

        inference_time_ms = int(result.inference_time * 1000)
        logger.info(
            "transcription: complete, duration_ms=%d, was_retranscribed=%s",
            inference_time_ms,
            result.was_retranscribed,
        )
        return result

    def get_rms_level(self) -> float:
        """Get current audio level from recorder for visualization.

        Returns:
            Audio RMS level (0.0-1.0), or 0.0 if not recording.
        """
        with self._recorder_lock:
            if self._recorder is not None and self._recorder.is_recording:
                return self._recorder.get_rms_level()
        return 0.0

    def cancel_recording(self) -> None:
        """Cancel the current recording without transcribing."""
        with self._recorder_lock:
            if self._recorder is not None and self._recorder.is_recording:
                # Stop recording but discard audio
                self._recorder.stop_recording()
                logger.info("recording: cancelled")


__all__ = ["RecordingPipeline", "PipelineResult"]
