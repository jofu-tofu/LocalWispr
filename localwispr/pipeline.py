"""Recording pipeline for LocalWispr.

This module provides the RecordingPipeline class that coordinates the
recording → transcription → output workflow. It encapsulates the recording
state, model management, and transcription logic that was previously
scattered across TrayApp.

Thread Safety:
    The pipeline uses locks for thread-safe access to recorder and transcriber.
    State updates should only be made from a single thread at a time.
    Async transcription uses generation IDs to handle stale results.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from localwispr.audio import AudioRecorder
    from localwispr.context import ContextDetector
    from localwispr.modes import ModeManager
    from localwispr.streaming import StreamingTranscriber
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
        executor: Executor | None = None,
    ) -> None:
        """Initialize the recording pipeline.

        Args:
            mode_manager: Mode manager for transcription mode selection.
            on_error: Optional callback for error messages.
            executor: Optional executor for async transcription (injectable for testing).
        """
        self._mode_manager = mode_manager
        self._on_error = on_error

        # Pipeline components (lazy loaded)
        self._recorder: "AudioRecorder | None" = None
        self._transcriber: "WhisperTranscriber | None" = None
        self._detector: "ContextDetector | None" = None

        # Streaming transcription components
        self._streaming_transcriber: "StreamingTranscriber | None" = None
        self._streaming_enabled = False

        # Thread-safe locks
        self._recorder_lock = threading.Lock()
        self._transcriber_lock = threading.Lock()

        # Model preloading state
        self._model_preload_complete = threading.Event()
        self._model_preload_error: Exception | None = None

        # Volume mute state tracking
        self._was_muted_before_recording = False

        # Background transcription (injectable executor for testing)
        self._transcription_executor = executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="transcribe",
        )
        self._owns_executor = executor is None  # Only shutdown if we created it
        self._current_generation: int = 0
        self._generation_lock = threading.Lock()
        self._shutting_down = False

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

            # Check if streaming mode is enabled
            self._streaming_enabled = self._check_streaming_enabled()

            # Initialize recorder on first use
            with self._recorder_lock:
                # Check if already recording BEFORE creating new recorder
                if self._recorder is not None and self._recorder.is_recording:
                    logger.warning("recording: already recording")
                    return False

                # Create new recorder for each recording session
                from localwispr.audio import AudioRecorder

                if self._streaming_enabled:
                    # Set up streaming transcription
                    self._init_streaming_transcriber()
                    # Create recorder with streaming callback
                    self._recorder = AudioRecorder(
                        on_chunk=self._on_audio_chunk,
                    )
                    logger.debug("recording: recorder initialized with streaming callback")
                else:
                    self._recorder = AudioRecorder()
                    self._streaming_transcriber = None
                    logger.debug("recording: recorder initialized (batch mode)")

                self._recorder.start_recording()

                # Start streaming transcriber after recorder starts
                if self._streaming_enabled and self._streaming_transcriber is not None:
                    self._streaming_transcriber.start(self._recorder.sample_rate)

            logger.info("recording: started, streaming=%s", self._streaming_enabled)
            return True

        except Exception as e:
            logger.error("recording: start failed, error_type=%s", type(e).__name__)
            if self._on_error:
                self._on_error("Failed to start recording")
            return False

    def _check_streaming_enabled(self) -> bool:
        """Check if streaming transcription is enabled in config.

        Returns:
            True if streaming is enabled.
        """
        from localwispr.config import get_config

        config = get_config()
        streaming = config.get("streaming", {})
        return streaming.get("enabled", False)

    def _init_streaming_transcriber(self) -> None:
        """Initialize streaming transcriber with current transcriber."""
        # Ensure transcriber is ready
        transcriber = self._get_transcriber()
        if transcriber is None:
            logger.warning("streaming: transcriber not ready, falling back to batch mode")
            self._streaming_enabled = False
            return

        from localwispr.streaming import StreamingTranscriber, get_streaming_config

        config = get_streaming_config()
        self._streaming_transcriber = StreamingTranscriber(
            transcriber=transcriber,
            config=config,
        )
        logger.debug("streaming: transcriber initialized")

    def _on_audio_chunk(self, chunk: np.ndarray) -> None:
        """Handle audio chunk for streaming transcription.

        Called from sounddevice callback thread.

        Args:
            chunk: Audio data from recorder.
        """
        if self._streaming_transcriber is not None:
            self._streaming_transcriber.process_chunk(chunk)

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
            # Handle streaming mode differently
            if self._streaming_enabled and self._streaming_transcriber is not None:
                return self._stop_and_transcribe_streaming(start_time)

            # Batch mode: Get audio from recorder
            audio = self._get_recorded_audio()
            if audio is None:
                return PipelineResult(success=False, error="No audio captured")

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

    def _stop_and_transcribe_streaming(self, start_time: float) -> PipelineResult:
        """Handle transcription in streaming mode.

        Args:
            start_time: Time when stop_and_transcribe was called.

        Returns:
            PipelineResult with streaming transcription result.
        """
        # Stop recording first
        with self._recorder_lock:
            if self._recorder is not None and self._recorder.is_recording:
                self._recorder.stop_recording()
                logger.debug("streaming: recording stopped")

        # Finalize streaming transcription
        streaming_result = self._streaming_transcriber.finalize()

        # Clean up streaming transcriber
        self._streaming_transcriber = None
        self._streaming_enabled = False

        total_time = time.time() - start_time
        logger.info(
            "pipeline: streaming complete, segments=%d, total_ms=%d",
            streaming_result.num_segments,
            int(total_time * 1000),
        )

        # Check if we got any transcription
        if not streaming_result.text.strip():
            return PipelineResult(success=False, error="No audio captured")

        return PipelineResult(
            success=True,
            text=streaming_result.text,
            audio_duration=streaming_result.audio_duration,
            inference_time=streaming_result.total_inference_time,
            was_retranscribed=False,
        )

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

        # Clean up streaming transcriber if active
        if self._streaming_transcriber is not None:
            self._streaming_transcriber = None
            self._streaming_enabled = False
            logger.debug("streaming: cancelled")

    def stop_and_transcribe_async(
        self,
        mute_system: bool = False,
        on_result: Callable[[PipelineResult, int], None] | None = None,
        on_complete: Callable[[int], None] | None = None,
    ) -> int:
        """Stop recording and transcribe in background.

        Args:
            mute_system: Whether system was muted (to restore state).
            on_result: Called with (PipelineResult, generation) when transcription completes.
            on_complete: Called with (generation) after on_result, always (success or failure).

        Returns:
            Generation ID for tracking stale results.

        Note:
            Callbacks are invoked on background thread with the generation ID.
            Caller must marshal to UI thread if needed (e.g., via queue or Tk.after()).
        """
        # Increment generation atomically
        with self._generation_lock:
            self._current_generation += 1
            generation = self._current_generation

        logger.info("transcription: async started, gen=%d", generation)

        # ALWAYS restore system audio first, even on early return
        if mute_system:
            self._restore_system_audio()

        # Get audio synchronously (fast - just numpy array copy, <5ms)
        audio = self._get_recorded_audio()
        if audio is None:
            result = PipelineResult(success=False, error="No audio captured")
            if on_result:
                on_result(result, generation)
            if on_complete:
                on_complete(generation)
            return generation

        # Submit background work
        self._transcription_executor.submit(
            self._transcribe_background,
            audio,
            generation,
            on_result,
            on_complete,
        )
        return generation

    def _transcribe_background(
        self,
        audio: np.ndarray,
        generation: int,
        on_result: Callable[[PipelineResult, int], None] | None,
        on_complete: Callable[[int], None] | None,
    ) -> None:
        """Background transcription with generation check and error handling.

        Args:
            audio: Audio array to transcribe.
            generation: Generation ID for staleness check.
            on_result: Callback for transcription result (receives result and generation).
            on_complete: Callback for completion (receives generation, always invoked).
        """
        result: PipelineResult | None = None
        start_time = time.time()

        try:
            # Check if shutting down or stale
            with self._generation_lock:
                if self._shutting_down:
                    logger.info("transcription: skipped, shutting down")
                    return
                if generation != self._current_generation:
                    logger.info(
                        "transcription: discarding stale, gen=%d, current=%d",
                        generation,
                        self._current_generation,
                    )
                    return

            # Wait for model
            if not self._wait_for_model():
                result = PipelineResult(success=False, error="Model load timeout")
            else:
                # Get transcriber
                transcriber = self._get_transcriber()
                if transcriber is None:
                    result = PipelineResult(
                        success=False,
                        error="Failed to initialize transcriber",
                    )
                else:
                    # Perform transcription (uses existing _perform_transcription)
                    trans_result = self._perform_transcription(audio, transcriber)
                    result = PipelineResult(
                        success=True,
                        text=trans_result.text,
                        audio_duration=trans_result.audio_duration,
                        inference_time=trans_result.inference_time,
                        was_retranscribed=trans_result.was_retranscribed,
                    )

        except Exception as e:
            logger.error(
                "transcription: background failed, error_type=%s, error=%s",
                type(e).__name__,
                str(e),
            )
            result = PipelineResult(success=False, error="Transcription failed")

        finally:
            # Check stale again before callbacks
            with self._generation_lock:
                if self._shutting_down or generation != self._current_generation:
                    logger.debug(
                        "transcription: skipping callbacks, stale or shutdown"
                    )
                    return

            elapsed_ms = int((time.time() - start_time) * 1000)
            if result is not None:
                logger.info(
                    "transcription: complete, gen=%d, success=%s, duration_ms=%d",
                    generation,
                    result.success,
                    elapsed_ms,
                )

            # Always invoke callbacks so UI can recover
            # Pass generation to callbacks so they don't need to capture it via closure
            if result is not None and on_result:
                try:
                    on_result(result, generation)
                except Exception as e:
                    logger.error(
                        "transcription: on_result callback failed, error=%s", e
                    )

            if on_complete:
                try:
                    on_complete(generation)
                except Exception as e:
                    logger.error(
                        "transcription: on_complete callback failed, error=%s", e
                    )

    def shutdown(self, timeout: float = 5.0) -> None:
        """Shutdown executor gracefully.

        Args:
            timeout: Max seconds to wait for in-flight work.
        """
        # Set flag to prevent new work and skip callbacks
        with self._generation_lock:
            self._shutting_down = True
            self._current_generation += 1  # Invalidate in-flight

        # Shutdown executor only if we own it
        if self._owns_executor:
            self._transcription_executor.shutdown(wait=True)
            logger.info("pipeline: executor shutdown complete")

    def is_current_generation(self, generation: int) -> bool:
        """Check if generation is still current (not stale).

        Args:
            generation: Generation ID to check.

        Returns:
            True if generation is current and not shutting down.
        """
        with self._generation_lock:
            return generation == self._current_generation and not self._shutting_down

    def get_model_name(self) -> str:
        """Get current Whisper model name for display.

        Returns:
            Model name (e.g., "large-v3"), or empty string if not loaded.
        """
        with self._transcriber_lock:
            if self._transcriber is not None:
                return self._transcriber.model_name
        return ""

    def invalidate_transcriber(self) -> None:
        """Invalidate transcriber so it's recreated with new settings.

        Use this when settings change (e.g., model name, device).
        """
        with self._transcriber_lock:
            if self._transcriber is not None:
                logger.info("pipeline: invalidating transcriber for settings reload")
                self._transcriber = None

    def clear_model_preload(self) -> None:
        """Clear model preload state so model reloads on next use.

        Use this when model-related settings change.
        """
        self._model_preload_complete.clear()
        self._model_preload_error = None
        logger.debug("pipeline: model preload state cleared")


__all__ = ["RecordingPipeline", "PipelineResult"]
