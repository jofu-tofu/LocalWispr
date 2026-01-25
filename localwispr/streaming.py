"""Streaming transcription module with VAD-based chunking.

This module provides incremental transcription during recording, using
Voice Activity Detection (VAD) to detect natural speech boundaries and
trigger transcription of segments while recording continues.

Key classes:
    - AudioBuffer: Thread-safe rolling buffer for audio chunks
    - VADProcessor: Silero VAD wrapper for speech/silence detection
    - StreamingTranscriber: Orchestrates streaming transcription pipeline
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from localwispr.audio import WHISPER_SAMPLE_RATE, prepare_for_whisper
from localwispr.context import ContextDetector, ContextType
from localwispr.prompts import load_prompt

if TYPE_CHECKING:
    from localwispr.transcribe import WhisperTranscriber

logger = logging.getLogger(__name__)

# VAD model sample rate (Silero requires 16kHz)
VAD_SAMPLE_RATE = 16000


@dataclass
class StreamingResult:
    """Result from streaming transcription.

    Attributes:
        text: Final combined transcription text.
        segments: List of transcribed segment texts.
        total_inference_time: Total time spent on all transcriptions.
        audio_duration: Total audio duration in seconds.
        num_segments: Number of segments transcribed.
        detected_context: The context used/detected for transcription.
        context_was_locked: Whether context was locked during transcription.
    """

    text: str
    segments: list[str]
    total_inference_time: float
    audio_duration: float
    num_segments: int
    detected_context: ContextType | None = None
    context_was_locked: bool = False


@dataclass
class SpeechSegment:
    """A detected speech segment with timing info.

    Attributes:
        start_sample: Start position in samples.
        end_sample: End position in samples.
        is_final: Whether this segment ended with confirmed silence.
    """

    start_sample: int
    end_sample: int
    is_final: bool = False


class AudioBuffer:
    """Thread-safe rolling buffer for streaming audio.

    Stores incoming audio chunks at NATIVE sample rate and batch-converts
    the entire buffer when get_pending() is called. This avoids per-chunk
    resampling artifacts and enables consistent global normalization.

    The conversion happens on each get_pending() call to ensure the entire
    audio stream is processed together with polyphase resampling.

    Thread Safety:
        All methods are thread-safe via internal locking.
    """

    def __init__(self, max_duration_seconds: float = 300.0) -> None:
        """Initialize the audio buffer.

        Args:
            max_duration_seconds: Maximum audio duration to store.
                Older audio is discarded when exceeded.
        """
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._total_samples = 0  # Total samples stored (at native rate)
        self._transcribed_samples = 0  # Position in converted 16kHz audio (offset)
        self._max_duration = max_duration_seconds
        self._source_sample_rate: int = WHISPER_SAMPLE_RATE
        self._global_max: float = 0.0  # Track max amplitude for normalization

    def set_source_sample_rate(self, rate: int) -> None:
        """Set the source sample rate for incoming audio.

        Args:
            rate: Source sample rate in Hz.
        """
        with self._lock:
            self._source_sample_rate = rate

    def get_transcribed_samples(self) -> int:
        """Get number of samples already transcribed (at 16kHz rate).

        Returns:
            Number of 16kHz samples that have been marked as transcribed.
        """
        with self._lock:
            return self._transcribed_samples

    def append(self, chunk: np.ndarray) -> None:
        """Add an audio chunk to the buffer.

        Chunks are stored at native sample rate. Conversion to 16kHz happens
        when get_pending() or get_full_audio() is called.

        Args:
            chunk: Audio data from sounddevice callback.
        """
        with self._lock:
            if chunk.size == 0:
                return

            # Convert to float32 and flatten
            chunk = chunk.astype(np.float32).flatten()

            # Track global max for consistent normalization
            chunk_max = float(np.max(np.abs(chunk)))
            if chunk_max > self._global_max:
                self._global_max = chunk_max

            # Store at native rate
            self._chunks.append(chunk)
            self._total_samples += len(chunk)

            # Enforce max buffer size by removing old chunks
            self._enforce_max_size()

    def _enforce_max_size(self) -> None:
        """Remove oldest chunks if buffer exceeds max duration."""
        max_samples = int(self._max_duration * self._source_sample_rate)
        while self._total_samples > max_samples and self._chunks:
            removed = self._chunks.pop(0)
            self._total_samples -= len(removed)
            # Adjust transcribed samples pointer (convert to native rate first)
            removed_16k = int(len(removed) * WHISPER_SAMPLE_RATE / self._source_sample_rate)
            self._transcribed_samples = max(0, self._transcribed_samples - removed_16k)

    def get_pending(self) -> np.ndarray:
        """Get untranscribed audio converted to 16kHz with global normalization.

        This is the main method for getting audio ready for VAD and transcription.
        Audio is batch-converted from native rate to 16kHz using polyphase
        resampling, then normalized using the global max for consistent amplitude.

        Returns:
            Audio array at 16kHz, normalized using global max amplitude.
        """
        with self._lock:
            if not self._chunks:
                return np.array([], dtype=np.float32)

            # Concatenate all chunks at native rate
            all_audio = np.concatenate(self._chunks)

            # Convert entire buffer to 16kHz with global normalization
            # This avoids per-chunk resampling artifacts
            converted = prepare_for_whisper(
                all_audio,
                self._source_sample_rate,
                normalize=True,
                reference_max=self._global_max if self._global_max > 0 else None,
            )

            # Return untranscribed portion
            return converted[self._transcribed_samples:].copy()

    def get_full_audio(self) -> np.ndarray:
        """Get all buffered audio converted to 16kHz.

        Returns:
            Complete audio array at 16kHz, normalized.
        """
        with self._lock:
            if not self._chunks:
                return np.array([], dtype=np.float32)

            all_audio = np.concatenate(self._chunks)
            return prepare_for_whisper(
                all_audio,
                self._source_sample_rate,
                normalize=True,
                reference_max=self._global_max if self._global_max > 0 else None,
            )

    def mark_transcribed(self, samples: int) -> None:
        """Mark audio as transcribed.

        Args:
            samples: Number of 16kHz samples transcribed from pending audio.
        """
        with self._lock:
            # Calculate total 16kHz samples in buffer
            total_16k = int(self._total_samples * WHISPER_SAMPLE_RATE / self._source_sample_rate)
            self._transcribed_samples = min(
                self._transcribed_samples + samples,
                total_16k,
            )

    def get_pending_duration(self) -> float:
        """Get duration of pending audio in seconds.

        Returns:
            Duration in seconds.
        """
        with self._lock:
            # Calculate pending at 16kHz rate
            total_16k = int(self._total_samples * WHISPER_SAMPLE_RATE / self._source_sample_rate)
            pending_samples = total_16k - self._transcribed_samples
            return pending_samples / WHISPER_SAMPLE_RATE

    def get_total_duration(self) -> float:
        """Get total buffered audio duration in seconds.

        Returns:
            Duration in seconds.
        """
        with self._lock:
            return self._total_samples / self._source_sample_rate

    def clear(self) -> None:
        """Clear all buffered audio."""
        with self._lock:
            self._chunks = []
            self._total_samples = 0
            self._transcribed_samples = 0
            self._global_max = 0.0


class VADProcessor:
    """Voice Activity Detection using Silero VAD.

    Detects speech and silence in audio to find natural segment
    boundaries for transcription.

    Silero VAD is loaded lazily on first use to avoid import-time
    overhead.
    """

    def __init__(
        self,
        min_silence_ms: int = 800,
        min_speech_ms: int = 250,
        threshold: float = 0.5,
    ) -> None:
        """Initialize VAD processor.

        Args:
            min_silence_ms: Minimum silence duration to trigger segment end.
            min_speech_ms: Minimum speech duration to consider valid.
            threshold: Speech probability threshold (0.0-1.0).
        """
        self._min_silence_samples = int(min_silence_ms * VAD_SAMPLE_RATE / 1000)
        self._min_speech_samples = int(min_speech_ms * VAD_SAMPLE_RATE / 1000)
        self._threshold = threshold

        # Lazy-loaded VAD model
        self._model = None
        self._model_lock = threading.Lock()

        # Tracking state
        self._speech_start: int | None = None
        self._silence_start: int | None = None
        self._processed_samples = 0

    def _load_model(self):
        """Load Silero VAD model."""
        import torch

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        return model

    @property
    def model(self):
        """Get VAD model, loading if necessary."""
        with self._model_lock:
            if self._model is None:
                logger.info("streaming: loading Silero VAD model")
                self._model = self._load_model()
                logger.info("streaming: Silero VAD model loaded")
            return self._model

    def reset(self) -> None:
        """Reset VAD state for new recording."""
        self._speech_start = None
        self._silence_start = None
        self._processed_samples = 0

    def process(self, audio: np.ndarray) -> list[SpeechSegment]:
        """Process audio and detect speech segments.

        Args:
            audio: Audio array (16kHz, mono, float32).

        Returns:
            List of detected speech segments with final silence.
        """
        import torch

        if audio.size == 0:
            return []

        segments = []

        # Process in 32ms windows (512 samples at 16kHz)
        window_size = 512
        start_idx = 0

        while start_idx + window_size <= len(audio):
            window = audio[start_idx : start_idx + window_size]

            # Convert to torch tensor
            tensor = torch.from_numpy(window).float()

            # Get speech probability
            with torch.no_grad():
                prob = self.model(tensor, VAD_SAMPLE_RATE).item()

            is_speech = prob >= self._threshold
            current_sample = self._processed_samples + start_idx

            if is_speech:
                # Start tracking speech if not already
                if self._speech_start is None:
                    self._speech_start = current_sample
                    logger.debug("streaming: speech started at sample %d", current_sample)
                # Reset silence tracking
                self._silence_start = None
            else:
                # Track silence
                if self._speech_start is not None:
                    if self._silence_start is None:
                        self._silence_start = current_sample

                    # Check if silence is long enough to end segment
                    silence_duration = current_sample - self._silence_start
                    if silence_duration >= self._min_silence_samples:
                        speech_duration = self._silence_start - self._speech_start
                        if speech_duration >= self._min_speech_samples:
                            segments.append(
                                SpeechSegment(
                                    start_sample=self._speech_start,
                                    end_sample=self._silence_start,
                                    is_final=True,
                                )
                            )
                            logger.debug(
                                "streaming: segment detected, start=%d, end=%d, duration_ms=%d",
                                self._speech_start,
                                self._silence_start,
                                int(speech_duration * 1000 / VAD_SAMPLE_RATE),
                            )
                        # Reset for next segment
                        self._speech_start = None
                        self._silence_start = None

            start_idx += window_size

        # Update processed samples count
        self._processed_samples += len(audio)

        return segments

    def is_speech(self, audio: np.ndarray) -> bool:
        """Quick check if audio contains speech.

        Args:
            audio: Audio array (16kHz, mono, float32).

        Returns:
            True if speech detected above threshold.
        """
        import torch

        if audio.size < 512:
            return False

        # Use first 512 samples
        window = audio[:512]
        tensor = torch.from_numpy(window).float()

        with torch.no_grad():
            prob = self.model(tensor, VAD_SAMPLE_RATE).item()

        return prob >= self._threshold


@dataclass
class StreamingConfig:
    """Configuration for streaming transcription.

    Attributes:
        enabled: Whether streaming mode is enabled.
        min_silence_ms: Minimum silence to trigger segment boundary.
        max_segment_duration: Maximum segment duration before forced split.
        min_segment_duration: Minimum segment duration to transcribe.
        overlap_ms: Audio overlap between segments for context.
        context_check_interval: Check context every N segments.
        context_lock_threshold: Lock context after N segments.
        context_word_threshold: Or lock after N words, whichever comes first.
    """

    enabled: bool = False
    min_silence_ms: int = 800
    max_segment_duration: float = 20.0
    min_segment_duration: float = 2.0
    overlap_ms: int = 100
    context_check_interval: int = 3
    context_lock_threshold: int = 4
    context_word_threshold: int = 50


class StreamingTranscriber:
    """Manages incremental transcription during recording.

    Coordinates the AudioBuffer, VADProcessor, and WhisperTranscriber
    to process audio in segments during recording, providing progressive
    transcription results.

    Thread Safety:
        All methods are thread-safe. Audio chunks can be added from the
        sounddevice callback thread while transcription runs in background.
    """

    def __init__(
        self,
        transcriber: "WhisperTranscriber",
        config: StreamingConfig | None = None,
        on_partial: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize streaming transcriber.

        Args:
            transcriber: WhisperTranscriber instance for transcription.
            config: Streaming configuration. Uses defaults if None.
            on_partial: Optional callback for partial transcription updates.
        """
        self._transcriber = transcriber
        self._config = config or StreamingConfig()
        self._on_partial = on_partial

        # Core components
        self._buffer = AudioBuffer()
        self._vad = VADProcessor(
            min_silence_ms=self._config.min_silence_ms,
            min_speech_ms=250,
        )

        # Transcription state
        self._segments: list[str] = []
        self._total_inference_time = 0.0
        self._lock = threading.Lock()

        # Background transcription thread
        self._pending_segments: list[SpeechSegment] = []
        self._transcription_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Track VAD progress separately from transcription progress
        # This prevents re-processing the same audio through VAD multiple times
        self._vad_lock = threading.Lock()  # Dedicated lock for VAD tracker
        self._vad_processed_samples = 0
        self._last_transcribed_samples = 0  # Track for wraparound detection

        # Context detection state (initialized in start())
        self._context_detector: ContextDetector | None = None
        self._current_context: ContextType = ContextType.GENERAL
        self._current_prompt: str | None = None
        self._context_locked: bool = False
        self._segments_since_context_check: int = 0
        self._total_words: int = 0

        logger.info("streaming: transcriber initialized, config=%s", self._config)

    def start(self, source_sample_rate: int) -> None:
        """Start streaming transcription.

        Args:
            source_sample_rate: Sample rate of incoming audio.
        """
        self._buffer.set_source_sample_rate(source_sample_rate)
        self._buffer.clear()
        self._vad.reset()
        self._vad_processed_samples = 0  # Reset VAD progress tracker
        self._last_transcribed_samples = 0  # Reset wraparound tracker
        self._segments = []
        self._total_inference_time = 0.0
        self._pending_segments = []
        self._stop_event.clear()

        # Initialize context detection
        self._context_detector = ContextDetector()
        self._current_context = self._context_detector.detect_from_window()
        self._current_prompt = load_prompt(self._current_context.value)
        self._context_locked = False
        self._segments_since_context_check = 0
        self._total_words = 0
        logger.info("streaming: initial context from window: %s", self._current_context.value)

        # Start background transcription thread
        self._transcription_thread = threading.Thread(
            target=self._transcription_loop,
            daemon=True,
        )
        self._transcription_thread.start()
        logger.info("streaming: started")

    def process_chunk(self, chunk: np.ndarray) -> None:
        """Process an audio chunk from recording callback.

        Thread-safe. Called from sounddevice callback thread.

        Args:
            chunk: Audio data from sounddevice.
        """
        # Detect buffer wraparound (max_duration exceeded, old chunks removed)
        # This handles recordings > 5 minutes gracefully
        current_transcribed = self._buffer.get_transcribed_samples()
        if current_transcribed < self._last_transcribed_samples:
            # Wraparound detected - reset VAD completely
            logger.warning("streaming: buffer wraparound detected, resetting VAD")
            self._vad.reset()

            with self._vad_lock:
                self._vad_processed_samples = 0

            with self._lock:
                self._pending_segments.clear()

        self._last_transcribed_samples = current_transcribed

        # Add to buffer
        self._buffer.append(chunk)

        # Run VAD on NEW audio only (not all pending)
        # This prevents re-processing the same audio and avoids inflated sample positions
        pending = self._buffer.get_pending()
        pending_size = pending.size

        # Read VAD offset with dedicated lock (minimizes contention)
        with self._vad_lock:
            vad_offset = self._vad_processed_samples
            should_process = pending_size > vad_offset

        if should_process:
            # Run VAD outside lock (expensive operation)
            new_audio = pending[vad_offset:]
            segments = self._vad.process(new_audio)

            # Update tracker with dedicated lock
            with self._vad_lock:
                self._vad_processed_samples = pending_size

            # Queue segments with main lock
            if segments:
                # Translate VAD's global positions to pending-relative coordinates
                pending_offset = self._buffer.get_transcribed_samples()

                with self._lock:
                    for seg in segments:
                        translated_seg = SpeechSegment(
                            start_sample=seg.start_sample - pending_offset,
                            end_sample=seg.end_sample - pending_offset,
                            is_final=seg.is_final,
                        )
                        self._pending_segments.append(translated_seg)

        # Check for max duration forced split
        pending_duration = self._buffer.get_pending_duration()
        if pending_duration >= self._config.max_segment_duration:
            logger.debug("streaming: forcing segment at %.2fs", pending_duration)

            # Reset VAD (safe: VAD is only written from sounddevice thread,
            # transcription thread only reads via process() which has internal locking)
            self._vad.reset()

            # Queue segment with main lock
            with self._lock:
                # Create forced segment
                pending_samples = int(pending_duration * WHISPER_SAMPLE_RATE)
                self._pending_segments.append(
                    SpeechSegment(
                        start_sample=0,
                        end_sample=pending_samples,
                        is_final=True,
                    )
                )

            # Reset tracker with dedicated lock
            with self._vad_lock:
                self._vad_processed_samples = 0

    def _transcription_loop(self) -> None:
        """Background thread for processing transcription segments."""
        max_drain_time = 30.0  # Maximum time to spend draining queue (seconds)
        drain_start = None  # Track when draining started

        while True:
            if self._stop_event.is_set():
                with self._lock:
                    if not self._pending_segments:
                        break
                    # Continue processing remaining segments
                # Check if we're taking too long to drain
                if drain_start is None:
                    drain_start = time.perf_counter()
                    with self._lock:
                        pending_count = len(self._pending_segments)
                    if pending_count > 0:
                        logger.debug("streaming: draining %d pending segments", pending_count)
                elif time.perf_counter() - drain_start > max_drain_time:
                    remaining = len(self._pending_segments)
                    logger.warning(
                        "streaming: drain timeout, abandoning %d segments after %.1fs",
                        remaining,
                        max_drain_time,
                    )
                    break

            segment = None
            with self._lock:
                if self._pending_segments:
                    segment = self._pending_segments.pop(0)

            if segment is not None:
                self._transcribe_segment(segment)
            else:
                # Wait a bit before checking again
                time.sleep(0.05)

    def _maybe_update_context(self, segment_text: str) -> None:
        """Check if context should be updated based on transcribed text.

        Called after each segment transcription. Updates context based on:
        - Number of segments since last check (context_check_interval)
        - Total word count vs threshold (context_word_threshold)
        - Whether context is already locked (context_lock_threshold)

        Args:
            segment_text: Text from the just-transcribed segment.
        """
        if self._context_locked or self._context_detector is None:
            return

        # Count words in this segment
        word_count = len(segment_text.split())
        self._total_words += word_count
        self._segments_since_context_check += 1

        # Check if we should lock context (threshold reached)
        num_segments = len(self._segments)
        if (
            num_segments >= self._config.context_lock_threshold
            or self._total_words >= self._config.context_word_threshold
        ):
            self._context_locked = True
            logger.info(
                "streaming: context locked to %s after %d segments, %d words",
                self._current_context.value,
                num_segments,
                self._total_words,
            )
            return

        # Check if it's time to re-evaluate context
        if self._segments_since_context_check >= self._config.context_check_interval:
            self._segments_since_context_check = 0

            # Combine all transcribed text
            combined_text = " ".join(self._segments)

            # Detect context from accumulated text
            new_context = self._context_detector.detect_from_text(combined_text)

            if new_context != self._current_context and new_context != ContextType.GENERAL:
                old_context = self._current_context
                self._current_context = new_context
                self._current_prompt = load_prompt(new_context.value)
                logger.info(
                    "streaming: context updated from %s to %s",
                    old_context.value,
                    new_context.value,
                )

    def _transcribe_segment(self, segment: SpeechSegment) -> None:
        """Transcribe a single audio segment.

        Args:
            segment: Speech segment to transcribe.
        """
        # Get audio for this segment
        pending = self._buffer.get_pending()
        if pending.size == 0:
            return

        # Calculate overlap for context
        overlap_samples = int(self._config.overlap_ms * WHISPER_SAMPLE_RATE / 1000)

        # Calculate segment boundaries (before clamping for bounds check)
        start_raw = segment.start_sample - overlap_samples
        end_raw = segment.end_sample

        # Validate and recover from out-of-bounds segment coordinates
        if (end_raw > len(pending) or start_raw >= len(pending) or
            end_raw <= start_raw or start_raw < 0):
            logger.warning(
                "streaming: segment out of bounds, attempting recovery: start=%d, end=%d, pending_size=%d",
                start_raw, end_raw, len(pending)
            )

        # Extract segment audio (with overlap at start if not first segment)
        start = max(0, start_raw)
        end = min(len(pending), end_raw)

        if end <= start:
            logger.error("streaming: segment recovery failed, skipping")
            return

        segment_audio = pending[start:end]

        # Check minimum duration
        duration = len(segment_audio) / WHISPER_SAMPLE_RATE
        if duration < self._config.min_segment_duration:
            logger.debug("streaming: skipping short segment, duration=%.2fs", duration)
            return

        logger.debug("streaming: transcribing segment, duration=%.2fs", duration)

        # Transcribe with context-aware prompt
        try:
            start_time = time.perf_counter()
            result = self._transcriber.transcribe(
                segment_audio,
                initial_prompt=self._current_prompt,
            )
            inference_time = time.perf_counter() - start_time

            if result.text.strip():
                segment_text = result.text.strip()
                with self._lock:
                    self._segments.append(segment_text)
                    self._total_inference_time += inference_time

                # Mark audio as transcribed
                self._buffer.mark_transcribed(end)

                # Adjust VAD tracker for buffer shift (dedicated lock)
                with self._vad_lock:
                    self._vad_processed_samples = max(0, self._vad_processed_samples - end)

                # Update pending segments for buffer shift (main lock)
                with self._lock:
                    # Remove segments that are now completely transcribed
                    self._pending_segments = [
                        seg for seg in self._pending_segments
                        if seg.end_sample > end
                    ]

                    # Adjust remaining segments' coordinates for the shifted buffer
                    for seg in self._pending_segments:
                        seg.start_sample = max(0, seg.start_sample - end)
                        seg.end_sample = seg.end_sample - end

                # Update context based on accumulated text
                self._maybe_update_context(segment_text)

                # Notify callback
                if self._on_partial:
                    combined = " ".join(self._segments)
                    self._on_partial(combined)

                logger.debug(
                    "streaming: segment transcribed, text='%s', inference_ms=%d, context=%s",
                    result.text[:50],
                    int(inference_time * 1000),
                    self._current_context.value,
                )

        except Exception as e:
            logger.error("streaming: transcription error, error=%s", e)

    def finalize(self) -> StreamingResult:
        """Stop streaming and return final result.

        Transcribes any remaining audio and combines all segments.

        Returns:
            StreamingResult with combined transcription.
        """
        # Stop background thread
        self._stop_event.set()
        if self._transcription_thread is not None:
            self._transcription_thread.join(timeout=5.0)

        # Transcribe remaining audio
        remaining = self._buffer.get_pending()
        remaining_duration = len(remaining) / WHISPER_SAMPLE_RATE

        if remaining_duration >= 0.5:  # Only if at least 0.5s remaining
            logger.debug("streaming: transcribing remaining audio, duration=%.2fs", remaining_duration)
            try:
                start_time = time.perf_counter()
                result = self._transcriber.transcribe(
                    remaining,
                    initial_prompt=self._current_prompt,
                )
                inference_time = time.perf_counter() - start_time

                if result.text.strip():
                    with self._lock:
                        self._segments.append(result.text.strip())
                        self._total_inference_time += inference_time
            except Exception as e:
                logger.error("streaming: final transcription error, error=%s", e)

        # Combine all segments
        combined_text = " ".join(self._segments)
        total_duration = self._buffer.get_total_duration()

        logger.info(
            "streaming: finalized, segments=%d, total_duration=%.2fs, total_inference=%.2fs, context=%s, locked=%s",
            len(self._segments),
            total_duration,
            self._total_inference_time,
            self._current_context.value,
            self._context_locked,
        )

        return StreamingResult(
            text=combined_text,
            segments=self._segments.copy(),
            total_inference_time=self._total_inference_time,
            audio_duration=total_duration,
            num_segments=len(self._segments),
            detected_context=self._current_context,
            context_was_locked=self._context_locked,
        )


def get_streaming_config() -> StreamingConfig:
    """Load streaming configuration from app config.

    Returns:
        StreamingConfig populated from config.toml.
    """
    from localwispr.config import get_config

    config = get_config()
    streaming = config.get("streaming", {})

    return StreamingConfig(
        enabled=streaming.get("enabled", False),
        min_silence_ms=streaming.get("min_silence_ms", 800),
        max_segment_duration=streaming.get("max_segment_duration", 20.0),
        min_segment_duration=streaming.get("min_segment_duration", 2.0),
        overlap_ms=streaming.get("overlap_ms", 100),
        context_check_interval=streaming.get("context_check_interval", 3),
        context_lock_threshold=streaming.get("context_lock_threshold", 4),
        context_word_threshold=streaming.get("context_word_threshold", 50),
    )
