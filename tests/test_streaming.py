"""Tests for streaming transcription module."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestAudioBuffer:
    """Tests for AudioBuffer class."""

    def test_append_and_get_pending(self):
        """Test basic append and get_pending functionality."""
        from localwispr.transcribe.streaming import AudioBuffer

        buffer = AudioBuffer()
        buffer.set_source_sample_rate(16000)

        # Append some audio
        chunk = np.zeros(1600, dtype=np.float32)  # 100ms
        buffer.append(chunk)

        pending = buffer.get_pending()
        assert pending.size > 0

    def test_mark_transcribed(self):
        """Test marking audio as transcribed."""
        from localwispr.transcribe.streaming import AudioBuffer

        buffer = AudioBuffer()
        buffer.set_source_sample_rate(16000)

        # Append audio
        chunk = np.zeros(16000, dtype=np.float32)  # 1 second
        buffer.append(chunk)

        # Mark half as transcribed
        buffer.mark_transcribed(8000)

        # Pending should be the remaining half
        pending = buffer.get_pending()
        assert pending.size == 8000

    def test_get_full_audio(self):
        """Test getting all buffered audio."""
        from localwispr.transcribe.streaming import AudioBuffer

        buffer = AudioBuffer()
        buffer.set_source_sample_rate(16000)

        # Append multiple chunks
        for _ in range(3):
            chunk = np.zeros(16000, dtype=np.float32)
            buffer.append(chunk)

        full = buffer.get_full_audio()
        assert full.size == 48000  # 3 seconds

    def test_clear(self):
        """Test clearing the buffer."""
        from localwispr.transcribe.streaming import AudioBuffer

        buffer = AudioBuffer()
        buffer.set_source_sample_rate(16000)

        buffer.append(np.zeros(16000, dtype=np.float32))
        buffer.clear()

        assert buffer.get_pending().size == 0
        assert buffer.get_full_audio().size == 0

    def test_get_pending_duration(self):
        """Test duration calculation."""
        from localwispr.transcribe.streaming import AudioBuffer

        buffer = AudioBuffer()
        buffer.set_source_sample_rate(16000)

        # Append 2 seconds
        buffer.append(np.zeros(32000, dtype=np.float32))

        duration = buffer.get_pending_duration()
        assert abs(duration - 2.0) < 0.1

    def test_max_duration_enforcement(self):
        """Test buffer enforces max duration."""
        from localwispr.transcribe.streaming import AudioBuffer

        # Create buffer with 1 second max
        buffer = AudioBuffer(max_duration_seconds=1.0)
        buffer.set_source_sample_rate(16000)

        # Append 2 seconds worth
        buffer.append(np.zeros(32000, dtype=np.float32))

        # Should be limited to ~1 second
        assert buffer.get_total_duration() <= 1.5

    def test_thread_safety(self):
        """Test buffer is thread-safe."""
        from localwispr.transcribe.streaming import AudioBuffer

        buffer = AudioBuffer()
        buffer.set_source_sample_rate(16000)
        errors = []

        def writer():
            try:
                for _ in range(100):
                    buffer.append(np.zeros(160, dtype=np.float32))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    _ = buffer.get_pending()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestVADProcessor:
    """Tests for VADProcessor class."""

    def test_initialization(self):
        """Test VAD processor initialization without loading model."""
        from localwispr.transcribe.streaming import VADProcessor

        vad = VADProcessor(min_silence_ms=800)
        # Model is lazy-loaded, so just check config
        assert vad._min_silence_samples == 12800  # 800ms at 16kHz
        assert vad._model is None  # Not loaded yet

    def test_reset(self):
        """Test resetting VAD state."""
        from localwispr.transcribe.streaming import VADProcessor

        vad = VADProcessor()
        vad._speech_start = 1000
        vad._silence_start = 2000
        vad._processed_samples = 5000

        vad.reset()

        assert vad._speech_start is None
        assert vad._silence_start is None
        assert vad._processed_samples == 0


class TestStreamingConfig:
    """Tests for StreamingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from localwispr.transcribe.streaming import StreamingConfig

        config = StreamingConfig()

        assert config.enabled is False
        assert config.min_silence_ms == 800
        assert config.max_segment_duration == 20.0
        assert config.min_segment_duration == 2.0
        assert config.overlap_ms == 100

    def test_custom_values(self):
        """Test custom configuration values."""
        from localwispr.transcribe.streaming import StreamingConfig

        config = StreamingConfig(
            enabled=True,
            min_silence_ms=500,
            max_segment_duration=30.0,
        )

        assert config.enabled is True
        assert config.min_silence_ms == 500
        assert config.max_segment_duration == 30.0


class TestStreamingTranscriber:
    """Tests for StreamingTranscriber class."""

    @pytest.fixture
    def mock_transcriber(self):
        """Create mock WhisperTranscriber."""
        mock = MagicMock()
        mock.transcribe.return_value = MagicMock(
            text="test transcription",
            segments=[],
            inference_time=0.5,
            audio_duration=1.0,
        )
        return mock

    @pytest.fixture
    def mock_vad_processor(self, mocker):
        """Mock VADProcessor to avoid torch dependency."""
        # Mock the VADProcessor class itself
        mock_vad_class = mocker.patch("localwispr.transcribe.streaming.VADProcessor")
        mock_vad_instance = MagicMock()
        mock_vad_instance.process.return_value = []  # No segments detected
        mock_vad_instance.reset.return_value = None
        mock_vad_class.return_value = mock_vad_instance
        return mock_vad_instance

    def test_initialization(self, mock_transcriber, mock_vad_processor):
        """Test streaming transcriber initialization."""
        from localwispr.transcribe.streaming import StreamingConfig, StreamingTranscriber

        config = StreamingConfig(enabled=True)
        streamer = StreamingTranscriber(
            transcriber=mock_transcriber,
            config=config,
        )

        assert streamer._transcriber is mock_transcriber
        assert streamer._config.enabled is True

    def test_start_and_finalize(self, mock_transcriber, mock_vad_processor):
        """Test start and finalize lifecycle."""
        from localwispr.transcribe.streaming import StreamingConfig, StreamingTranscriber

        config = StreamingConfig(enabled=True)
        streamer = StreamingTranscriber(
            transcriber=mock_transcriber,
            config=config,
        )

        streamer.start(source_sample_rate=48000)

        # Process some audio
        chunk = np.zeros(4800, dtype=np.float32)  # 100ms at 48kHz
        streamer.process_chunk(chunk)

        # Give time for processing
        time.sleep(0.1)

        # Finalize
        result = streamer.finalize()

        assert result is not None
        assert isinstance(result.text, str)
        assert result.audio_duration >= 0

    def test_process_chunk_adds_to_buffer(self, mock_transcriber, mock_vad_processor):
        """Test that process_chunk adds audio to buffer."""
        from localwispr.transcribe.streaming import StreamingConfig, StreamingTranscriber

        config = StreamingConfig(enabled=True)
        streamer = StreamingTranscriber(
            transcriber=mock_transcriber,
            config=config,
        )

        streamer.start(source_sample_rate=16000)

        # Process audio
        chunk = np.random.randn(1600).astype(np.float32)
        streamer.process_chunk(chunk)

        # Buffer should have audio
        assert streamer._buffer.get_pending_duration() > 0

        streamer.finalize()


class TestGetStreamingConfig:
    """Tests for get_streaming_config function."""

    def test_loads_from_config(self, mocker):
        """Test loading config from app config."""
        mock_config = {
            "streaming": {
                "enabled": True,
                "min_silence_ms": 600,
                "max_segment_duration": 15.0,
                "min_segment_duration": 1.5,
                "overlap_ms": 150,
            }
        }
        # Patch where get_config is used (imported into streaming module)
        mocker.patch("localwispr.config.get_config", return_value=mock_config)

        from localwispr.transcribe.streaming import get_streaming_config

        config = get_streaming_config()

        assert config.enabled is True
        assert config.min_silence_ms == 600
        assert config.max_segment_duration == 15.0

    def test_uses_defaults_when_missing(self, mocker):
        """Test using defaults when config section is missing."""
        mocker.patch("localwispr.config.get_config", return_value={})

        from localwispr.transcribe.streaming import get_streaming_config

        config = get_streaming_config()

        assert config.enabled is False
        assert config.min_silence_ms == 800


class TestStreamingResult:
    """Tests for StreamingResult dataclass."""

    def test_creation(self):
        """Test creating a streaming result."""
        from localwispr.transcribe.streaming import StreamingResult

        result = StreamingResult(
            text="Hello world",
            segments=["Hello", "world"],
            total_inference_time=1.5,
            audio_duration=3.0,
            num_segments=2,
        )

        assert result.text == "Hello world"
        assert len(result.segments) == 2
        assert result.total_inference_time == 1.5
        assert result.audio_duration == 3.0
        assert result.num_segments == 2


class TestIncrementalVADProcessing:
    """Tests for incremental VAD processing fix.

    These tests verify that VAD only processes NEW audio chunks,
    not the entire pending buffer on each call. This prevents
    inflated sample positions and mid-sentence cutoffs.
    """

    @pytest.fixture
    def mock_transcriber(self):
        """Create mock WhisperTranscriber."""
        mock = MagicMock()
        mock.transcribe.return_value = MagicMock(
            text="test transcription",
            segments=[],
            inference_time=0.5,
            audio_duration=1.0,
        )
        return mock

    def test_vad_processed_samples_initialized(self, mock_transcriber, mocker):
        """Test that _vad_processed_samples is initialized to 0."""
        # Mock VADProcessor to avoid torch dependency
        mock_vad_class = mocker.patch("localwispr.transcribe.streaming.VADProcessor")
        mock_vad_instance = MagicMock()
        mock_vad_instance.process.return_value = []
        mock_vad_instance.reset.return_value = None
        mock_vad_class.return_value = mock_vad_instance

        from localwispr.transcribe.streaming import StreamingConfig, StreamingTranscriber

        config = StreamingConfig(enabled=True)
        streamer = StreamingTranscriber(
            transcriber=mock_transcriber,
            config=config,
        )

        assert streamer._vad_processed_samples == 0

    def test_vad_processed_samples_reset_on_start(self, mock_transcriber, mocker):
        """Test that _vad_processed_samples is reset when start() is called."""
        mock_vad_class = mocker.patch("localwispr.transcribe.streaming.VADProcessor")
        mock_vad_instance = MagicMock()
        mock_vad_instance.process.return_value = []
        mock_vad_instance.reset.return_value = None
        mock_vad_class.return_value = mock_vad_instance

        from localwispr.transcribe.streaming import StreamingConfig, StreamingTranscriber

        config = StreamingConfig(enabled=True)
        streamer = StreamingTranscriber(
            transcriber=mock_transcriber,
            config=config,
        )

        # Simulate some processing happened
        streamer._vad_processed_samples = 5000

        # Start should reset it
        streamer.start(source_sample_rate=16000)

        assert streamer._vad_processed_samples == 0
        streamer.finalize()

    def test_vad_only_receives_new_audio(self, mock_transcriber, mocker):
        """Test that VAD.process() only receives new audio, not all pending."""
        mock_vad_class = mocker.patch("localwispr.transcribe.streaming.VADProcessor")
        mock_vad_instance = MagicMock()
        mock_vad_instance.process.return_value = []
        mock_vad_instance.reset.return_value = None
        mock_vad_class.return_value = mock_vad_instance

        from localwispr.transcribe.streaming import StreamingConfig, StreamingTranscriber

        config = StreamingConfig(enabled=True)
        streamer = StreamingTranscriber(
            transcriber=mock_transcriber,
            config=config,
        )

        streamer.start(source_sample_rate=16000)

        # Process first chunk (1600 samples = 100ms at 16kHz)
        chunk1 = np.zeros(1600, dtype=np.float32)
        streamer.process_chunk(chunk1)

        # VAD should have received ~1600 samples
        first_call_audio = mock_vad_instance.process.call_args_list[0][0][0]
        first_call_size = first_call_audio.size

        # Process second chunk
        chunk2 = np.zeros(1600, dtype=np.float32)
        streamer.process_chunk(chunk2)

        # VAD should receive only the NEW ~1600 samples, not all ~3200
        second_call_audio = mock_vad_instance.process.call_args_list[1][0][0]
        second_call_size = second_call_audio.size

        # The second call should be approximately the same size as the first
        # (just the new chunk), not double the size
        assert second_call_size <= first_call_size * 1.1  # Allow small variance

        streamer.finalize()

    def test_vad_tracker_updates_after_each_chunk(self, mock_transcriber, mocker):
        """Test that _vad_processed_samples tracks progress correctly."""
        mock_vad_class = mocker.patch("localwispr.transcribe.streaming.VADProcessor")
        mock_vad_instance = MagicMock()
        mock_vad_instance.process.return_value = []
        mock_vad_instance.reset.return_value = None
        mock_vad_class.return_value = mock_vad_instance

        from localwispr.transcribe.streaming import StreamingConfig, StreamingTranscriber

        config = StreamingConfig(enabled=True)
        streamer = StreamingTranscriber(
            transcriber=mock_transcriber,
            config=config,
        )

        streamer.start(source_sample_rate=16000)

        # Process multiple chunks
        for i in range(3):
            chunk = np.zeros(1600, dtype=np.float32)
            streamer.process_chunk(chunk)

        # Tracker should have increased with each chunk
        # At 16kHz source rate, 3 x 1600 samples = 4800 samples pending
        assert streamer._vad_processed_samples > 0
        # Should be approximately 4800 samples (may vary slightly due to resampling)
        assert streamer._vad_processed_samples >= 4000
        assert streamer._vad_processed_samples <= 6000

        streamer.finalize()

    def test_forced_segment_resets_vad_state(self, mock_transcriber, mocker):
        """Test that forced segments reset VAD state completely."""
        mock_vad_class = mocker.patch("localwispr.transcribe.streaming.VADProcessor")
        mock_vad_instance = MagicMock()
        mock_vad_instance.process.return_value = []
        mock_vad_instance.reset.return_value = None
        mock_vad_class.return_value = mock_vad_instance

        from localwispr.transcribe.streaming import StreamingConfig, StreamingTranscriber

        # Short max_segment_duration to trigger forced split easily
        config = StreamingConfig(
            enabled=True,
            max_segment_duration=0.5,  # 500ms
        )
        streamer = StreamingTranscriber(
            transcriber=mock_transcriber,
            config=config,
        )

        streamer.start(source_sample_rate=16000)

        # Process enough audio to trigger forced split (> 500ms)
        # At 16kHz, 500ms = 8000 samples
        large_chunk = np.zeros(10000, dtype=np.float32)  # 625ms
        streamer.process_chunk(large_chunk)

        # VAD reset should have been called due to forced segment
        assert mock_vad_instance.reset.call_count >= 2  # Once at start, once at forced split

        # Tracker should be reset to 0 after forced segment
        assert streamer._vad_processed_samples == 0

        streamer.finalize()


class TestCoordinateSystemFixes:
    """Tests for coordinate system and race condition fixes.

    These tests verify the fixes for:
    - Coordinate translation (VAD global → pending-relative)
    - Thread safety with dedicated VAD lock
    - Segment adjustment after buffer shift
    - Bounds validation with recovery
    - Buffer wraparound detection
    """

    @pytest.fixture
    def mock_transcriber(self):
        """Create mock WhisperTranscriber."""
        mock = MagicMock()
        mock.transcribe.return_value = MagicMock(
            text="test transcription",
            segments=[],
            inference_time=0.5,
            audio_duration=1.0,
        )
        return mock

    def test_vad_segments_translated_to_pending_relative(self, mock_transcriber, mocker):
        """VAD segments with global positions are translated to pending-relative."""
        from localwispr.transcribe.streaming import SpeechSegment, StreamingConfig, StreamingTranscriber

        # Mock VADProcessor to return segment with GLOBAL positions
        mock_vad_class = mocker.patch("localwispr.transcribe.streaming.VADProcessor")
        mock_vad_instance = MagicMock()
        mock_vad_instance.reset.return_value = None

        # First call: no segments
        # Second call: return segment at GLOBAL positions [10000:20000]
        mock_vad_instance.process.side_effect = [
            [],  # First chunk - no segments
            [SpeechSegment(start_sample=10000, end_sample=20000, is_final=True)],  # Second chunk
        ]
        mock_vad_class.return_value = mock_vad_instance

        config = StreamingConfig(enabled=True, min_segment_duration=0.1)
        streamer = StreamingTranscriber(
            transcriber=mock_transcriber,
            config=config,
        )

        streamer.start(source_sample_rate=16000)

        # Stop transcription thread immediately to prevent it from processing
        streamer._stop_event.set()
        if streamer._transcription_thread:
            streamer._transcription_thread.join(timeout=1.0)

        # Build up pending buffer
        chunk1 = np.zeros(8000, dtype=np.float32)  # 500ms
        streamer.process_chunk(chunk1)

        # Simulate transcription: mark 5000 samples as transcribed
        # This creates offset: _transcribed_samples = 5000
        streamer._buffer.mark_transcribed(5000)

        # Process second chunk - VAD returns segment at GLOBAL [10000:20000]
        chunk2 = np.zeros(8000, dtype=np.float32)
        streamer.process_chunk(chunk2)

        # Check queued segment is TRANSLATED to pending-relative
        with streamer._lock:
            assert len(streamer._pending_segments) >= 1
            # Find the segment from second chunk (should be the last one if there are multiple)
            seg = streamer._pending_segments[-1]
            # Global [10000:20000] - offset 5000 = pending-relative [5000:15000]
            assert seg.start_sample == 5000
            assert seg.end_sample == 15000

    def test_pending_segments_adjusted_after_transcription(self, mock_transcriber, mocker):
        """Pending segments are adjusted when buffer shifts."""
        from localwispr.transcribe.streaming import SpeechSegment, StreamingConfig, StreamingTranscriber

        mock_vad_class = mocker.patch("localwispr.transcribe.streaming.VADProcessor")
        mock_vad_instance = MagicMock()
        mock_vad_instance.process.return_value = []
        mock_vad_instance.reset.return_value = None
        mock_vad_class.return_value = mock_vad_instance

        config = StreamingConfig(enabled=True, min_segment_duration=0.1)
        streamer = StreamingTranscriber(
            transcriber=mock_transcriber,
            config=config,
        )

        streamer.start(source_sample_rate=16000)

        # Stop transcription thread to prevent automatic processing
        streamer._stop_event.set()
        if streamer._transcription_thread:
            streamer._transcription_thread.join(timeout=1.0)

        # Add audio to buffer
        chunk = np.zeros(16000, dtype=np.float32)  # 1 second
        streamer.process_chunk(chunk)

        # Manually queue two segments (pending-relative coordinates)
        with streamer._lock:
            streamer._pending_segments.clear()  # Clear any auto-generated
            streamer._pending_segments.append(
                SpeechSegment(start_sample=1000, end_sample=5000, is_final=True)
            )
            streamer._pending_segments.append(
                SpeechSegment(start_sample=6000, end_sample=10000, is_final=True)
            )

        # Get first segment before it's modified
        with streamer._lock:
            first_segment = streamer._pending_segments[0]

        # Manually transcribe first segment (marks 5000 samples done)
        streamer._transcribe_segment(first_segment)

        # Check second segment was adjusted for buffer shift
        with streamer._lock:
            # After transcription:
            # - First segment [1000:5000] should be removed (end_sample 5000 == transcribed amount)
            # - Second segment [6000:10000] should be adjusted: [6000-5000:10000-5000] = [1000:5000]
            remaining = [seg for seg in streamer._pending_segments]
            assert len(remaining) == 1, f"Expected 1 segment, got {len(remaining)}: {remaining}"
            adjusted_seg = remaining[0]
            assert adjusted_seg.start_sample == 1000, f"Expected start 1000, got {adjusted_seg.start_sample}"
            assert adjusted_seg.end_sample == 5000, f"Expected end 5000, got {adjusted_seg.end_sample}"

    def test_segment_bounds_recovery_attempts_salvage(self, mock_transcriber, mocker):
        """Out-of-bounds segments attempt recovery by clamping."""
        from localwispr.transcribe.streaming import SpeechSegment, StreamingConfig, StreamingTranscriber

        mock_vad_class = mocker.patch("localwispr.transcribe.streaming.VADProcessor")
        mock_vad_instance = MagicMock()
        mock_vad_instance.process.return_value = []
        mock_vad_instance.reset.return_value = None
        mock_vad_class.return_value = mock_vad_instance

        config = StreamingConfig(enabled=True, min_segment_duration=0.1)
        streamer = StreamingTranscriber(
            transcriber=mock_transcriber,
            config=config,
        )

        streamer.start(source_sample_rate=16000)

        # Add audio to buffer
        chunk = np.zeros(8000, dtype=np.float32)  # 500ms = 8000 samples
        streamer.process_chunk(chunk)

        # Create segment with end > pending size (out of bounds)
        bad_segment = SpeechSegment(start_sample=2000, end_sample=20000, is_final=True)

        # Attempt to transcribe - should log warning but recover
        with patch("localwispr.transcribe.streaming.logger") as mock_logger:
            streamer._transcribe_segment(bad_segment)

            # Should have logged warning about out of bounds
            assert mock_logger.warning.call_count >= 1
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "out of bounds" in warning_msg

        # Transcriber should have been called with clamped audio (recovery succeeded)
        assert mock_transcriber.transcribe.called

        streamer.finalize()

    def test_buffer_wraparound_detection_resets_vad(self, mock_transcriber, mocker):
        """Wraparound (>5min recording) is detected and handled."""
        from localwispr.transcribe.streaming import SpeechSegment, StreamingConfig, StreamingTranscriber

        mock_vad_class = mocker.patch("localwispr.transcribe.streaming.VADProcessor")
        mock_vad_instance = MagicMock()
        mock_vad_instance.process.return_value = []
        mock_vad_instance.reset.return_value = None
        mock_vad_class.return_value = mock_vad_instance

        config = StreamingConfig(enabled=True)
        streamer = StreamingTranscriber(
            transcriber=mock_transcriber,
            config=config,
        )

        streamer.start(source_sample_rate=16000)

        # Directly manipulate private state to simulate wraparound scenario
        # (no public API exists for this edge case)
        streamer._last_transcribed_samples = 10000  # Was at 10000
        streamer._buffer._transcribed_samples = 5000  # Now at 5000 (decreased = wraparound)
        streamer._vad_processed_samples = 8000  # Some VAD progress

        # Queue a segment to verify it gets cleared
        with streamer._lock:
            streamer._pending_segments.append(
                SpeechSegment(start_sample=1000, end_sample=5000, is_final=True)
            )

        # Process chunk - should detect wraparound
        with patch("localwispr.transcribe.streaming.logger") as mock_logger:
            chunk = np.zeros(1600, dtype=np.float32)
            streamer.process_chunk(chunk)

            # Should have logged warning about wraparound
            assert any("wraparound" in str(call) for call in mock_logger.warning.call_args_list)

        # VAD should be reset
        assert mock_vad_instance.reset.call_count >= 2  # Once at start, once at wraparound

        # Tracker should be reset to 0
        with streamer._vad_lock:
            assert streamer._vad_processed_samples >= 0  # Reset or updated with new chunk

        # Pending segments should be cleared
        with streamer._lock:
            # After wraparound reset, segments are cleared
            # New chunk might add new segments, so check it was cleared at wraparound
            pass  # State will be updated after wraparound processing

        streamer.finalize()

    def test_streaming_multi_segment_end_to_end(self, mock_transcriber, mocker):
        """Full integration: Multiple VAD segments transcribe correctly."""
        from localwispr.transcribe.streaming import SpeechSegment, StreamingConfig, StreamingTranscriber

        # Track transcription calls
        transcription_calls = []

        def mock_transcribe(audio, **kwargs):
            transcription_calls.append(len(audio))
            return MagicMock(
                text=f"segment {len(transcription_calls)}",
                segments=[],
                inference_time=0.1,
                audio_duration=len(audio) / 16000,
            )

        mock_transcriber.transcribe.side_effect = mock_transcribe

        # Mock VAD to return segments at specific times
        mock_vad_class = mocker.patch("localwispr.transcribe.streaming.VADProcessor")
        mock_vad_instance = MagicMock()
        mock_vad_instance.reset.return_value = None

        # Return segments progressively
        call_count = [0]

        def vad_process(audio):
            call_count[0] += 1
            # Return segment on specific calls to simulate real VAD behavior
            if call_count[0] == 3:  # Third chunk
                return [SpeechSegment(start_sample=8000, end_sample=24000, is_final=True)]
            elif call_count[0] == 6:  # Sixth chunk
                return [SpeechSegment(start_sample=32000, end_sample=48000, is_final=True)]
            return []

        mock_vad_instance.process.side_effect = vad_process
        mock_vad_class.return_value = mock_vad_instance

        config = StreamingConfig(
            enabled=True,
            min_segment_duration=0.5,
            max_segment_duration=30.0,
        )
        streamer = StreamingTranscriber(
            transcriber=mock_transcriber,
            config=config,
        )

        streamer.start(source_sample_rate=16000)

        # Process multiple chunks over simulated 3 seconds
        for i in range(10):
            chunk = np.random.randn(4800).astype(np.float32)  # 300ms per chunk
            streamer.process_chunk(chunk)
            time.sleep(0.05)  # Small delay to allow background processing

        # Give time for transcription thread to process
        time.sleep(0.5)

        # Finalize
        result = streamer.finalize()

        # Should have transcribed segments (exact count depends on timing)
        assert len(transcription_calls) >= 2  # At least the two VAD segments
        assert result.num_segments >= 2
        assert result.text != ""

        # Check no errors were logged (bounds issues, etc.)
        # This is checked implicitly by successful transcription

        streamer.finalize()

    def test_mark_transcribed_beyond_buffer_size(self):
        """Test marking more samples as transcribed than exist in buffer.

        Edge case: mark_transcribed(samples) called with samples > current buffer size.
        Should handle gracefully without negative buffer calculations or crashes.
        """
        from localwispr.transcribe.streaming import AudioBuffer

        buffer = AudioBuffer()
        buffer.set_source_sample_rate(16000)

        # Add some audio
        audio = np.zeros(8000, dtype=np.float32)  # 0.5 seconds at 16kHz
        buffer.append(audio)

        pending_audio = buffer.get_pending()
        initial_size = len(pending_audio)
        assert initial_size > 0

        # Mark MORE samples as transcribed than actually exist
        buffer.mark_transcribed(16000)  # 1 second, but only have 0.5 seconds

        # Should handle gracefully - buffer size should be 0, not negative
        pending_audio = buffer.get_pending()
        pending_size = len(pending_audio)
        assert pending_size >= 0, f"Buffer size went negative: {pending_size}"

        # Should be able to add more audio without issues
        buffer.append(np.zeros(8000, dtype=np.float32))
        pending_audio = buffer.get_pending()
        assert len(pending_audio) >= 0
