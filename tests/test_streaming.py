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
        from localwispr.streaming import AudioBuffer

        buffer = AudioBuffer()
        buffer.set_source_sample_rate(16000)

        # Append some audio
        chunk = np.zeros(1600, dtype=np.float32)  # 100ms
        buffer.append(chunk)

        pending = buffer.get_pending()
        assert pending.size > 0

    def test_mark_transcribed(self):
        """Test marking audio as transcribed."""
        from localwispr.streaming import AudioBuffer

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
        from localwispr.streaming import AudioBuffer

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
        from localwispr.streaming import AudioBuffer

        buffer = AudioBuffer()
        buffer.set_source_sample_rate(16000)

        buffer.append(np.zeros(16000, dtype=np.float32))
        buffer.clear()

        assert buffer.get_pending().size == 0
        assert buffer.get_full_audio().size == 0

    def test_get_pending_duration(self):
        """Test duration calculation."""
        from localwispr.streaming import AudioBuffer

        buffer = AudioBuffer()
        buffer.set_source_sample_rate(16000)

        # Append 2 seconds
        buffer.append(np.zeros(32000, dtype=np.float32))

        duration = buffer.get_pending_duration()
        assert abs(duration - 2.0) < 0.1

    def test_max_duration_enforcement(self):
        """Test buffer enforces max duration."""
        from localwispr.streaming import AudioBuffer

        # Create buffer with 1 second max
        buffer = AudioBuffer(max_duration_seconds=1.0)
        buffer.set_source_sample_rate(16000)

        # Append 2 seconds worth
        buffer.append(np.zeros(32000, dtype=np.float32))

        # Should be limited to ~1 second
        assert buffer.get_total_duration() <= 1.5

    def test_thread_safety(self):
        """Test buffer is thread-safe."""
        from localwispr.streaming import AudioBuffer

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
        from localwispr.streaming import VADProcessor

        vad = VADProcessor(min_silence_ms=800)
        # Model is lazy-loaded, so just check config
        assert vad._min_silence_samples == 12800  # 800ms at 16kHz
        assert vad._model is None  # Not loaded yet

    def test_reset(self):
        """Test resetting VAD state."""
        from localwispr.streaming import VADProcessor

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
        from localwispr.streaming import StreamingConfig

        config = StreamingConfig()

        assert config.enabled is False
        assert config.min_silence_ms == 800
        assert config.max_segment_duration == 20.0
        assert config.min_segment_duration == 2.0
        assert config.overlap_ms == 100

    def test_custom_values(self):
        """Test custom configuration values."""
        from localwispr.streaming import StreamingConfig

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
        mock_vad_class = mocker.patch("localwispr.streaming.VADProcessor")
        mock_vad_instance = MagicMock()
        mock_vad_instance.process.return_value = []  # No segments detected
        mock_vad_instance.reset.return_value = None
        mock_vad_class.return_value = mock_vad_instance
        return mock_vad_instance

    def test_initialization(self, mock_transcriber, mock_vad_processor):
        """Test streaming transcriber initialization."""
        from localwispr.streaming import StreamingConfig, StreamingTranscriber

        config = StreamingConfig(enabled=True)
        streamer = StreamingTranscriber(
            transcriber=mock_transcriber,
            config=config,
        )

        assert streamer._transcriber is mock_transcriber
        assert streamer._config.enabled is True

    def test_start_and_finalize(self, mock_transcriber, mock_vad_processor):
        """Test start and finalize lifecycle."""
        from localwispr.streaming import StreamingConfig, StreamingTranscriber

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
        from localwispr.streaming import StreamingConfig, StreamingTranscriber

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

        from localwispr.streaming import get_streaming_config

        config = get_streaming_config()

        assert config.enabled is True
        assert config.min_silence_ms == 600
        assert config.max_segment_duration == 15.0

    def test_uses_defaults_when_missing(self, mocker):
        """Test using defaults when config section is missing."""
        mocker.patch("localwispr.config.get_config", return_value={})

        from localwispr.streaming import get_streaming_config

        config = get_streaming_config()

        assert config.enabled is False
        assert config.min_silence_ms == 800


class TestStreamingResult:
    """Tests for StreamingResult dataclass."""

    def test_creation(self):
        """Test creating a streaming result."""
        from localwispr.streaming import StreamingResult

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
