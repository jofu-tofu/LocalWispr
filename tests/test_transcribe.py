"""Tests for the transcription module."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest


@dataclass
class MockSegment:
    """Mock Whisper transcription segment."""

    text: str
    start: float = 0.0
    end: float = 1.0


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_transcription_result_creation(self):
        """Test that TranscriptionResult can be created with required fields."""
        from localwispr.transcribe import TranscriptionResult

        result = TranscriptionResult(
            text="hello world",
            segments=[{"start": 0.0, "end": 1.0, "text": "hello world"}],
            inference_time=0.5,
            audio_duration=1.0,
        )

        assert result.text == "hello world"
        assert len(result.segments) == 1
        assert result.inference_time == 0.5
        assert result.audio_duration == 1.0
        assert result.detected_context is None
        assert result.was_retranscribed is False

    def test_transcription_result_with_context(self):
        """Test TranscriptionResult with context detection fields."""
        from localwispr.context import ContextType
        from localwispr.transcribe import TranscriptionResult

        result = TranscriptionResult(
            text="create a function",
            segments=[],
            inference_time=0.3,
            audio_duration=0.8,
            detected_context=ContextType.CODING,
            context_detection_time=0.01,
            was_retranscribed=True,
        )

        assert result.detected_context == ContextType.CODING
        assert result.context_detection_time == 0.01
        assert result.was_retranscribed is True


class TestWhisperTranscriber:
    """Tests for WhisperTranscriber class."""

    def test_transcriber_initialization_with_defaults(self, mocker, mock_config):
        """Test transcriber uses config defaults when no args provided."""
        mocker.patch("localwispr.transcribe.get_config", return_value=mock_config)

        from localwispr.transcribe import WhisperTranscriber

        transcriber = WhisperTranscriber()

        assert transcriber.model_name == "tiny"
        assert transcriber.device == "cpu"
        assert transcriber.compute_type == "int8"
        assert transcriber.language is None  # "auto" becomes None
        assert transcriber.is_loaded is False

    def test_transcriber_initialization_with_custom_args(self, mocker, mock_config):
        """Test transcriber uses provided arguments over config."""
        mocker.patch("localwispr.transcribe.get_config", return_value=mock_config)

        from localwispr.transcribe import WhisperTranscriber

        transcriber = WhisperTranscriber(
            model_name="large-v3",
            device="cuda",
            compute_type="float16",
        )

        assert transcriber.model_name == "large-v3"
        assert transcriber.device == "cuda"
        assert transcriber.compute_type == "float16"

    def test_transcriber_lazy_model_loading(self, mocker, mock_config):
        """Test that model is not loaded until accessed."""
        mocker.patch("localwispr.transcribe.get_config", return_value=mock_config)
        mock_whisper = mocker.patch("faster_whisper.WhisperModel")

        from localwispr.transcribe import WhisperTranscriber

        transcriber = WhisperTranscriber()

        # Model should not be loaded yet
        assert transcriber.is_loaded is False
        mock_whisper.assert_not_called()

        # Access model property triggers load
        _ = transcriber.model

        assert transcriber.is_loaded is True
        mock_whisper.assert_called_once_with(
            "tiny",
            device="cpu",
            compute_type="int8",
        )

    def test_transcriber_hotwords_from_config(self, mocker, mock_config):
        """Test that hotwords are loaded from config."""
        mocker.patch("localwispr.transcribe.get_config", return_value=mock_config)

        from localwispr.transcribe import WhisperTranscriber

        transcriber = WhisperTranscriber()

        assert transcriber.hotwords == ["LocalWispr", "pytest"]

    def test_transcriber_language_explicit(self, mocker, mock_config):
        """Test explicit language setting."""
        mock_config["model"]["language"] = "en"
        mocker.patch("localwispr.transcribe.get_config", return_value=mock_config)

        from localwispr.transcribe import WhisperTranscriber

        transcriber = WhisperTranscriber()

        assert transcriber.language == "en"

    def test_transcribe_returns_result(self, mocker, mock_config, mock_audio_data):
        """Test transcribe method returns TranscriptionResult."""
        mocker.patch("localwispr.transcribe.get_config", return_value=mock_config)

        # Mock model
        mock_model = MagicMock()
        segments = [MockSegment(text=" hello world", start=0.0, end=1.0)]
        mock_model.transcribe.return_value = (iter(segments), None)

        mock_whisper = mocker.patch("faster_whisper.WhisperModel")
        mock_whisper.return_value = mock_model

        from localwispr.transcribe import WhisperTranscriber

        transcriber = WhisperTranscriber()
        result = transcriber.transcribe(mock_audio_data)

        assert result.text == "hello world"
        assert len(result.segments) == 1
        assert result.audio_duration == 1.0  # 16000 samples / 16000 Hz
        assert result.inference_time > 0

    def test_transcribe_with_vad_filter(self, mocker, mock_config, mock_audio_data):
        """Test transcribe passes VAD filter setting."""
        mocker.patch("localwispr.transcribe.get_config", return_value=mock_config)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), None)
        mock_whisper = mocker.patch("faster_whisper.WhisperModel")
        mock_whisper.return_value = mock_model

        from localwispr.transcribe import WhisperTranscriber

        transcriber = WhisperTranscriber()
        transcriber.transcribe(mock_audio_data, vad_filter=False)

        # Check that vad_filter was passed
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["vad_filter"] is False

    def test_transcribe_with_beam_size(self, mocker, mock_config, mock_audio_data):
        """Test transcribe passes beam size setting."""
        mocker.patch("localwispr.transcribe.get_config", return_value=mock_config)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), None)
        mock_whisper = mocker.patch("faster_whisper.WhisperModel")
        mock_whisper.return_value = mock_model

        from localwispr.transcribe import WhisperTranscriber

        transcriber = WhisperTranscriber()
        transcriber.transcribe(mock_audio_data, beam_size=10)

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["beam_size"] == 10

    def test_transcribe_with_initial_prompt(self, mocker, mock_config, mock_audio_data):
        """Test transcribe passes initial prompt."""
        mocker.patch("localwispr.transcribe.get_config", return_value=mock_config)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), None)
        mock_whisper = mocker.patch("faster_whisper.WhisperModel")
        mock_whisper.return_value = mock_model

        from localwispr.transcribe import WhisperTranscriber

        transcriber = WhisperTranscriber()
        transcriber.transcribe(mock_audio_data, initial_prompt="coding context")

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["initial_prompt"] == "coding context"

    def test_transcribe_includes_hotwords(self, mocker, mock_config, mock_audio_data):
        """Test that hotwords are passed to model.transcribe."""
        mocker.patch("localwispr.transcribe.get_config", return_value=mock_config)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), None)
        mock_whisper = mocker.patch("faster_whisper.WhisperModel")
        mock_whisper.return_value = mock_model

        from localwispr.transcribe import WhisperTranscriber

        transcriber = WhisperTranscriber()
        transcriber.transcribe(mock_audio_data)

        call_kwargs = mock_model.transcribe.call_args[1]
        assert "hotwords" in call_kwargs
        assert "LocalWispr pytest" == call_kwargs["hotwords"]


class TestTranscribeWithContext:
    """Tests for transcribe_with_context function."""

    def test_transcribe_with_context_uses_window_detection(
        self, mocker, mock_config, mock_audio_data
    ):
        """Test that context detection uses window title."""
        mocker.patch("localwispr.transcribe.get_config", return_value=mock_config)

        # Mock window detection
        mock_detector = MagicMock()
        from localwispr.context import ContextType

        mock_detector.detect_from_window.return_value = ContextType.CODING
        mock_detector.detect_from_text.return_value = ContextType.CODING

        # Mock transcriber
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = MagicMock(
            text="test",
            segments=[],
            inference_time=0.1,
            audio_duration=1.0,
        )

        # Mock load_prompt
        mocker.patch("localwispr.transcribe.load_prompt", return_value="coding prompt")

        from localwispr.transcribe import transcribe_with_context

        result = transcribe_with_context(mock_audio_data, mock_transcriber, mock_detector)

        mock_detector.detect_from_window.assert_called_once()
        assert result.detected_context == ContextType.CODING

    def test_transcribe_with_context_retranscribes_on_mismatch(
        self, mocker, mock_config, mock_audio_data
    ):
        """Test retranscription when pre and post contexts differ."""
        mocker.patch("localwispr.transcribe.get_config", return_value=mock_config)

        from localwispr.context import ContextType

        mock_detector = MagicMock()
        mock_detector.detect_from_window.return_value = ContextType.GENERAL
        mock_detector.detect_from_text.return_value = ContextType.CODING

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = MagicMock(
            text="create a function",
            segments=[],
            inference_time=0.1,
            audio_duration=1.0,
        )

        mocker.patch("localwispr.transcribe.load_prompt", return_value="prompt")

        from localwispr.transcribe import transcribe_with_context

        result = transcribe_with_context(mock_audio_data, mock_transcriber, mock_detector)

        # Should call transcribe twice (initial + retranscribe)
        assert mock_transcriber.transcribe.call_count == 2
        assert result.was_retranscribed is True

    def test_transcribe_with_context_no_retranscribe_on_general_post(
        self, mocker, mock_config, mock_audio_data
    ):
        """Test no retranscription when post-detection is GENERAL."""
        mocker.patch("localwispr.transcribe.get_config", return_value=mock_config)

        from localwispr.context import ContextType

        mock_detector = MagicMock()
        mock_detector.detect_from_window.return_value = ContextType.CODING
        mock_detector.detect_from_text.return_value = ContextType.GENERAL

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = MagicMock(
            text="hello world",
            segments=[],
            inference_time=0.1,
            audio_duration=1.0,
        )

        mocker.patch("localwispr.transcribe.load_prompt", return_value="prompt")

        from localwispr.transcribe import transcribe_with_context

        result = transcribe_with_context(mock_audio_data, mock_transcriber, mock_detector)

        # Should only call transcribe once
        assert mock_transcriber.transcribe.call_count == 1
        assert result.was_retranscribed is False


class TestTranscribeRecording:
    """Tests for transcribe_recording convenience function."""

    def test_transcribe_recording_stops_recorder(self, mocker, mock_config, mock_audio_data):
        """Test that transcribe_recording stops the recorder and gets audio."""
        mocker.patch("localwispr.transcribe.get_config", return_value=mock_config)

        mock_recorder = MagicMock()
        mock_recorder.get_whisper_audio.return_value = mock_audio_data

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = MagicMock(
            text="test",
            segments=[],
            inference_time=0.1,
            audio_duration=1.0,
            detected_context=None,
            was_retranscribed=False,
        )

        from localwispr.transcribe import transcribe_recording

        result = transcribe_recording(
            mock_recorder, mock_transcriber, use_context=False
        )

        mock_recorder.get_whisper_audio.assert_called_once()
        assert result.text == "test"
