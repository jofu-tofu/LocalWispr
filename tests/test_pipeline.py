"""Tests for the recording pipeline module."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_pipeline_result_success(self):
        """Test creating a successful PipelineResult."""
        from localwispr.pipeline import PipelineResult

        result = PipelineResult(
            success=True,
            text="transcribed text",
            audio_duration=2.5,
            inference_time=0.5,
        )

        assert result.success is True
        assert result.text == "transcribed text"
        assert result.audio_duration == 2.5
        assert result.inference_time == 0.5
        assert result.error == ""
        assert result.was_retranscribed is False

    def test_pipeline_result_failure(self):
        """Test creating a failed PipelineResult."""
        from localwispr.pipeline import PipelineResult

        result = PipelineResult(
            success=False,
            error="Model load timeout",
        )

        assert result.success is False
        assert result.error == "Model load timeout"
        assert result.text == ""


class TestRecordingPipeline:
    """Tests for RecordingPipeline class."""

    def test_pipeline_initialization(self, mocker, reset_mode_manager):
        """Test pipeline initialization."""
        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)

        assert pipeline.is_recording is False
        assert pipeline.is_model_ready is False

    def test_pipeline_start_recording(self, mocker, reset_mode_manager):
        """Test starting recording."""
        # Mock AudioRecorder
        mock_recorder = MagicMock()
        mock_recorder.is_recording = False
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            return_value=mock_recorder,
        )

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)

        result = pipeline.start_recording()

        assert result is True
        mock_recorder.start_recording.assert_called_once()

    def test_pipeline_start_recording_already_recording(self, mocker, reset_mode_manager):
        """Test starting recording when already recording."""
        mock_recorder = MagicMock()
        mock_recorder.is_recording = True
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            return_value=mock_recorder,
        )

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)
        pipeline._recorder = mock_recorder

        result = pipeline.start_recording()

        assert result is False

    def test_pipeline_is_recording_property(self, mocker, reset_mode_manager):
        """Test is_recording property."""
        mock_recorder = MagicMock()
        mock_recorder.is_recording = True
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            return_value=mock_recorder,
        )

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)
        pipeline._recorder = mock_recorder

        assert pipeline.is_recording is True

        mock_recorder.is_recording = False
        assert pipeline.is_recording is False

    def test_pipeline_preload_model_async(self, mocker, reset_mode_manager):
        """Test async model preloading."""
        mock_transcriber = MagicMock()
        mock_transcriber.model = MagicMock()
        mocker.patch(
            "localwispr.transcribe.WhisperTranscriber",
            return_value=mock_transcriber,
        )

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)

        pipeline.preload_model_async()

        # Wait for preload to complete
        pipeline._model_preload_complete.wait(timeout=2.0)

        assert pipeline.is_model_ready is True

    def test_pipeline_stop_and_transcribe_no_audio(self, mocker, reset_mode_manager):
        """Test stop_and_transcribe with no audio captured."""
        mock_recorder = MagicMock()
        mock_recorder.is_recording = False
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            return_value=mock_recorder,
        )

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)
        pipeline._recorder = mock_recorder

        result = pipeline.stop_and_transcribe()

        assert result.success is False
        assert "No audio" in result.error or "not recording" in result.error.lower()

    def test_pipeline_stop_and_transcribe_success(self, mocker, reset_mode_manager):
        """Test successful stop_and_transcribe."""
        # Mock recorder
        mock_recorder = MagicMock()
        mock_recorder.is_recording = True
        mock_recorder.get_whisper_audio.return_value = np.zeros(16000, dtype=np.float32)
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            return_value=mock_recorder,
        )

        # Mock transcriber
        mock_result = MagicMock()
        mock_result.text = "test transcription"
        mock_result.audio_duration = 1.0
        mock_result.inference_time = 0.5
        mock_result.was_retranscribed = False

        mock_transcriber = MagicMock()
        mock_transcriber.model = MagicMock()
        mock_transcriber.transcribe.return_value = mock_result
        mocker.patch(
            "localwispr.transcribe.WhisperTranscriber",
            return_value=mock_transcriber,
        )

        # Mock get_mode_prompt
        mocker.patch(
            "localwispr.modes.get_mode_prompt",
            return_value="test prompt",
        )

        from localwispr.modes import ModeManager, ModeType
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        mode_manager.set_mode(ModeType.CODE)  # Manual override

        pipeline = RecordingPipeline(mode_manager=mode_manager)
        pipeline._recorder = mock_recorder
        pipeline._model_preload_complete.set()

        result = pipeline.stop_and_transcribe()

        assert result.success is True
        assert result.text == "test transcription"

    def test_pipeline_cancel_recording(self, mocker, reset_mode_manager):
        """Test canceling a recording."""
        mock_recorder = MagicMock()
        mock_recorder.is_recording = True
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            return_value=mock_recorder,
        )

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)
        pipeline._recorder = mock_recorder

        pipeline.cancel_recording()

        mock_recorder.stop_recording.assert_called_once()

    def test_pipeline_get_rms_level(self, mocker, reset_mode_manager):
        """Test getting RMS level during recording."""
        mock_recorder = MagicMock()
        mock_recorder.is_recording = True
        mock_recorder.get_rms_level.return_value = 0.75
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            return_value=mock_recorder,
        )

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)
        pipeline._recorder = mock_recorder

        level = pipeline.get_rms_level()

        assert level == 0.75

    def test_pipeline_get_rms_level_not_recording(self, mocker, reset_mode_manager):
        """Test getting RMS level when not recording."""
        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)

        level = pipeline.get_rms_level()

        assert level == 0.0

    def test_pipeline_model_timeout(self, mocker, reset_mode_manager):
        """Test pipeline handles model load timeout."""
        mock_recorder = MagicMock()
        mock_recorder.is_recording = True
        mock_recorder.get_whisper_audio.return_value = np.zeros(16000, dtype=np.float32)
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            return_value=mock_recorder,
        )

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)
        pipeline._recorder = mock_recorder
        pipeline.MODEL_LOAD_TIMEOUT = 0.01  # Very short timeout

        result = pipeline.stop_and_transcribe()

        assert result.success is False
        assert "timeout" in result.error.lower()

    def test_pipeline_on_error_callback(self, mocker, reset_mode_manager):
        """Test that on_error callback is invoked on failure."""
        mock_recorder = MagicMock()
        mock_recorder.is_recording = False
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            side_effect=Exception("device error"),
        )

        error_callback = MagicMock()

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager, on_error=error_callback)

        result = pipeline.start_recording()

        assert result is False
        error_callback.assert_called_once()

    def test_pipeline_thread_safety(self, mocker, reset_mode_manager):
        """Test pipeline operations are thread-safe."""
        mock_recorder = MagicMock()
        mock_recorder.is_recording = False
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            return_value=mock_recorder,
        )

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)

        results = []

        def start():
            results.append(pipeline.start_recording())

        threads = [threading.Thread(target=start) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only one should succeed (first one)
        # The rest should either succeed or fail due to "already recording"
        # But no exceptions should occur
        assert len(results) == 5

    def test_pipeline_mute_system_audio(self, mocker, reset_mode_manager):
        """Test system audio muting during recording."""
        mock_recorder = MagicMock()
        mock_recorder.is_recording = False
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            return_value=mock_recorder,
        )

        mock_mute = mocker.patch(
            "localwispr.volume.mute_system",
            return_value=True,
        )

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)

        pipeline.start_recording(mute_system=True)

        mock_mute.assert_called_once()
