"""Integration tests for LocalWispr workflows.

These tests verify complete workflows with minimal mocking, testing
the integration between multiple modules.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.helpers import MockSegment, MockTranscriptionInfo

# Detect which backend is active for test mocking
from localwispr.transcribe.transcriber import _FASTER_WHISPER_AVAILABLE

pytestmark = pytest.mark.integration


def _mock_whisper_backend(mocker, mock_model=None, segments=None, side_effect=None):
    """Mock the active whisper backend consistently.

    Args:
        mocker: pytest mocker fixture.
        mock_model: Optional pre-configured mock model.
        segments: Optional list of MockSegment for transcribe return.
        side_effect: Optional side_effect for the model class constructor.

    Returns:
        The mock model class.
    """
    if mock_model is None:
        mock_model = MagicMock()

    if segments is not None:
        if _FASTER_WHISPER_AVAILABLE:
            mock_model.transcribe.side_effect = lambda *a, **kw: (iter(segments), MockTranscriptionInfo())
        else:
            mock_model.transcribe.return_value = segments

    if _FASTER_WHISPER_AVAILABLE:
        if side_effect is not None:
            return mocker.patch("faster_whisper.WhisperModel", side_effect=side_effect)
        return mocker.patch("faster_whisper.WhisperModel", return_value=mock_model)
    else:
        if side_effect is not None:
            return mocker.patch("pywhispercpp.model.Model", side_effect=side_effect)
        return mocker.patch("pywhispercpp.model.Model", return_value=mock_model)


class TestContextDetectionWorkflow:
    """Tests for context detection integration."""

    def test_context_detection_with_coding_keywords(self, mocker, mock_config):
        """Test that coding keywords trigger context detection."""
        mocker.patch("localwispr.config.get_config", return_value=mock_config)

        from localwispr.transcribe.context import ContextDetector, ContextType

        detector = ContextDetector()

        # Text with coding keywords
        result = detector.detect_from_text("create a function with variables")

        assert result == ContextType.CODING

    def test_context_detection_with_planning_keywords(self, mocker, mock_config):
        """Test that planning keywords trigger context detection."""
        mocker.patch("localwispr.config.get_config", return_value=mock_config)

        from localwispr.transcribe.context import ContextDetector, ContextType

        detector = ContextDetector()

        # Text with planning keywords
        result = detector.detect_from_text("create a task for the project deadline")

        assert result == ContextType.PLANNING

    def test_context_aware_transcription_workflow(self, mocker, mock_config):
        """Test transcription with context detection."""
        mocker.patch("localwispr.config.get_config", return_value=mock_config)

        # Mock transcriber
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = MagicMock(
            text="create a function",
            segments=[],
            inference_time=0.1,
            audio_duration=1.0,
        )

        # Mock detector
        from localwispr.transcribe.context import ContextType

        mock_detector = MagicMock()
        mock_detector.detect_from_window.return_value = ContextType.GENERAL
        mock_detector.detect_from_text.return_value = ContextType.CODING

        # Mock load_prompt
        mocker.patch("localwispr.transcribe.transcriber.load_prompt", return_value="coding prompt")

        audio = np.zeros(16000, dtype=np.float32)

        from localwispr.transcribe.transcriber import transcribe_with_context

        result = transcribe_with_context(audio, mock_transcriber, mock_detector)

        # Should detect coding context from text
        assert result.detected_context == ContextType.CODING


class TestConfigurationWorkflow:
    """Tests for configuration loading and saving workflow."""

    def test_config_save_and_reload_workflow(self, tmp_path):
        """Test saving config and reloading it."""
        import localwispr.config as config_module

        config_path = tmp_path / "config.toml"

        # Clear cache
        config_module._cached_config = None

        # Create initial config
        initial_config = {
            "model": {
                "name": "tiny",
                "device": "cpu",
                "compute_type": "int8",
                "language": "auto",
            },
            "hotkeys": {
                "mode": "push-to-talk",
                "modifiers": ["ctrl"],
                "audio_feedback": False,
                "mute_system": False,
            },
            "context": {
                "coding_apps": [],
                "planning_apps": [],
                "coding_keywords": [],
                "planning_keywords": [],
            },
            "output": {
                "auto_paste": True,
                "paste_delay_ms": 100,
            },
            "vocabulary": {
                "words": ["test"],
            },
        }

        from localwispr.config import load_config, save_config

        # Save config
        save_config(initial_config, config_path)

        # Reload and verify
        loaded = load_config(config_path)

        assert loaded["model"]["name"] == "tiny"
        assert loaded["hotkeys"]["mode"] == "push-to-talk"
        assert loaded["output"]["paste_delay_ms"] == 100

    def test_config_partial_override_workflow(self, tmp_path):
        """Test that partial config properly overrides defaults."""
        config_path = tmp_path / "config.toml"

        # Write partial config
        config_path.write_text("""
[model]
name = "small"

[hotkeys]
audio_feedback = false
""")

        from localwispr.config import load_config

        loaded = load_config(config_path)

        # Custom values
        assert loaded["model"]["name"] == "small"
        assert loaded["hotkeys"]["audio_feedback"] is False

        # Default values preserved
        assert loaded["model"]["device"] == "auto"
        assert loaded["hotkeys"]["mode"] == "push-to-talk"


class TestModeManagementWorkflow:
    """Tests for mode management workflow."""

    def test_mode_cycle_workflow(self, reset_mode_manager):
        """Test cycling through all modes."""
        from localwispr.modes import (
            MODE_CYCLE_ORDER,
            ModeType,
            cycle_mode,
            get_current_mode,
            set_mode,
        )

        # Start at CODE
        set_mode(ModeType.CODE)
        assert get_current_mode().mode_type == ModeType.CODE

        # Cycle through all modes
        for i in range(len(MODE_CYCLE_ORDER)):
            new_mode = cycle_mode()
            expected_index = (MODE_CYCLE_ORDER.index(ModeType.CODE) + i + 1) % len(
                MODE_CYCLE_ORDER
            )
            assert new_mode.mode_type == MODE_CYCLE_ORDER[expected_index]

    def test_mode_with_callback_workflow(self, reset_mode_manager):
        """Test mode changes trigger callbacks."""
        callback_log = []

        from localwispr.modes import ModeManager, ModeType

        manager = ModeManager(
            on_mode_change=lambda mode: callback_log.append(mode.mode_type)
        )

        manager.set_mode(ModeType.CODE)
        manager.set_mode(ModeType.EMAIL)
        manager.set_mode(ModeType.CHAT)

        assert callback_log == [ModeType.CODE, ModeType.EMAIL, ModeType.CHAT]


class TestErrorHandlingWorkflow:
    """Tests for error handling workflows."""

    def test_pipeline_handles_recorder_error(self, mocker, reset_mode_manager):
        """Test pipeline gracefully handles recorder errors."""
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            side_effect=Exception("Device not found"),
        )

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)

        result = pipeline.start_recording()

        assert result is False

    def test_pipeline_handles_transcription_error(self, mocker, reset_mode_manager):
        """Test pipeline gracefully handles transcription errors."""
        mock_recorder = MagicMock()
        mock_recorder.is_recording = True
        mock_recorder.get_whisper_audio.return_value = np.zeros(16000, dtype=np.float32)
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            return_value=mock_recorder,
        )

        mock_transcriber = MagicMock()
        mock_transcriber.model = MagicMock()
        mock_transcriber.transcribe.side_effect = Exception("Transcription failed")
        mocker.patch(
            "localwispr.transcribe.transcriber.WhisperTranscriber",
            return_value=mock_transcriber,
        )

        from localwispr.modes import ModeManager, ModeType
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        mode_manager.set_mode(ModeType.DICTATION)

        pipeline = RecordingPipeline(mode_manager=mode_manager)
        pipeline._recorder = mock_recorder
        pipeline._model_preload_complete.set()

        result = pipeline.stop_and_transcribe()

        assert result.success is False

    def test_output_handles_clipboard_locked(self, mocker):
        """Test output handles clipboard being locked."""
        import pyperclip

        mock_pyperclip = mocker.patch("localwispr.output.pyperclip")
        mock_pyperclip.PyperclipException = pyperclip.PyperclipException
        mock_pyperclip.copy.side_effect = pyperclip.PyperclipException("Clipboard locked")

        mocker.patch("localwispr.output.time.sleep")

        from localwispr.output import output_transcription

        result = output_transcription("test", auto_paste=False)

        assert result is False


class TestAudioProcessingWorkflow:
    """Tests for audio processing workflow."""

    def test_audio_format_conversion_workflow(self):
        """Test audio conversion through the pipeline."""
        from localwispr.audio import prepare_for_whisper

        # Simulate 48kHz stereo audio
        stereo_48k = np.random.randn(48000, 2).astype(np.float32)

        # Convert to Whisper format
        whisper_audio = prepare_for_whisper(stereo_48k, 48000)

        # Should be 16kHz mono
        assert whisper_audio.ndim == 1
        assert len(whisper_audio) == 16000
        assert whisper_audio.dtype == np.float32
        assert np.max(np.abs(whisper_audio)) <= 1.0


class TestModelLoadingWorkflows:
    """Behavior-focused tests for model loading and transcription.

    These tests mock only system boundaries (pywhispercpp, sounddevice)
    and let internal code (is_model_downloaded, preload_model_async,
    WhisperTranscriber, RecordingPipeline) run for real.
    """

    def test_model_not_downloaded_returns_immediate_error(
        self, mocker, mock_config, models_in_tmp, reset_mode_manager, sync_executor,
    ):
        """User records with no model -> gets 'not downloaded' error in < 1 second."""
        mocker.patch("localwispr.config.get_config", return_value=mock_config)
        _mock_whisper_backend(mocker)
        mocker.patch("localwispr.audio.recorder.sd")

        # models_in_tmp() with no arg = no model file = not downloaded
        models_in_tmp()

        from localwispr.modes import ModeManager, ModeType
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        mode_manager.set_mode(ModeType.DICTATION)

        pipeline = RecordingPipeline(mode_manager=mode_manager, executor=sync_executor)

        # Run preload — should detect model not downloaded and set error
        pipeline.preload_model_async()
        pipeline._model_preload_complete.wait(timeout=5.0)

        # Simulate recording
        mock_recorder = MagicMock()
        mock_recorder.is_recording = True
        mock_recorder.get_whisper_audio.return_value = np.zeros(16000, dtype=np.float32)
        pipeline._recorder = mock_recorder

        start = time.monotonic()
        result = pipeline.stop_and_transcribe()
        elapsed = time.monotonic() - start

        assert result.success is False
        assert "not downloaded" in result.error.lower()
        assert elapsed < 1.0

    def test_recording_produces_transcription_text(
        self, mocker, mock_config, models_in_tmp, reset_mode_manager, sync_executor,
    ):
        """User records and releases -> gets transcription text."""
        mocker.patch("localwispr.config.get_config", return_value=mock_config)
        mocker.patch("localwispr.transcribe.transcriber.get_config", return_value=mock_config)

        # Create dummy model file so is_model_downloaded returns True
        models_in_tmp("tiny")

        # Mock whisper backend to return transcription segments
        segments = [MockSegment(text=" Hello world transcription.")]
        _mock_whisper_backend(mocker, segments=segments)

        mocker.patch("localwispr.audio.recorder.sd")

        from localwispr.modes import ModeManager, ModeType
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        mode_manager.set_mode(ModeType.DICTATION)

        pipeline = RecordingPipeline(mode_manager=mode_manager, executor=sync_executor)

        # Run preload — should succeed since model file exists
        pipeline.preload_model_async()
        pipeline._model_preload_complete.wait(timeout=5.0)

        # Simulate recording
        mock_recorder = MagicMock()
        mock_recorder.is_recording = True
        mock_recorder.get_whisper_audio.return_value = np.zeros(16000, dtype=np.float32)
        pipeline._recorder = mock_recorder

        result = pipeline.stop_and_transcribe()

        assert result.success is True
        assert "hello world transcription" in result.text.lower()

    def test_preload_error_recovery(
        self, mocker, mock_config, models_in_tmp, reset_mode_manager, sync_executor,
    ):
        """Model file exists but first load fails -> sync retry succeeds."""
        mocker.patch("localwispr.config.get_config", return_value=mock_config)
        mocker.patch("localwispr.transcribe.transcriber.get_config", return_value=mock_config)

        models_in_tmp("tiny")

        # First Model() call raises (during preload), second succeeds (during sync fallback)
        mock_model = MagicMock()
        segments = [MockSegment(text=" Recovery transcription.")]
        if _FASTER_WHISPER_AVAILABLE:
            mock_model.transcribe.return_value = (iter(segments), MockTranscriptionInfo())
        else:
            mock_model.transcribe.return_value = segments
        _mock_whisper_backend(mocker, side_effect=[RuntimeError("corrupted model"), mock_model])

        mocker.patch("localwispr.audio.recorder.sd")

        from localwispr.modes import ModeManager, ModeType
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        mode_manager.set_mode(ModeType.DICTATION)

        pipeline = RecordingPipeline(mode_manager=mode_manager, executor=sync_executor)

        # Preload fails (first Model() call raises)
        pipeline.preload_model_async()
        pipeline._model_preload_complete.wait(timeout=5.0)

        assert pipeline.model_preload_state == "error"

        # Simulate recording
        mock_recorder = MagicMock()
        mock_recorder.is_recording = True
        mock_recorder.get_whisper_audio.return_value = np.zeros(16000, dtype=np.float32)
        pipeline._recorder = mock_recorder

        # Sync fallback should retry and succeed
        result = pipeline.stop_and_transcribe()

        assert result.success is True
        assert "recovery transcription" in result.text.lower()

    def test_model_preload_state_reflects_reality(
        self, mocker, mock_config, models_in_tmp, reset_mode_manager,
    ):
        """model_preload_state returns correct values at each lifecycle stage."""
        mocker.patch("localwispr.config.get_config", return_value=mock_config)
        _mock_whisper_backend(mocker)

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)

        # Before preload starts
        assert pipeline.model_preload_state == "loading"

        # Case 1: Model not downloaded
        models_in_tmp()  # no model file
        pipeline.preload_model_async()
        pipeline._model_preload_complete.wait(timeout=5.0)

        assert pipeline.model_preload_state == "not_downloaded"

    def test_get_model_name_shows_useful_status(
        self, mocker, mock_config, models_in_tmp, reset_mode_manager,
    ):
        """get_model_name() returns user-facing strings at correct stages."""
        mocker.patch("localwispr.config.get_config", return_value=mock_config)
        _mock_whisper_backend(mocker)

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)

        # Before preload
        assert pipeline.get_model_name() == "Loading..."

        # Model not downloaded
        models_in_tmp()
        pipeline.preload_model_async()
        pipeline._model_preload_complete.wait(timeout=5.0)

        assert pipeline.get_model_name() == "Not downloaded"

    def test_async_transcription_callbacks_fire_correctly(
        self, mocker, mock_config, models_in_tmp, reset_mode_manager, sync_executor,
    ):
        """on_result and on_complete callbacks fire with correct data."""
        mocker.patch("localwispr.config.get_config", return_value=mock_config)
        mocker.patch("localwispr.transcribe.transcriber.get_config", return_value=mock_config)

        models_in_tmp("tiny")

        segments = [MockSegment(text=" Async test result.")]
        _mock_whisper_backend(mocker, segments=segments)

        mocker.patch("localwispr.audio.recorder.sd")

        from localwispr.modes import ModeManager, ModeType
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        mode_manager.set_mode(ModeType.DICTATION)

        pipeline = RecordingPipeline(mode_manager=mode_manager, executor=sync_executor)

        # Preload model
        pipeline.preload_model_async()
        pipeline._model_preload_complete.wait(timeout=5.0)

        # Simulate recording
        mock_recorder = MagicMock()
        mock_recorder.is_recording = True
        mock_recorder.get_whisper_audio.return_value = np.zeros(16000, dtype=np.float32)
        pipeline._recorder = mock_recorder

        # Track callbacks
        results = []
        completions = []

        def on_result(result, gen):
            results.append((result, gen))

        def on_complete(gen):
            completions.append(gen)

        gen = pipeline.stop_and_transcribe_async(
            on_result=on_result, on_complete=on_complete,
        )

        assert len(results) == 1
        assert results[0][0].success is True
        assert "async test result" in results[0][0].text.lower()
        assert results[0][1] == gen
        assert len(completions) == 1
        assert completions[0] == gen

    def test_record_rejected_when_model_missing(
        self, mocker, mock_config, models_in_tmp, reset_mode_manager,
    ):
        """Pressing hotkey with no model -> overlay shows error, no recording."""
        mocker.patch("localwispr.config.get_config", return_value=mock_config)
        _mock_whisper_backend(mocker)

        # No model file
        models_in_tmp()

        # Mock UI boundaries
        mock_overlay = MagicMock()
        mocker.patch("localwispr.ui.overlay.OverlayWidget", return_value=mock_overlay)

        import localwispr.settings.manager as sm_module
        sm_module._settings_manager = None

        from localwispr.ui.tray import TrayApp

        app = TrayApp()

        # Run preload so pipeline knows model is missing
        app._pipeline.preload_model_async()
        app._pipeline._model_preload_complete.wait(timeout=5.0)

        assert app._pipeline.model_preload_state == "not_downloaded"

        # Simulate hotkey press
        app._on_record_start()

        # Overlay should show error
        mock_overlay.show_error.assert_called_once_with(
            "Model not downloaded. Open Settings to download."
        )
        # Recording should NOT have started
        assert app._pipeline.is_recording is False
