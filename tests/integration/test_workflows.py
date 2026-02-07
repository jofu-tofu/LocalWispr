"""Integration tests for LocalWispr workflows.

These tests verify complete workflows with minimal mocking, testing
the integration between multiple modules.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from tests.helpers import MockSegment


class TestRecordingWorkflow:
    """Tests for the complete recording → transcription → output workflow."""

    def test_recording_to_transcription_workflow(self, mocker, mock_config, tmp_path, mock_model_downloaded):
        """Test complete recording → transcription workflow."""
        # Setup config
        mocker.patch("localwispr.config.get_config", return_value=mock_config)
        mocker.patch("localwispr.config._get_defaults_path", return_value=tmp_path / "config-defaults.toml")
        mocker.patch("localwispr.config._get_appdata_config_path", return_value=tmp_path / "user-settings.toml")

        # Mock audio recorder to return test audio
        mock_recorder = MagicMock()
        mock_recorder.is_recording = True
        mock_recorder.get_whisper_audio.return_value = np.random.randn(16000).astype(
            np.float32
        )
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            return_value=mock_recorder,
        )

        # Mock pywhispercpp Model
        mock_model = MagicMock()
        segments = [MockSegment(text=" This is a test transcription.")]
        mock_model.transcribe.return_value = segments
        mocker.patch(
            "pywhispercpp.model.Model",
            return_value=mock_model,
        )
        mocker.patch("localwispr.transcribe.model_manager.is_model_downloaded", return_value=True)
        mocker.patch("localwispr.transcribe.model_manager.get_model_path", return_value="/fake/path/model.bin")

        # Mock output (unused but prevents real clipboard operations)
        mocker.patch(
            "localwispr.output.copy_to_clipboard",
            return_value=True,
        )

        # Reset mode manager
        import localwispr.modes.manager as manager_module

        manager_module._mode_manager = None

        from localwispr.modes import ModeManager, ModeType
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        mode_manager.set_mode(ModeType.DICTATION)

        pipeline = RecordingPipeline(mode_manager=mode_manager)
        pipeline._recorder = mock_recorder
        pipeline._model_preload_complete.set()

        # Execute workflow
        result = pipeline.stop_and_transcribe()

        # Verify
        assert result.success is True
        assert "test transcription" in result.text.lower()

    def test_mode_affects_transcription(self, mocker, mock_config, tmp_path, mock_model_downloaded):
        """Test that different modes use different prompts."""
        mocker.patch("localwispr.config.get_config", return_value=mock_config)
        mocker.patch("localwispr.config._get_defaults_path", return_value=tmp_path / "config-defaults.toml")
        mocker.patch("localwispr.config._get_appdata_config_path", return_value=tmp_path / "user-settings.toml")

        # Reset mode manager
        import localwispr.modes.manager as manager_module

        manager_module._mode_manager = None

        # Track prompts used
        prompts_used = []

        def track_prompt(*args, **kwargs):
            if "initial_prompt" in kwargs:
                prompts_used.append(kwargs["initial_prompt"])
            return MagicMock(
                text="test",
                segments=[],
                inference_time=0.1,
                audio_duration=1.0,
                detected_context=None,
                was_retranscribed=False,
            )

        mock_transcriber = MagicMock()
        mock_transcriber.model = MagicMock()
        mock_transcriber.transcribe.side_effect = track_prompt

        mocker.patch(
            "localwispr.transcribe.transcriber.WhisperTranscriber",
            return_value=mock_transcriber,
        )

        mock_recorder = MagicMock()
        mock_recorder.is_recording = True
        mock_recorder.get_whisper_audio.return_value = np.zeros(16000, dtype=np.float32)
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            return_value=mock_recorder,
        )

        from localwispr.modes import ModeManager, ModeType
        from localwispr.pipeline import RecordingPipeline

        # Test with CODE mode
        mode_manager = ModeManager()
        mode_manager.set_mode(ModeType.CODE)

        pipeline = RecordingPipeline(mode_manager=mode_manager)
        pipeline._recorder = mock_recorder
        pipeline._model_preload_complete.set()

        pipeline.stop_and_transcribe()

        # Verify prompt was passed
        assert len(prompts_used) == 1
        # The prompt should be loaded for coding mode
        assert prompts_used[0] is not None


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
