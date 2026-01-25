"""Shared pytest fixtures for LocalWispr tests.

This module provides reusable fixtures for mocking:
- Audio recording (sounddevice)
- Whisper transcription (faster_whisper)
- Clipboard operations (pyperclip)
- Configuration loading
- Mode management
- Synchronous executor for deterministic async testing
"""

from __future__ import annotations

from concurrent.futures import Executor, Future
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.helpers import MockSegment, MockTranscriptionInfo


# ============================================================================
# Synchronous Executor for Testing
# ============================================================================


class SynchronousExecutor(Executor):
    """Executor that runs tasks immediately in calling thread.

    Useful for testing async code deterministically without threading.
    """

    def submit(self, fn, *args, **kwargs):
        """Execute fn immediately and return a completed Future."""
        future = Future()
        try:
            result = fn(*args, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        return future

    def shutdown(self, wait: bool = True) -> None:
        """No-op shutdown since tasks run synchronously."""
        pass


# ============================================================================
# Audio Fixtures
# ============================================================================


@pytest.fixture
def mock_audio_data() -> np.ndarray:
    """Generate 1 second of silence as mock audio data.

    Returns:
        NumPy array of zeros (16kHz, float32).
    """
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def mock_audio_with_signal() -> np.ndarray:
    """Generate 1 second of audio with a sine wave signal.

    Returns:
        NumPy array with sine wave (16kHz, float32).
    """
    t = np.linspace(0, 1, 16000, dtype=np.float32)
    # 440Hz sine wave at 0.5 amplitude
    return 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def mock_stereo_audio() -> np.ndarray:
    """Generate 1 second of stereo audio.

    Returns:
        NumPy array with shape (16000, 2).
    """
    mono = np.zeros(16000, dtype=np.float32)
    return np.column_stack([mono, mono])


@pytest.fixture
def mock_sounddevice(mocker) -> MagicMock:
    """Mock sounddevice module to avoid actual audio capture.

    Returns:
        Mock sounddevice module.
    """
    mock_sd = mocker.patch("localwispr.audio.sd")

    # Mock default device
    mock_sd.default.device = (0, None)

    # Mock device query
    mock_sd.query_devices.return_value = {
        "name": "Mock Microphone",
        "max_input_channels": 2,
        "default_samplerate": 48000.0,
        "index": 0,
    }

    # Mock InputStream
    mock_stream = MagicMock()
    mock_sd.InputStream.return_value = mock_stream

    return mock_sd


# ============================================================================
# Transcription Fixtures
# ============================================================================


@pytest.fixture
def mock_whisper_model(mocker) -> MagicMock:
    """Mock faster_whisper.WhisperModel.

    Returns:
        Mock WhisperModel that returns test transcription.
    """
    mock_model = MagicMock()

    # Default transcription result
    segments = [MockSegment(text="test transcription")]
    info = MockTranscriptionInfo()

    mock_model.transcribe.return_value = (iter(segments), info)

    return mock_model


@pytest.fixture
def mock_whisper_module(mocker, mock_whisper_model) -> MagicMock:
    """Mock the faster_whisper module.

    Returns:
        Mock module with WhisperModel class.

    NOTE: Patches faster_whisper.WhisperModel (source) because transcribe.py
    uses lazy imports (imports WhisperModel inside functions, not at module level).
    """
    mock_module = mocker.patch("faster_whisper.WhisperModel")
    mock_module.return_value = mock_whisper_model
    return mock_module


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def mock_config() -> dict[str, Any]:
    """Create a test configuration dictionary.

    Returns:
        Configuration dict with sensible test defaults.
    """
    return {
        "model": {
            "name": "tiny",
            "device": "cpu",
            "compute_type": "int8",
            "language": "auto",
        },
        "hotkeys": {
            "mode": "push-to-talk",
            "modifiers": ["ctrl", "shift"],
            "audio_feedback": False,
            "mute_system": False,
        },
        "context": {
            "coding_apps": ["code", "pycharm"],
            "planning_apps": ["notion", "obsidian"],
            "coding_keywords": ["function", "variable", "class"],
            "planning_keywords": ["task", "project", "deadline"],
        },
        "output": {
            "auto_paste": False,
            "paste_delay_ms": 10,
        },
        "vocabulary": {
            "words": ["LocalWispr", "pytest"],
        },
    }


@pytest.fixture
def mock_config_file(tmp_path, mock_config) -> Path:
    """Create a temporary config.toml file.

    Returns:
        Path to temporary config file.
    """
    config_file = tmp_path / "config.toml"
    content = """
[model]
name = "tiny"
device = "cpu"
compute_type = "int8"
language = "auto"

[hotkeys]
mode = "push-to-talk"
modifiers = ["ctrl", "shift"]
audio_feedback = false
mute_system = false

[context]
coding_apps = ["code", "pycharm"]
planning_apps = ["notion", "obsidian"]
coding_keywords = ["function", "variable", "class"]
planning_keywords = ["task", "project", "deadline"]

[output]
auto_paste = false
paste_delay_ms = 10

[vocabulary]
words = ["LocalWispr", "pytest"]
"""
    config_file.write_text(content)
    return config_file


@pytest.fixture
def patch_get_config(mocker, mock_config):
    """Patch get_config to return mock configuration.

    Returns:
        The patched function.
    """
    return mocker.patch("localwispr.config.get_config", return_value=mock_config)


# ============================================================================
# Clipboard/Output Fixtures
# ============================================================================


@pytest.fixture
def mock_pyperclip(mocker) -> MagicMock:
    """Mock pyperclip module.

    Returns:
        Mock pyperclip with copy/paste tracking.
    """
    mock = mocker.patch("localwispr.output.pyperclip")
    mock.clipboard_content = ""

    def copy_side_effect(text):
        mock.clipboard_content = text

    mock.copy.side_effect = copy_side_effect
    mock.paste.side_effect = lambda: mock.clipboard_content

    return mock


@pytest.fixture
def mock_keyboard(mocker) -> MagicMock:
    """Mock pynput keyboard Controller.

    Returns:
        Mock keyboard controller.
    """
    mock_controller = MagicMock()
    mocker.patch("localwispr.output.Controller", return_value=mock_controller)
    return mock_controller


# ============================================================================
# Mode Fixtures
# ============================================================================


@pytest.fixture
def reset_mode_manager():
    """Reset the global mode manager between tests.

    Yields after test completes to perform cleanup.
    """
    yield

    # Reset global mode manager
    import localwispr.modes.manager as manager_module

    manager_module._mode_manager = None


@pytest.fixture
def mock_mode_manager(mocker, reset_mode_manager):
    """Create a fresh ModeManager for testing.

    Returns:
        New ModeManager instance.
    """
    from localwispr.modes import ModeManager

    return ModeManager(auto_reset=False)


# ============================================================================
# Executor Fixtures
# ============================================================================


@pytest.fixture
def sync_executor():
    """Provide a synchronous executor for deterministic async testing.

    Returns:
        SynchronousExecutor that runs tasks immediately.
    """
    return SynchronousExecutor()


# ============================================================================
# Pipeline Fixtures
# ============================================================================


@pytest.fixture
def mock_pipeline_dependencies(mocker, mock_config):
    """Mock all pipeline dependencies for isolated testing.

    Returns:
        Dict of all mocked components.
    """
    mocks = {}

    # Mock config
    mocks["config"] = mocker.patch(
        "localwispr.pipeline.get_config",
        return_value=mock_config,
    )

    # Mock AudioRecorder
    mock_recorder = MagicMock()
    mock_recorder.is_recording = False
    mock_recorder.get_whisper_audio.return_value = np.zeros(16000, dtype=np.float32)
    mock_recorder.get_rms_level.return_value = 0.5
    mocks["recorder_class"] = mocker.patch(
        "localwispr.audio.AudioRecorder",
        return_value=mock_recorder,
    )
    mocks["recorder"] = mock_recorder

    # Mock WhisperTranscriber
    mock_transcriber = MagicMock()
    mock_transcriber.model = MagicMock()
    mock_transcriber.is_loaded = True

    # Mock transcribe return value with test data
    mock_transcriber.transcribe.return_value = MagicMock(
        text="test transcription",
        segments=[{"start": 0.0, "end": 1.0, "text": "test transcription"}],
        inference_time=0.5,
        audio_duration=1.0,
        was_retranscribed=False,
    )
    mocks["transcriber_class"] = mocker.patch(
        "localwispr.transcribe.WhisperTranscriber",
        return_value=mock_transcriber,
    )
    mocks["transcriber"] = mock_transcriber

    return mocks


@pytest.fixture
def mock_audio_recorder(mocker):
    """Mock AudioRecorder with standard test configuration.

    Provides a reusable AudioRecorder mock that can be used across tests
    instead of manually creating the same mock setup repeatedly.

    Returns:
        MagicMock configured as AudioRecorder with common test behaviors.
    """
    mock_recorder = MagicMock()
    mock_recorder.is_recording = False
    mock_recorder.sample_rate = 16000
    mock_recorder.get_whisper_audio.return_value = np.zeros(16000, dtype=np.float32)
    mock_recorder.get_rms_level.return_value = 0.5
    mocker.patch("localwispr.audio.AudioRecorder", return_value=mock_recorder)
    return mock_recorder


@pytest.fixture
def mock_whisper_transcriber(mocker):
    """Mock WhisperTranscriber with standard test transcription result.

    Provides a reusable WhisperTranscriber mock with properly structured
    transcription results for testing.

    Returns:
        MagicMock configured as WhisperTranscriber with test transcription.
    """
    mock_transcriber = MagicMock()
    mock_transcriber.model = MagicMock()
    mock_transcriber.is_loaded = True

    # Create properly structured mock result
    mock_result = MagicMock()
    mock_result.text = "test transcription"
    mock_result.audio_duration = 1.0
    mock_result.inference_time = 0.5
    mock_result.was_retranscribed = False
    mock_result.segments = [{"start": 0.0, "end": 1.0, "text": "test transcription"}]

    mock_transcriber.transcribe.return_value = mock_result
    mocker.patch("localwispr.transcribe.WhisperTranscriber", return_value=mock_transcriber)
    return mock_transcriber


# ============================================================================
# Context Detection Fixtures
# ============================================================================


@pytest.fixture
def mock_window_title(mocker) -> MagicMock:
    """Mock window title detection.

    Returns:
        Mock that can be configured with different window titles.
    """
    mock = mocker.patch("localwispr.context.pygetwindow")
    mock.getActiveWindowTitle.return_value = "Test Window"
    return mock


# ============================================================================
# Utility Fixtures
# ============================================================================


@pytest.fixture
def temp_dir(tmp_path) -> Path:
    """Provide a temporary directory for test files.

    Returns:
        Path to temporary directory.
    """
    return tmp_path


@pytest.fixture
def silence_logging(mocker):
    """Silence all logging during tests.

    Useful for tests that might produce noisy log output.
    """
    mocker.patch("logging.Logger.debug")
    mocker.patch("logging.Logger.info")
    mocker.patch("logging.Logger.warning")
    mocker.patch("logging.Logger.error")


# ============================================================================
# Integration Test Fixtures
# ============================================================================


@pytest.fixture
def full_app_context(tmp_path, mocker):
    """Provide integration test context with temp config and mocked dependencies.

    This fixture provides:
    - Temporary config file (isolated from user settings)
    - Mocked external deps: sounddevice, faster_whisper
    - Real components: config system, settings manager, pipeline, mode manager

    Use this for integration tests that need to verify end-to-end workflows
    like settings propagation without initializing the full UI.

    Returns:
        Dict with keys:
            - "config_path": Path to temp config file
            - "mocks": Dict of mocked external dependencies
            - "pipeline": RecordingPipeline instance (lazy-loaded)
            - "mode_manager": ModeManager instance (lazy-loaded)
    """
    from localwispr.config import clear_config_cache, load_config
    import localwispr.modes.manager as manager_module

    # Create temp config file
    config_path = tmp_path / "test-config.toml"
    config_path.write_text("""
[model]
name = "tiny"
device = "cpu"
compute_type = "int8"
language = "auto"

[hotkeys]
mode = "push-to-talk"
modifiers = ["ctrl", "shift"]
key = "space"
audio_feedback = false
mute_system = false

[context]
coding_apps = ["code", "pycharm"]
planning_apps = ["notion", "obsidian"]
coding_keywords = ["function", "variable", "class"]
planning_keywords = ["task", "project", "deadline"]

[output]
auto_paste = false
paste_delay_ms = 10

[vocabulary]
words = ["LocalWispr", "pytest"]

[streaming]
enabled = false
vad_threshold = 0.5
""")

    # Mock external dependencies
    mocks = {}

    # Mock sounddevice
    mock_sd = mocker.patch("localwispr.audio.sd")
    mock_sd.default.device = (0, None)
    mock_sd.query_devices.return_value = {
        "name": "Mock Microphone",
        "max_input_channels": 2,
        "default_samplerate": 48000.0,
        "index": 0,
    }
    mock_stream = MagicMock()
    mock_sd.InputStream.return_value = mock_stream
    mocks["sounddevice"] = mock_sd

    # Mock WhisperModel
    # NOTE: Patch faster_whisper.WhisperModel (source) because transcribe.py
    # uses lazy imports (imports WhisperModel inside functions, not at module level)
    mock_whisper_model = MagicMock()
    segments = [MockSegment(text="test transcription")]
    info = MockTranscriptionInfo()
    mock_whisper_model.transcribe.return_value = (iter(segments), info)
    mock_whisper_class = mocker.patch("faster_whisper.WhisperModel")
    mock_whisper_class.return_value = mock_whisper_model
    mocks["whisper"] = mock_whisper_class

    # Patch load_config to use temp file
    clear_config_cache()
    manager_module._mode_manager = None

    # Import the real load_config function to use in our side_effect
    from localwispr.config import load_config as real_load_config
    mocker.patch("localwispr.config.load_config", side_effect=lambda path=None: real_load_config(config_path))

    context = {
        "config_path": config_path,
        "mocks": mocks,
    }

    # Lazy-load components (only create when accessed)
    def get_pipeline():
        if "pipeline" not in context:
            from localwispr.pipeline import RecordingPipeline
            from localwispr.modes import get_mode_manager

            mode_manager = get_mode_manager(auto_reset=False)
            pipeline = RecordingPipeline(mode_manager=mode_manager)
            context["pipeline"] = pipeline
            context["mode_manager"] = mode_manager
        return context["pipeline"]

    def get_mode_manager():
        if "mode_manager" not in context:
            get_pipeline()  # This will create both
        return context["mode_manager"]

    context["get_pipeline"] = get_pipeline
    context["get_mode_manager"] = get_mode_manager

    yield context

    # Cleanup
    clear_config_cache()
    manager_module._mode_manager = None


@pytest.fixture
def settings_flow_helper(mocker):
    """Helper for simulating settings changes via SettingsController.

    This fixture provides utilities to test the full settings flow:
    GUI Change → Controller Save → Manager Apply → Handlers Execute

    Returns:
        SettingsFlowHelper instance with simulation methods.
    """
    from tests.helpers import SettingsFlowHelper

    return SettingsFlowHelper(mocker)
