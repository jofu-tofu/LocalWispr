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
            "coding_keywords": ["function", "variable", "import", "class", "def", "return", "async", "await", "const", "let", "var", "public", "private", "interface", "type", "null", "undefined"],
            "planning_keywords": ["task", "project", "milestone", "deadline", "goal", "plan", "schedule", "priority", "action", "item", "todo", "complete", "review"],
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
coding_keywords = ["function", "variable", "import", "class", "def", "return", "async", "await", "const", "let", "var", "public", "private", "interface", "type", "null", "undefined"]
planning_keywords = ["task", "project", "milestone", "deadline", "goal", "plan", "schedule", "priority", "action", "item", "todo", "complete", "review"]

[output]
auto_paste = false
paste_delay_ms = 10

[vocabulary]
words = ["LocalWispr", "pytest"]
"""
    config_file.write_text(content)
    return config_file


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
def mock_audio_recorder(mocker):
    """Mock AudioRecorder with proper state transitions.

    Provides a reusable AudioRecorder mock that can be used across tests
    instead of manually creating the same mock setup repeatedly.

    The mock handles recording state automatically:
    - start_recording() sets is_recording = True
    - stop_recording() sets is_recording = False and returns audio

    Returns:
        MagicMock configured as AudioRecorder with common test behaviors.
    """
    mock_recorder = MagicMock()
    mock_recorder.is_recording = False
    mock_recorder.sample_rate = 16000

    # Store audio chunks
    mock_recorder._audio_buffer = []

    def start_recording_side_effect():
        mock_recorder.is_recording = True

    def stop_recording_side_effect():
        mock_recorder.is_recording = False
        if mock_recorder._audio_buffer:
            return np.concatenate(mock_recorder._audio_buffer)
        return np.zeros(16000, dtype=np.float32)

    def get_whisper_audio_side_effect():
        return stop_recording_side_effect()

    mock_recorder.start_recording.side_effect = start_recording_side_effect
    mock_recorder.stop_recording.side_effect = stop_recording_side_effect
    mock_recorder.get_whisper_audio.side_effect = get_whisper_audio_side_effect
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


# ============================================================================
# Utility Fixtures
# ============================================================================


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
coding_keywords = ["function", "variable", "import", "class", "def", "return", "async", "await", "const", "let", "var", "public", "private", "interface", "type", "null", "undefined"]
planning_keywords = ["task", "project", "milestone", "deadline", "goal", "plan", "schedule", "priority", "action", "item", "todo", "complete", "review"]

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
    # NOTE: Patches faster_whisper.WhisperModel at SOURCE, not usage location.
    # This is correct: transcribe.py uses TYPE_CHECKING guard (line 16), meaning
    # WhisperModel is only imported inside methods for lazy loading. Source-level
    # patching is required for lazy imports - usage location doesn't exist yet.
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
def isolated_config_cache():
    """Isolate config cache to prevent test interference.

    Use this fixture for tests that modify or stress-test the config cache.
    Restores original cache state after test completes.
    """
    from localwispr import config as config_module
    from localwispr.config import clear_config_cache

    # Save original state
    original_cache = config_module._cached_config

    # Clear for test isolation
    clear_config_cache()

    yield

    # Restore original state
    config_module._cached_config = original_cache
