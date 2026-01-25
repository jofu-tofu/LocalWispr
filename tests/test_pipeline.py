"""Tests for the recording pipeline module."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import numpy as np
import pytest


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

    def test_pipeline_start_recording(self, mock_audio_recorder, reset_mode_manager):
        """Test starting recording."""
        # Configure mock to report recording state correctly
        mock_audio_recorder.is_recording = False

        def start_recording_side_effect():
            mock_audio_recorder.is_recording = True

        mock_audio_recorder.start_recording.side_effect = start_recording_side_effect

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)

        result = pipeline.start_recording()

        assert result is True
        assert pipeline.is_recording is True
        mock_audio_recorder.start_recording.assert_called_once()

    def test_pipeline_start_recording_already_recording(self, mock_audio_recorder, reset_mode_manager):
        """Test starting recording when already recording."""
        # Configure mock for already-recording state
        mock_audio_recorder.is_recording = True

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)
        pipeline._recorder = mock_audio_recorder

        result = pipeline.start_recording()

        assert result is False

    def test_pipeline_preload_model_async(self, mocker, reset_mode_manager):
        """Test async model preloading."""
        mock_transcriber = MagicMock()
        mock_transcriber.model = MagicMock()
        mock_transcriber_class = mocker.patch(
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
        # Verify model was actually instantiated
        mock_transcriber_class.assert_called_once()
        assert pipeline._transcriber is mock_transcriber

    def test_pipeline_stop_and_transcribe_no_audio(self, mocker, reset_mode_manager):
        """Test stop_and_transcribe when not currently recording."""
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
        assert "no audio" in result.error.lower() or "not recording" in result.error.lower()

    def test_pipeline_stop_and_transcribe_success(
        self,
        mocker,
        mock_audio_recorder,
        mock_whisper_transcriber,
        reset_mode_manager
    ):
        """Test successful stop_and_transcribe."""
        # Configure mocks for recording state
        mock_audio_recorder.is_recording = True

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
        pipeline._recorder = mock_audio_recorder
        pipeline._model_preload_complete.set()

        result = pipeline.stop_and_transcribe()

        assert result.success is True
        assert result.text == "test transcription"

    def test_pipeline_cancel_recording(self, mock_audio_recorder, reset_mode_manager):
        """Test canceling a recording."""
        # Configure mock for recording state
        mock_audio_recorder.is_recording = True

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)
        pipeline._recorder = mock_audio_recorder

        pipeline.cancel_recording()

        mock_audio_recorder.stop_recording.assert_called_once()

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

    def test_pipeline_model_timeout(self, mocker, reset_mode_manager):
        """Test pipeline handles model load timeout during transcription."""
        import threading

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
        pipeline.MODEL_LOAD_TIMEOUT = 0.1  # Short but realistic timeout

        # Start async preload that will never complete (don't call set())
        # This simulates a hung model load
        pipeline._model_preload_thread = threading.Thread(target=lambda: None)
        pipeline._model_preload_thread.start()

        result = pipeline.stop_and_transcribe()

        assert result.success is False
        assert "timeout" in result.error.lower()

    def test_model_preload_failure_triggers_sync_fallback(
        self, mocker, reset_mode_manager
    ):
        """Test that preload failure is recovered via synchronous loading."""
        # Track transcriber initialization calls
        init_count = [0]

        def mock_transcriber_init(*args, **kwargs):
            init_count[0] += 1
            mock = MagicMock()
            mock.model = MagicMock()
            return mock

        mocker.patch(
            "localwispr.transcribe.WhisperTranscriber",
            side_effect=mock_transcriber_init,
        )

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)

        # Simulate preload failure
        pipeline._model_preload_error = Exception("Network error during preload")
        pipeline._model_preload_complete.set()

        # Call _get_transcriber - should detect error and retry sync
        transcriber = pipeline._get_transcriber()

        # Verify sync fallback succeeded
        assert transcriber is not None
        assert init_count[0] == 1  # Sync init happened
        assert pipeline._model_preload_error is None  # Error cleared
        assert pipeline._transcriber is transcriber  # Stored for reuse

    def test_get_model_name_during_loading(self, mocker, reset_mode_manager):
        """Test get_model_name returns loading status before model ready."""
        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)

        # Before preload completes
        assert pipeline.get_model_name() == "Loading..."

        # After preload completes but before transcriber init
        pipeline._model_preload_complete.set()
        assert pipeline.get_model_name() == "Initializing..."

        # After transcriber initialized
        mock_transcriber = mocker.Mock()
        mock_transcriber.model_name = "large-v3"
        with pipeline._transcriber_lock:
            pipeline._transcriber = mock_transcriber
        assert pipeline.get_model_name() == "large-v3"

    def test_pipeline_on_error_callback(self, mocker, reset_mode_manager):
        """Test that on_error callback is invoked when recorder initialization fails."""
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            side_effect=Exception("Audio device not found"),
        )

        error_callback = MagicMock()

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager, on_error=error_callback)

        result = pipeline.start_recording()

        assert result is False
        error_callback.assert_called_once()
        # Verify error callback was invoked with an error message
        call_args = error_callback.call_args[0][0]
        assert isinstance(call_args, str) and len(call_args) > 0

    def test_pipeline_thread_safety(self, mocker, reset_mode_manager):
        """Test that concurrent start_recording calls enforce mutual exclusion via lock.

        This test verifies mutual exclusion: when multiple threads try to
        start recording simultaneously, only ONE should succeed. The rest
        should fail because recording is already in progress.

        This catches race conditions where the lock isn't working properly.
        """
        # Track constructor calls to verify only ONE recorder instance is created
        constructor_calls = []

        # Create a stateful mock class that simulates real AudioRecorder behavior
        class MockAudioRecorder:
            def __init__(self, *args, **kwargs):
                constructor_calls.append(1)
                self.is_recording = False
                self.sample_rate = 16000

            def start_recording(self):
                self.is_recording = True

            def stop_recording(self):
                self.is_recording = False
                return np.zeros(16000, dtype=np.float32)

            def get_whisper_audio(self):
                return np.zeros(16000, dtype=np.float32)

        mocker.patch(
            "localwispr.audio.AudioRecorder",
            MockAudioRecorder,
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

        # Verify all threads completed
        assert len(results) == 5

        # Verify mutual exclusion - only ONE thread should succeed
        successes = sum(1 for r in results if r is True)
        assert successes == 1, f"Expected exactly 1 success due to mutual exclusion, got {successes}"

        # Verify only ONE AudioRecorder was actually created (strongest proof of mutual exclusion)
        assert len(constructor_calls) == 1, f"Expected 1 AudioRecorder instance, got {len(constructor_calls)}"

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


class TestAsyncTranscription:
    """Tests for async transcription methods."""

    def test_stop_and_transcribe_async_success(
        self, mocker, reset_mode_manager, sync_executor
    ):
        """Verify async transcription returns result via callback."""
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
        mock_result.text = "async test transcription"
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
        mode_manager.set_mode(ModeType.CODE)

        pipeline = RecordingPipeline(
            mode_manager=mode_manager,
            executor=sync_executor,
        )
        pipeline._recorder = mock_recorder
        pipeline._model_preload_complete.set()

        # Capture callbacks (callbacks now receive generation parameter)
        results = []
        completed = []

        gen = pipeline.stop_and_transcribe_async(
            on_result=lambda r, g: results.append(r),
            on_complete=lambda g: completed.append(True),
        )

        assert gen == 1
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].text == "async test transcription"
        assert len(completed) == 1

    def test_stop_and_transcribe_async_no_audio(
        self, mocker, reset_mode_manager, sync_executor
    ):
        """Verify async transcription handles no audio case."""
        mock_recorder = MagicMock()
        mock_recorder.is_recording = False  # Not recording
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            return_value=mock_recorder,
        )

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(
            mode_manager=mode_manager,
            executor=sync_executor,
        )
        pipeline._recorder = mock_recorder

        results = []
        completed = []

        pipeline.stop_and_transcribe_async(
            on_result=lambda r, g: results.append(r),
            on_complete=lambda g: completed.append(True),
        )

        assert len(results) == 1
        assert results[0].success is False
        assert "No audio" in results[0].error
        assert len(completed) == 1

    def test_stale_generation_skips_callback(
        self, mocker, reset_mode_manager, sync_executor
    ):
        """Verify callbacks are skipped when generation counter increments (stale transcription detection)."""
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

        # Use a custom executor that increments generation before completing
        class StaleExecutor:
            def __init__(self, pipeline):
                self._pipeline = pipeline

            def submit(self, fn, *args, **kwargs):
                from concurrent.futures import Future

                future = Future()
                # Increment generation to make this request stale
                with self._pipeline._generation_lock:
                    self._pipeline._current_generation += 1
                # Now run the function (should detect staleness)
                try:
                    result = fn(*args, **kwargs)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                return future

            def shutdown(self, wait=True):
                pass

        pipeline = RecordingPipeline(mode_manager=mode_manager)
        pipeline._executor = StaleExecutor(pipeline)  # Replace executor
        pipeline._transcription_executor = StaleExecutor(pipeline)
        pipeline._owns_executor = False
        pipeline._recorder = mock_recorder
        pipeline._model_preload_complete.set()

        results = []
        completed = []

        pipeline.stop_and_transcribe_async(
            on_result=lambda r, g: results.append(r),
            on_complete=lambda g: completed.append(True),
        )

        # Callbacks should NOT be invoked because generation became stale
        assert len(results) == 0
        assert len(completed) == 0

    def test_shutdown_skips_callbacks(
        self, mocker, reset_mode_manager, sync_executor
    ):
        """Verify callbacks are skipped when shutdown flag is set (prevents callbacks during cleanup)."""
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

        # Use a custom executor that calls shutdown before completing
        class ShutdownExecutor:
            def __init__(self, pipeline):
                self._pipeline = pipeline

            def submit(self, fn, *args, **kwargs):
                from concurrent.futures import Future

                future = Future()
                # Call shutdown to mark shutting_down
                self._pipeline.shutdown(timeout=0)
                # Now run the function (should detect shutdown)
                try:
                    result = fn(*args, **kwargs)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                return future

            def shutdown(self, wait=True):
                pass

        pipeline = RecordingPipeline(mode_manager=mode_manager)
        pipeline._transcription_executor = ShutdownExecutor(pipeline)
        pipeline._owns_executor = False
        pipeline._recorder = mock_recorder
        pipeline._model_preload_complete.set()

        results = []
        completed = []

        pipeline.stop_and_transcribe_async(
            on_result=lambda r, g: results.append(r),
            on_complete=lambda g: completed.append(True),
        )

        # Callbacks should NOT be invoked because shutdown was called
        assert len(results) == 0
        assert len(completed) == 0

    def test_callback_exception_doesnt_crash(
        self, mocker, reset_mode_manager, sync_executor
    ):
        """Verify exception in callback is caught and logged."""
        mock_recorder = MagicMock()
        mock_recorder.is_recording = True
        mock_recorder.get_whisper_audio.return_value = np.zeros(16000, dtype=np.float32)
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            return_value=mock_recorder,
        )

        # Mock transcriber
        mock_result = MagicMock()
        mock_result.text = "test"
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
        mocker.patch(
            "localwispr.modes.get_mode_prompt",
            return_value="test prompt",
        )

        from localwispr.modes import ModeManager, ModeType
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        mode_manager.set_mode(ModeType.CODE)

        pipeline = RecordingPipeline(
            mode_manager=mode_manager,
            executor=sync_executor,
        )
        pipeline._recorder = mock_recorder
        pipeline._model_preload_complete.set()

        def bad_callback(result, gen):
            raise ValueError("Callback error!")

        completed = []

        # Should not raise, even with bad callback
        pipeline.stop_and_transcribe_async(
            on_result=bad_callback,
            on_complete=lambda g: completed.append(True),
        )

        # on_complete should still be called
        assert len(completed) == 1

    def test_no_audio_restores_system_mute(
        self, mocker, reset_mode_manager, sync_executor
    ):
        """Verify system audio restored even when no audio captured."""
        mock_recorder = MagicMock()
        mock_recorder.is_recording = False  # Not recording initially
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            return_value=mock_recorder,
        )

        # Mock mute functions
        mock_mute = mocker.patch("localwispr.volume.mute_system", return_value=True)
        mock_restore = mocker.patch("localwispr.volume.restore_mute_state")

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(
            mode_manager=mode_manager,
            executor=sync_executor,
        )

        # Properly start recording with mute to set internal state
        pipeline.start_recording(mute_system=True)
        mock_mute.assert_called_once()

        # Now stop recording when no audio available
        mock_recorder.is_recording = False
        results = []

        pipeline.stop_and_transcribe_async(
            mute_system=True,
            on_result=lambda r, g: results.append(r),
        )

        # Should restore mute state even with no audio
        mock_restore.assert_called_once_with(True)

    def test_is_current_generation(self, reset_mode_manager, sync_executor):
        """Test is_current_generation helper method."""
        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(
            mode_manager=mode_manager,
            executor=sync_executor,
        )

        # Initially generation is 0
        assert pipeline.is_current_generation(0) is True
        assert pipeline.is_current_generation(1) is False

        # Increment generation
        with pipeline._generation_lock:
            pipeline._current_generation = 5

        assert pipeline.is_current_generation(5) is True
        assert pipeline.is_current_generation(4) is False

        # Shutdown makes all generations stale
        pipeline._shutting_down = True
        assert pipeline.is_current_generation(5) is False


class TestCriticalEdgeCases:
    """Tests for critical edge cases identified in code review."""

    def test_concurrent_start_recording_during_preload(self, mocker, reset_mode_manager):
        """Test race condition: start_recording() called while model is preloading."""
        import threading
        import time

        # Mock slow model loading
        mock_transcriber = MagicMock()
        mock_transcriber.model = MagicMock()

        load_count = [0]

        def slow_model_init(*args, **kwargs):
            load_count[0] += 1
            time.sleep(0.2)  # Simulate slow model load
            return mock_transcriber

        mocker.patch(
            "localwispr.transcribe.WhisperTranscriber",
            side_effect=slow_model_init,
        )

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

        # Start async preload
        pipeline.preload_model_async()

        # Immediately try to start recording (before preload completes)
        result = pipeline.start_recording()

        # Should succeed - start_recording waits for preload
        assert result is True
        # Model should only be loaded once (not duplicated)
        assert load_count[0] == 1

    def test_multiple_rapid_async_transcribe_calls(
        self, mocker, reset_mode_manager, sync_executor
    ):
        """Test that multiple rapid async calls handle generation counter correctly."""
        mock_recorder = MagicMock()
        mock_recorder.is_recording = True
        mock_recorder.get_whisper_audio.return_value = np.zeros(16000, dtype=np.float32)
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            return_value=mock_recorder,
        )

        mock_result = MagicMock()
        mock_result.text = "test"
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
        mocker.patch(
            "localwispr.modes.get_mode_prompt",
            return_value="test prompt",
        )

        from localwispr.modes import ModeManager, ModeType
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        mode_manager.set_mode(ModeType.CODE)

        pipeline = RecordingPipeline(
            mode_manager=mode_manager,
            executor=sync_executor,
        )
        pipeline._recorder = mock_recorder
        pipeline._model_preload_complete.set()

        results = []
        generations = []

        # Rapidly call async transcribe 5 times
        for _ in range(5):
            gen = pipeline.stop_and_transcribe_async(
                on_result=lambda r, g: (results.append(r), generations.append(g)),
            )

        # With sync executor, all calls complete. Verify generation increments.
        assert len(results) == 5
        # Each call should have unique generation number
        assert len(set(generations)) == 5

    def test_shutdown_during_start_recording(self, mocker, reset_mode_manager):
        """Test race condition: shutdown() called while start_recording() is executing."""
        import threading
        import time

        mock_recorder = MagicMock()
        mock_recorder.is_recording = False

        init_count = [0]

        def slow_recorder_init(*args, **kwargs):
            init_count[0] += 1
            time.sleep(0.2)  # Simulate slow init
            return mock_recorder

        mocker.patch(
            "localwispr.audio.AudioRecorder",
            side_effect=slow_recorder_init,
        )

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)

        results = []

        def start_rec():
            results.append(pipeline.start_recording())

        # Start recording in background
        t = threading.Thread(target=start_rec)
        t.start()

        # Immediately shutdown
        time.sleep(0.05)  # Brief delay to start init
        pipeline.shutdown(timeout=0.5)

        t.join()

        # start_recording should fail or return early due to shutdown
        assert len(results) == 1

    def test_zero_length_audio_array(self, mocker, reset_mode_manager):
        """Test boundary condition: AudioRecorder returns zero-length audio."""
        mock_recorder = MagicMock()
        mock_recorder.is_recording = True
        mock_recorder.get_whisper_audio.return_value = np.zeros(0, dtype=np.float32)  # Empty
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            return_value=mock_recorder,
        )

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)
        pipeline._recorder = mock_recorder
        pipeline._model_preload_complete.set()

        result = pipeline.stop_and_transcribe()

        # Should fail gracefully, not crash
        assert result.success is False
        assert "audio" in result.error.lower() or "short" in result.error.lower()

    def test_recorder_cleanup_after_exception(self, mocker, reset_mode_manager):
        """Test resource management: pipeline handles exceptions gracefully."""
        mock_recorder = MagicMock()
        mock_recorder.is_recording = True
        mock_recorder.get_whisper_audio.side_effect = Exception("Hardware error")
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            return_value=mock_recorder,
        )

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)
        pipeline._recorder = mock_recorder
        pipeline._model_preload_complete.set()

        result = pipeline.stop_and_transcribe()

        # Should fail but not crash
        assert result.success is False
        # Error should be captured in result
        assert "error" in result.error.lower() or len(result.error) > 0

    def test_external_executor_not_shutdown(self, mocker, reset_mode_manager, sync_executor):
        """Test resource management: external executor not shut down by pipeline."""
        mock_recorder = MagicMock()
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            return_value=mock_recorder,
        )

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(
            mode_manager=mode_manager,
            executor=sync_executor,
        )

        # Track if shutdown was called
        original_shutdown = sync_executor.shutdown
        shutdown_called = [False]

        def track_shutdown(*args, **kwargs):
            shutdown_called[0] = True
            return original_shutdown(*args, **kwargs)

        sync_executor.shutdown = track_shutdown

        pipeline.shutdown(timeout=0.1)

        # External executor should NOT be shut down by pipeline
        assert shutdown_called[0] is False


class TestStreamingMode:
    """Tests for streaming transcription mode."""

    def test_streaming_fallback_when_transcriber_not_ready(
        self, mocker, reset_mode_manager
    ):
        """Test graceful fallback to batch mode when transcriber unavailable."""
        # Mock complete config structure
        mock_config = {
            "streaming": {"enabled": True, "vad_threshold": 0.5},
            "model": {"name": "tiny", "device": "cpu", "compute_type": "int8"},
            "hotkeys": {"mode": "push-to-talk"},
            "output": {"auto_paste": False},
            "context": {
                "coding_apps": [],
                "planning_apps": [],
                "coding_keywords": [],
                "planning_keywords": []
            },
        }
        mocker.patch("localwispr.config.get_config", return_value=mock_config)

        # Mock transcriber loading to fail
        mocker.patch("localwispr.transcribe.WhisperTranscriber", side_effect=Exception("Model loading failed"))

        # Mock recorder
        mock_recorder = MagicMock()
        mock_recorder.is_recording = False
        mock_recorder.start_recording.side_effect = lambda: setattr(mock_recorder, 'is_recording', True)
        mocker.patch("localwispr.audio.AudioRecorder", return_value=mock_recorder)

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(mode_manager=mode_manager)

        # Mark preload as complete but with error (simulates failed preload)
        pipeline._model_preload_error = Exception("Model loading failed")
        pipeline._model_preload_complete.set()

        # Attempt to start recording (triggers _init_streaming_transcriber)
        # This should fall back to batch mode when transcriber can't be loaded
        result = pipeline.start_recording()

        # Should succeed but fall back to batch mode
        assert result is True
        assert pipeline._streaming_enabled is False

    def test_streaming_fallback_preserves_recording_capability(
        self, mocker, mock_audio_recorder, mock_whisper_transcriber, reset_mode_manager
    ):
        """Test recording continues in batch mode after streaming fallback."""
        # Mock complete config structure
        mock_config = {
            "streaming": {"enabled": False},  # Disabled to use batch mode
            "model": {"name": "tiny", "device": "cpu", "compute_type": "int8"},
            "hotkeys": {"mode": "push-to-talk"},
            "output": {"auto_paste": False},
            "context": {
                "coding_apps": [],
                "planning_apps": [],
                "coding_keywords": [],
                "planning_keywords": []
            },
        }
        mocker.patch("localwispr.config.get_config", return_value=mock_config)
        mocker.patch("localwispr.modes.get_mode_prompt", return_value="test prompt")

        from localwispr.modes import ModeManager, ModeType
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        mode_manager.set_mode(ModeType.CODE)
        pipeline = RecordingPipeline(mode_manager=mode_manager)

        # Model preload succeeds (for batch mode)
        pipeline._model_preload_complete.set()

        # Start and stop recording
        pipeline.start_recording()
        result = pipeline.stop_and_transcribe()

        # Should work in batch mode
        assert result.success is True
        assert result.text == "test transcription"

    def test_stop_and_transcribe_async_streaming_mode(
        self, mocker, reset_mode_manager, sync_executor
    ):
        """Test that async path handles streaming mode correctly."""
        # Mock recorder
        mock_recorder = MagicMock()
        mock_recorder.is_recording = True
        mocker.patch(
            "localwispr.audio.AudioRecorder",
            return_value=mock_recorder,
        )

        # Mock streaming transcriber with finalize result
        mock_streaming_result = MagicMock()
        mock_streaming_result.text = "streaming result"
        mock_streaming_result.audio_duration = 2.0
        mock_streaming_result.total_inference_time = 0.5
        mock_streaming_result.num_segments = 2

        mock_streaming = MagicMock()
        mock_streaming.finalize.return_value = mock_streaming_result

        # Mock batch transcriber (should NOT be called)
        mock_transcriber = MagicMock()
        mocker.patch(
            "localwispr.transcribe.WhisperTranscriber",
            return_value=mock_transcriber,
        )

        from localwispr.modes import ModeManager
        from localwispr.pipeline import RecordingPipeline

        mode_manager = ModeManager()
        pipeline = RecordingPipeline(
            mode_manager=mode_manager,
            executor=sync_executor,
        )
        pipeline._recorder = mock_recorder
        pipeline._model_preload_complete.set()

        # Enable streaming mode
        pipeline._streaming_enabled = True
        pipeline._streaming_transcriber = mock_streaming

        # Callback tracking
        results = []
        completed = []

        # Call async version
        gen = pipeline.stop_and_transcribe_async(
            on_result=lambda r, g: results.append(r),
            on_complete=lambda g: completed.append(True),
        )

        # Should have called streaming finalize, NOT batch processing
        mock_streaming.finalize.assert_called_once()

        # Should NOT have called get_recorded_audio for batch processing
        mock_recorder.get_whisper_audio.assert_not_called()

        # Callbacks should be invoked with streaming result
        assert len(results) == 1
        assert len(completed) == 1

        # Result should have streaming text
        result = results[0]
        assert result.text == "streaming result"
        assert result.success is True
        assert result.audio_duration == 2.0
        assert result.inference_time == 0.5


