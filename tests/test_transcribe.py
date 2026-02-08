"""Tests for the transcription module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.helpers import MockSegment, MockTranscriptionInfo


class TestIsModelDownloaded:
    """Tests for is_model_downloaded() with faster-whisper backend."""

    def test_ct2_cached_returns_true(self, mocker):
        """When CT2 model is in HuggingFace cache, returns True."""
        mocker.patch(
            "localwispr.transcribe.model_manager.get_active_backend",
            return_value="faster-whisper",
        )
        mocker.patch(
            "localwispr.transcribe.model_manager._is_model_downloaded_faster_whisper",
            return_value=True,
        )
        from localwispr.transcribe.model_manager import is_model_downloaded

        assert is_model_downloaded("large-v3") is True

    def test_ggml_exists_ct2_not_cached_returns_true(self, mocker):
        """When GGML exists but CT2 is not cached, returns True (auto-download)."""
        mocker.patch(
            "localwispr.transcribe.model_manager.get_active_backend",
            return_value="faster-whisper",
        )
        mocker.patch(
            "localwispr.transcribe.model_manager._is_model_downloaded_faster_whisper",
            return_value=False,
        )
        mocker.patch(
            "localwispr.transcribe.model_manager._is_model_downloaded_ggml",
            return_value=True,
        )
        from localwispr.transcribe.model_manager import is_model_downloaded

        assert is_model_downloaded("large-v3") is True

    def test_neither_format_exists_returns_false(self, mocker):
        """When neither CT2 nor GGML exists, returns False."""
        mocker.patch(
            "localwispr.transcribe.model_manager.get_active_backend",
            return_value="faster-whisper",
        )
        mocker.patch(
            "localwispr.transcribe.model_manager._is_model_downloaded_faster_whisper",
            return_value=False,
        )
        mocker.patch(
            "localwispr.transcribe.model_manager._is_model_downloaded_ggml",
            return_value=False,
        )
        from localwispr.transcribe.model_manager import is_model_downloaded

        assert is_model_downloaded("large-v3") is False


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_transcription_result_creation(self):
        """Test that TranscriptionResult can be created with required fields."""
        from localwispr.transcribe.transcriber import TranscriptionResult

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
        from localwispr.transcribe.context import ContextType
        from localwispr.transcribe.transcriber import TranscriptionResult

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
        mocker.patch("localwispr.transcribe.transcriber.get_config", return_value=mock_config)

        from localwispr.transcribe.transcriber import WhisperTranscriber

        transcriber = WhisperTranscriber()

        assert transcriber.model_name == "tiny"
        assert transcriber.device == "cpu"
        assert transcriber.language is None  # "auto" becomes None
        assert transcriber.is_loaded is False

    def test_transcriber_initialization_with_custom_args(self, mocker, mock_config):
        """Test transcriber uses provided arguments over config."""
        mocker.patch("localwispr.transcribe.transcriber.get_config", return_value=mock_config)
        # Mock CUDA as available so device="cuda" works
        mocker.patch("localwispr.transcribe.gpu.check_cuda_available", return_value=True)

        from localwispr.transcribe.transcriber import WhisperTranscriber

        transcriber = WhisperTranscriber(
            model_name="large-v3",
            device="cuda",
            compute_type="float16",
        )

        assert transcriber.model_name == "large-v3"
        assert transcriber.device == "cuda"

    def test_transcriber_lazy_model_loading(self, mocker, mock_config):
        """Test that model is not loaded until accessed."""
        mocker.patch("localwispr.transcribe.transcriber.get_config", return_value=mock_config)
        mocker.patch("localwispr.transcribe.model_manager.is_model_downloaded", return_value=True)

        from localwispr.transcribe.transcriber import WhisperTranscriber, _FASTER_WHISPER_AVAILABLE

        if _FASTER_WHISPER_AVAILABLE:
            mock_model = mocker.patch("faster_whisper.WhisperModel")
        else:
            mocker.patch("localwispr.transcribe.model_manager.get_model_path", return_value="/fake/path/model.bin")
            mock_model = mocker.patch("pywhispercpp.model.Model")

        transcriber = WhisperTranscriber()

        # Model should not be loaded yet
        assert transcriber.is_loaded is False
        mock_model.assert_not_called()

        # Access model property triggers load
        _ = transcriber.model

        assert transcriber.is_loaded is True
        mock_model.assert_called_once()

    def test_transcriber_hotwords_from_config(self, mocker, mock_config):
        """Test that hotwords are loaded from config."""
        mocker.patch("localwispr.transcribe.transcriber.get_config", return_value=mock_config)

        from localwispr.transcribe.transcriber import WhisperTranscriber

        transcriber = WhisperTranscriber()

        assert transcriber.hotwords == ["LocalWispr", "pytest"]

    def test_transcriber_language_explicit(self, mocker, mock_config):
        """Test explicit language setting."""
        mock_config["model"]["language"] = "en"
        mocker.patch("localwispr.transcribe.transcriber.get_config", return_value=mock_config)

        from localwispr.transcribe.transcriber import WhisperTranscriber

        transcriber = WhisperTranscriber()

        assert transcriber.language == "en"

    def test_transcribe_returns_result(self, mocker, mock_config, mock_audio_data):
        """Test transcribe method returns TranscriptionResult."""
        mocker.patch("localwispr.transcribe.transcriber.get_config", return_value=mock_config)
        mocker.patch("localwispr.transcribe.model_manager.is_model_downloaded", return_value=True)

        from localwispr.transcribe.transcriber import WhisperTranscriber, _FASTER_WHISPER_AVAILABLE

        mock_model = MagicMock()
        segments = [MockSegment(text=" hello world", t0=0, t1=100)]

        if _FASTER_WHISPER_AVAILABLE:
            # faster-whisper returns (segments_iter, info)
            mock_model.transcribe.return_value = (iter(segments), MockTranscriptionInfo())
            mock_model_class = mocker.patch("faster_whisper.WhisperModel")
        else:
            mock_model.transcribe.return_value = segments
            mocker.patch("localwispr.transcribe.model_manager.get_model_path", return_value="/fake/path/model.bin")
            mock_model_class = mocker.patch("pywhispercpp.model.Model")

        mock_model_class.return_value = mock_model

        transcriber = WhisperTranscriber()
        result = transcriber.transcribe(mock_audio_data)

        assert result.text == "hello world"
        assert len(result.segments) == 1
        assert result.audio_duration == 1.0  # 16000 samples / 16000 Hz
        assert result.inference_time > 0

    def test_transcribe_with_initial_prompt(self, mocker, mock_config, mock_audio_data):
        """Test transcribe passes initial prompt."""
        mocker.patch("localwispr.transcribe.transcriber.get_config", return_value=mock_config)
        mocker.patch("localwispr.transcribe.model_manager.is_model_downloaded", return_value=True)

        from localwispr.transcribe.transcriber import WhisperTranscriber, _FASTER_WHISPER_AVAILABLE

        mock_model = MagicMock()

        if _FASTER_WHISPER_AVAILABLE:
            mock_model.transcribe.return_value = (iter([]), MockTranscriptionInfo())
            mock_model_class = mocker.patch("faster_whisper.WhisperModel")
        else:
            mock_model.transcribe.return_value = []
            mocker.patch("localwispr.transcribe.model_manager.get_model_path", return_value="/fake/path/model.bin")
            mock_model_class = mocker.patch("pywhispercpp.model.Model")

        mock_model_class.return_value = mock_model

        transcriber = WhisperTranscriber()
        # Clear hotwords for this test
        transcriber._hotwords = []
        transcriber.transcribe(mock_audio_data, initial_prompt="coding context")

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["initial_prompt"] == "coding context"

    def test_transcribe_includes_hotwords(self, mocker, mock_config, mock_audio_data):
        """Test that hotwords are passed to model.transcribe."""
        mocker.patch("localwispr.transcribe.transcriber.get_config", return_value=mock_config)
        mocker.patch("localwispr.transcribe.model_manager.is_model_downloaded", return_value=True)

        from localwispr.transcribe.transcriber import WhisperTranscriber, _FASTER_WHISPER_AVAILABLE

        mock_model = MagicMock()

        if _FASTER_WHISPER_AVAILABLE:
            mock_model.transcribe.return_value = (iter([]), MockTranscriptionInfo())
            mock_model_class = mocker.patch("faster_whisper.WhisperModel")
        else:
            mock_model.transcribe.return_value = []
            mocker.patch("localwispr.transcribe.model_manager.get_model_path", return_value="/fake/path/model.bin")
            mock_model_class = mocker.patch("pywhispercpp.model.Model")

        mock_model_class.return_value = mock_model

        transcriber = WhisperTranscriber()
        transcriber.transcribe(mock_audio_data)

        call_kwargs = mock_model.transcribe.call_args[1]
        if _FASTER_WHISPER_AVAILABLE:
            # faster-whisper gets hotwords as a separate kwarg
            assert call_kwargs.get("hotwords") == "LocalWispr pytest"
            assert call_kwargs.get("initial_prompt") == "LocalWispr pytest"
        else:
            assert "initial_prompt" in call_kwargs
            assert "LocalWispr pytest" == call_kwargs["initial_prompt"]


class TestTranscribeWithContext:
    """Tests for transcribe_with_context function."""

    def test_transcribe_with_context_uses_window_detection(
        self, mocker, mock_config, mock_audio_data
    ):
        """Test that context detection uses window title."""
        mocker.patch("localwispr.transcribe.transcriber.get_config", return_value=mock_config)

        # Mock window detection
        mock_detector = MagicMock()
        from localwispr.transcribe.context import ContextType

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
        mocker.patch("localwispr.transcribe.transcriber.load_prompt", return_value="coding prompt")

        from localwispr.transcribe.transcriber import transcribe_with_context

        result = transcribe_with_context(mock_audio_data, mock_transcriber, mock_detector)

        mock_detector.detect_from_window.assert_called_once()
        assert result.detected_context == ContextType.CODING

    def test_transcribe_with_context_retranscribes_on_mismatch(
        self, mocker, mock_config, mock_audio_data
    ):
        """Test retranscription when pre and post contexts differ."""
        mocker.patch("localwispr.transcribe.transcriber.get_config", return_value=mock_config)

        from localwispr.transcribe.context import ContextType

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

        mocker.patch("localwispr.transcribe.transcriber.load_prompt", return_value="prompt")

        from localwispr.transcribe.transcriber import transcribe_with_context

        result = transcribe_with_context(mock_audio_data, mock_transcriber, mock_detector)

        # Should call transcribe twice (initial + retranscribe)
        assert mock_transcriber.transcribe.call_count == 2
        assert result.was_retranscribed is True

    def test_transcribe_with_context_no_retranscribe_on_general_post(
        self, mocker, mock_config, mock_audio_data
    ):
        """Test no retranscription when post-detection is GENERAL."""
        mocker.patch("localwispr.transcribe.transcriber.get_config", return_value=mock_config)

        from localwispr.transcribe.context import ContextType

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

        mocker.patch("localwispr.transcribe.transcriber.load_prompt", return_value="prompt")

        from localwispr.transcribe.transcriber import transcribe_with_context

        result = transcribe_with_context(mock_audio_data, mock_transcriber, mock_detector)

        # Should only call transcribe once
        assert mock_transcriber.transcribe.call_count == 1
        assert result.was_retranscribed is False


class TestTranscribeRecording:
    """Tests for transcribe_recording convenience function."""

    def test_transcribe_recording_stops_recorder(self, mocker, mock_config, mock_audio_data):
        """Test that transcribe_recording stops the recorder and gets audio."""
        mocker.patch("localwispr.transcribe.transcriber.get_config", return_value=mock_config)

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

        from localwispr.transcribe.transcriber import transcribe_recording

        result = transcribe_recording(
            mock_recorder, mock_transcriber, use_context=False
        )

        mock_recorder.get_whisper_audio.assert_called_once()
        assert result.text == "test"


class TestCudaDllLoading:
    """Tests that CUDA DLLs are findable for GPU inference."""

    def test_cublas_dll_exists_in_nvidia_package(self):
        """cublas64_12.dll must exist in the nvidia.cublas package."""
        import importlib.util
        import os

        spec = importlib.util.find_spec("nvidia.cublas")
        if not spec or not spec.submodule_search_locations:
            pytest.skip("nvidia.cublas not installed")

        pkg_dir = list(spec.submodule_search_locations)[0]
        dll_path = os.path.join(pkg_dir, "bin", "cublas64_12.dll")
        assert os.path.isfile(dll_path), f"cublas64_12.dll not at {dll_path}"

    def test_cublas_dll_loadable_via_ctypes(self):
        """cublas64_12.dll must be loadable via ctypes when directory is on PATH."""
        import ctypes
        import importlib.util
        import os

        spec = importlib.util.find_spec("nvidia.cublas")
        if not spec or not spec.submodule_search_locations:
            pytest.skip("nvidia.cublas not installed")

        pkg_dir = list(spec.submodule_search_locations)[0]
        dll_path = os.path.join(pkg_dir, "bin", "cublas64_12.dll")
        if not os.path.isfile(dll_path):
            pytest.skip("cublas64_12.dll not found")

        lib = ctypes.WinDLL(dll_path)
        assert lib is not None

    def test_cudnn_dll_exists_in_ctranslate2(self):
        """cudnn64_9.dll must exist in the ctranslate2 package."""
        import os
        import pathlib

        try:
            import ctranslate2
        except ImportError:
            pytest.skip("ctranslate2 not installed")

        ct2_dir = pathlib.Path(ctranslate2.__file__).parent
        dll_path = ct2_dir / "cudnn64_9.dll"
        assert dll_path.is_file(), f"cudnn64_9.dll not at {dll_path}"

    def test_ctranslate2_finds_cublas_when_on_path(self):
        """ctranslate2 can load cublas when nvidia.cublas/bin is on PATH."""
        import importlib.util
        import os

        spec = importlib.util.find_spec("nvidia.cublas")
        if not spec or not spec.submodule_search_locations:
            pytest.skip("nvidia.cublas not installed")

        pkg_dir = list(spec.submodule_search_locations)[0]
        bin_dir = os.path.join(pkg_dir, "bin")

        # Prepend to PATH (same approach as __main__.py uses for frozen builds)
        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = bin_dir + os.pathsep + old_path
        try:
            import ctranslate2
            # get_cuda_device_count uses cublas internally — if it doesn't
            # raise RuntimeError about missing DLLs, the DLLs are loaded
            try:
                count = ctranslate2.get_cuda_device_count()
                assert count >= 0  # 0 is fine (no GPU), just no DLL error
            except RuntimeError as e:
                if "cublas" in str(e).lower() or "library" in str(e).lower():
                    pytest.fail(f"cublas DLL still not found even with PATH set: {e}")
                raise
        finally:
            os.environ["PATH"] = old_path
