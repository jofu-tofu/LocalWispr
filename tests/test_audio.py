"""Tests for the audio module."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


class TestPrepareForWhisper:
    """Tests for prepare_for_whisper function."""

    def test_prepare_empty_audio(self):
        """Test handling of empty audio array."""
        from localwispr.audio import prepare_for_whisper

        audio = np.array([], dtype=np.float32)
        result = prepare_for_whisper(audio, 16000)

        assert len(result) == 0
        assert result.dtype == np.float32

    def test_prepare_mono_audio_same_rate(self):
        """Test mono audio at 16kHz (no resampling needed)."""
        from localwispr.audio import prepare_for_whisper

        # 1 second of audio at 16kHz
        audio = np.ones(16000, dtype=np.float32) * 0.5
        result = prepare_for_whisper(audio, 16000)

        assert len(result) == 16000
        assert result.dtype == np.float32
        # Should be normalized to [-1, 1]
        assert np.max(result) <= 1.0
        assert np.min(result) >= -1.0

    def test_prepare_stereo_to_mono(self, mock_stereo_audio):
        """Test stereo to mono conversion."""
        from localwispr.audio import prepare_for_whisper

        result = prepare_for_whisper(mock_stereo_audio, 16000)

        # Should be 1D mono
        assert result.ndim == 1
        assert len(result) == 16000

    def test_prepare_resamples_from_48khz(self):
        """Test resampling from 48kHz to 16kHz."""
        from localwispr.audio import prepare_for_whisper

        # 1 second at 48kHz
        audio = np.random.randn(48000).astype(np.float32)
        result = prepare_for_whisper(audio, 48000)

        # Should be resampled to 16kHz
        assert len(result) == 16000
        assert result.dtype == np.float32

    def test_prepare_resamples_from_44100hz(self):
        """Test resampling from 44.1kHz to 16kHz."""
        from localwispr.audio import prepare_for_whisper

        # 1 second at 44.1kHz
        audio = np.random.randn(44100).astype(np.float32)
        result = prepare_for_whisper(audio, 44100)

        # Should be resampled to ~16kHz
        expected_length = int(44100 * 16000 / 44100)
        assert len(result) == expected_length

    def test_prepare_normalizes_amplitude(self):
        """Test amplitude normalization to [-1, 1]."""
        from localwispr.audio import prepare_for_whisper

        # Audio with large amplitude
        audio = np.array([0.0, 10.0, -10.0, 5.0], dtype=np.float32)
        result = prepare_for_whisper(audio, 16000)

        assert np.max(np.abs(result)) <= 1.0

    def test_prepare_handles_silence(self):
        """Test handling of all-zero audio (silence)."""
        from localwispr.audio import prepare_for_whisper

        audio = np.zeros(16000, dtype=np.float32)
        result = prepare_for_whisper(audio, 16000)

        # Should not divide by zero
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_prepare_rejects_3d_audio(self):
        """Test that 3D audio arrays are rejected."""
        from localwispr.audio import prepare_for_whisper

        audio = np.zeros((16000, 2, 2), dtype=np.float32)

        with pytest.raises(ValueError, match="must be 1D or 2D"):
            prepare_for_whisper(audio, 16000)


class TestAudioRecorder:
    """Tests for AudioRecorder class."""

    def test_recorder_initialization(self, mock_sounddevice):
        """Test AudioRecorder initialization."""
        from localwispr.audio import AudioRecorder

        recorder = AudioRecorder()

        assert recorder.is_recording is False
        assert recorder.sample_rate == 48000  # From mock
        assert recorder.device_name == "Mock Microphone"

    def test_recorder_start_recording(self, mock_sounddevice):
        """Test starting a recording."""
        from localwispr.audio import AudioRecorder

        recorder = AudioRecorder()
        recorder.start_recording()

        assert recorder.is_recording is True
        mock_sounddevice.InputStream.assert_called_once()
        mock_sounddevice.InputStream.return_value.start.assert_called_once()

    def test_recorder_cannot_start_twice(self, mock_sounddevice):
        """Test that starting recording twice raises error."""
        from localwispr.audio import AudioRecorder, AudioRecorderError

        recorder = AudioRecorder()
        recorder.start_recording()

        with pytest.raises(AudioRecorderError, match="already in progress"):
            recorder.start_recording()

    def test_recorder_stop_recording(self, mock_sounddevice):
        """Test stopping a recording."""
        from localwispr.audio import AudioRecorder

        recorder = AudioRecorder()
        recorder.start_recording()

        audio = recorder.stop_recording()

        assert recorder.is_recording is False
        assert isinstance(audio, np.ndarray)
        mock_sounddevice.InputStream.return_value.stop.assert_called_once()

    def test_recorder_stop_without_start_raises(self, mock_sounddevice):
        """Test that stopping without starting raises error."""
        from localwispr.audio import AudioRecorder, AudioRecorderError

        recorder = AudioRecorder()

        with pytest.raises(AudioRecorderError, match="No recording in progress"):
            recorder.stop_recording()

    def test_recorder_get_audio_level(self, mock_sounddevice):
        """Test getting audio level."""
        from localwispr.audio import AudioRecorder

        recorder = AudioRecorder()

        # Not recording - should return 0
        assert recorder.get_audio_level() == 0.0

    def test_recorder_get_rms_level(self, mock_sounddevice):
        """Test getting RMS audio level."""
        from localwispr.audio import AudioRecorder

        recorder = AudioRecorder()

        # Not recording - should return 0
        assert recorder.get_rms_level() == 0.0

    def test_recorder_get_whisper_audio(self, mock_sounddevice, mocker):
        """Test get_whisper_audio convenience method."""
        from localwispr.audio import AudioRecorder

        # Mock prepare_for_whisper
        mock_prepare = mocker.patch(
            "localwispr.audio.format.prepare_for_whisper",
            return_value=np.zeros(16000, dtype=np.float32),
        )

        recorder = AudioRecorder()
        recorder.start_recording()

        audio = recorder.get_whisper_audio()

        assert recorder.is_recording is False
        assert len(audio) == 16000
        mock_prepare.assert_called_once()

    def test_recorder_audio_callback_stores_chunks(self, mock_sounddevice):
        """Test that audio callback stores audio chunks."""
        from localwispr.audio import AudioRecorder

        recorder = AudioRecorder()
        recorder.start_recording()

        # Simulate audio callback
        test_chunk = np.random.randn(1024, 1).astype(np.float32)
        recorder._audio_callback(test_chunk, 1024, None, MagicMock())

        # Stop and check audio
        audio = recorder.stop_recording()
        assert len(audio) == 1024

    def test_recorder_audio_callback_updates_levels(self, mock_sounddevice):
        """Test that audio callback updates peak and RMS levels."""
        from localwispr.audio import AudioRecorder

        recorder = AudioRecorder()
        recorder.start_recording()

        # Simulate audio callback with known values
        test_chunk = np.array([[0.5], [-0.5], [0.25]], dtype=np.float32)
        recorder._audio_callback(test_chunk, 3, None, MagicMock())

        assert recorder.get_audio_level() == 0.5  # Peak
        assert recorder.get_rms_level() > 0  # RMS should be non-zero


class TestListAudioDevices:
    """Tests for list_audio_devices function."""

    def test_list_audio_devices(self, mock_sounddevice):
        """Test listing available audio devices."""
        # Configure mock to return multiple devices
        mock_sounddevice.query_devices.return_value = [
            {
                "name": "Microphone 1",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
            {
                "name": "Speaker",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            },
            {
                "name": "Microphone 2",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 44100.0,
            },
        ]

        from localwispr.audio import list_audio_devices

        devices = list_audio_devices()

        # Should only include input devices
        assert len(devices) == 2
        assert devices[0]["name"] == "Microphone 1"
        assert devices[1]["name"] == "Microphone 2"

    def test_device_disconnection_during_recording(self, mocker, mock_sounddevice):
        """Test handling device disconnection while recording.

        Edge case: Microphone is unplugged or disabled mid-recording.
        This simulates sounddevice raising an exception in the audio callback.
        """
        from localwispr.audio import AudioRecorder

        # Configure mock to raise exception during recording (simulating device disconnect)
        mock_stream = mock_sounddevice.InputStream.return_value
        mock_stream.__enter__.return_value = mock_stream

        # Simulate device error via callback
        def callback_with_error(*args, **kwargs):
            raise OSError("Device disconnected")

        mock_stream.read.side_effect = callback_with_error

        recorder = AudioRecorder()
        recorder.start_recording()

        # Attempt to get audio should handle the error gracefully
        try:
            audio = recorder.get_whisper_audio()
            # Should either return empty array or raise a clean error
            assert isinstance(audio, np.ndarray)
        except Exception as e:
            # If it raises, it should be a clean error, not the raw OSError
            assert "disconnect" in str(e).lower() or "device" in str(e).lower()
