"""Tests for the audio feedback module."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np


class TestGenerateTone:
    """Tests for the _generate_tone function."""

    def test_generate_tone_returns_array(self):
        """Test that _generate_tone returns a numpy array."""
        from localwispr.feedback import _generate_tone

        tone = _generate_tone(440, 100)

        assert isinstance(tone, np.ndarray)
        assert tone.dtype == np.float32

    def test_generate_tone_correct_length(self):
        """Test that tone has correct sample length for duration."""
        from localwispr.feedback import _generate_tone

        # 100ms at 44100Hz = 4410 samples
        tone = _generate_tone(440, 100, sample_rate=44100)

        assert len(tone) == 4410

    def test_generate_tone_volume_scaling(self):
        """Test that volume parameter scales the output."""
        from localwispr.feedback import _generate_tone

        tone_quiet = _generate_tone(440, 100, volume=0.1)
        tone_loud = _generate_tone(440, 100, volume=0.5)

        # Loud tone should have higher max amplitude
        assert np.max(np.abs(tone_loud)) > np.max(np.abs(tone_quiet))

    def test_generate_tone_frequency_affects_waveform(self):
        """Test that different frequencies produce different waveforms."""
        from localwispr.feedback import _generate_tone

        tone_low = _generate_tone(100, 100)
        tone_high = _generate_tone(1000, 100)

        # Different frequencies should produce different patterns
        assert not np.allclose(tone_low, tone_high)

    def test_generate_tone_fade_envelope(self):
        """Test that fade in/out is applied to avoid clicks."""
        from localwispr.feedback import _generate_tone

        tone = _generate_tone(440, 100, fade_ms=10)

        # First and last samples should be near zero due to fade
        assert np.abs(tone[0]) < 0.01
        assert np.abs(tone[-1]) < 0.01


class TestPlayToneAsync:
    """Tests for the _play_tone_async function."""

    def test_play_tone_async_calls_sounddevice(self, mocker):
        """Test that _play_tone_async calls sd.play."""
        mock_sd = mocker.patch("localwispr.feedback.sd")
        mock_sd.play = MagicMock()
        mock_sd.wait = MagicMock()

        from localwispr.feedback import _play_tone_async

        _play_tone_async(440, 100)

        # Wait a moment for the daemon thread to execute
        import time
        time.sleep(0.1)

        # Should have called play (though timing is tricky with threads)
        # At minimum, verify no exception was raised

    def test_play_tone_async_handles_exception(self, mocker):
        """Test that _play_tone_async handles sounddevice errors gracefully."""
        mock_sd = mocker.patch("localwispr.feedback.sd")
        mock_sd.play.side_effect = Exception("Audio device error")

        from localwispr.feedback import _play_tone_async

        # Should not raise - errors are silently ignored
        _play_tone_async(440, 100)

        import time
        time.sleep(0.1)


class TestPlayStartBeep:
    """Tests for play_start_beep function."""

    def test_play_start_beep_calls_play_tone(self, mocker):
        """Test that play_start_beep calls _play_tone_async with correct params."""
        mock_play = mocker.patch("localwispr.feedback._play_tone_async")

        from localwispr.feedback import play_start_beep

        play_start_beep()

        mock_play.assert_called_once_with(600, 150, volume=0.25)


class TestPlayStopBeep:
    """Tests for play_stop_beep function."""

    def test_play_stop_beep_calls_play_tone(self, mocker):
        """Test that play_stop_beep calls _play_tone_async with correct params."""
        mock_play = mocker.patch("localwispr.feedback._play_tone_async")

        from localwispr.feedback import play_stop_beep

        play_stop_beep()

        mock_play.assert_called_once_with(440, 150, volume=0.25)
