"""Shared test helper classes and utilities.

This module contains classes and utilities that need to be imported
directly in test files (as opposed to pytest fixtures which are
auto-injected).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MockSegment:
    """Mock transcription segment compatible with both backends.

    pywhispercpp Segment has: t0, t1, text (centiseconds)
    faster-whisper Segment has: start, end, text (seconds)
    This class supports both via properties.
    """

    text: str
    t0: int = 0  # Start time in centiseconds
    t1: int = 100  # End time in centiseconds
    probability: float = 0.95

    @property
    def start(self) -> float:
        return self.t0 / 100.0

    @property
    def end(self) -> float:
        return self.t1 / 100.0


@dataclass
class MockTranscriptionInfo:
    """Mock Whisper transcription info (faster-whisper info object)."""

    language: str = "en"
    language_probability: float = 0.99
