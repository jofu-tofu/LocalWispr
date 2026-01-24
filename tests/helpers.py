"""Shared test helper classes and utilities.

This module contains classes and utilities that need to be imported
directly in test files (as opposed to pytest fixtures which are
auto-injected).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MockSegment:
    """Mock Whisper transcription segment."""

    text: str
    start: float = 0.0
    end: float = 1.0


@dataclass
class MockTranscriptionInfo:
    """Mock Whisper transcription info."""

    language: str = "en"
    language_probability: float = 0.99
