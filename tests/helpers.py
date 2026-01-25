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


class SettingsFlowHelper:
    """Helper for testing settings propagation workflows.

    Provides utilities for:
    - Spying on handler execution
    - Asserting config propagation
    - Capturing settings change events
    """

    def __init__(self, mocker):
        """Initialize helper.

        Args:
            mocker: pytest-mock mocker fixture
        """
        self.mocker = mocker
        self._handler_calls = []

    def spy_on_handler(self, handler_name: str):
        """Create a spy for a settings handler.

        Args:
            handler_name: Name of the handler to spy on

        Returns:
            Mock spy that can be asserted on
        """
        spy = self.mocker.MagicMock()
        self._handler_calls.append((handler_name, spy))
        return spy

    def assert_config_propagated(self, app, expected_config_values: dict):
        """Assert that config values propagated to components.

        Args:
            app: TrayApp instance
            expected_config_values: Dict of expected config values to verify
        """
        from localwispr.config import get_config

        current_config = get_config()

        for key_path, expected_value in expected_config_values.items():
            # Navigate nested dict
            keys = key_path.split(".")
            value = current_config
            for key in keys:
                value = value[key]

            assert value == expected_value, f"Config {key_path} = {value}, expected {expected_value}"

    def get_handler_calls(self):
        """Get all handler calls recorded.

        Returns:
            List of (handler_name, spy) tuples
        """
        return self._handler_calls
