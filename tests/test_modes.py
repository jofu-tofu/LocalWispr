"""Tests for the modes module."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest


class TestModeType:
    """Tests for ModeType enum."""

    def test_mode_type_values(self):
        """Test that ModeType has expected values."""
        from localwispr.modes import ModeType

        assert ModeType.CODE.value == "code"
        assert ModeType.NOTES.value == "notes"
        assert ModeType.DICTATION.value == "dictation"
        assert ModeType.EMAIL.value == "email"
        assert ModeType.CHAT.value == "chat"

    def test_mode_type_from_value(self):
        """Test creating ModeType from string value."""
        from localwispr.modes import ModeType

        assert ModeType("code") == ModeType.CODE
        assert ModeType("dictation") == ModeType.DICTATION


class TestMode:
    """Tests for Mode dataclass."""

    def test_mode_creation(self):
        """Test creating a Mode instance."""
        from localwispr.modes import Mode, ModeType

        mode = Mode(
            mode_type=ModeType.CODE,
            name="Code",
            description="Programming mode",
            prompt_file="coding",
            icon="</>",
        )

        assert mode.mode_type == ModeType.CODE
        assert mode.name == "Code"
        assert mode.description == "Programming mode"
        assert mode.prompt_file == "coding"
        assert mode.icon == "</>"

    def test_mode_load_prompt(self, mocker):
        """Test Mode.load_prompt method."""
        mock_load = mocker.patch(
            "localwispr.modes.definitions.load_prompt",
            return_value="test prompt content",
        )

        from localwispr.modes import Mode, ModeType

        mode = Mode(
            mode_type=ModeType.CODE,
            name="Code",
            description="Test",
            prompt_file="coding",
        )

        prompt = mode.load_prompt()

        assert prompt == "test prompt content"
        mock_load.assert_called_once_with("coding")


class TestMODES:
    """Tests for MODES dictionary."""

    def test_modes_contains_all_types(self):
        """Test that MODES contains all ModeType values."""
        from localwispr.modes import MODES, ModeType

        for mode_type in ModeType:
            assert mode_type in MODES

    def test_modes_have_correct_structure(self):
        """Test that all modes have required attributes."""
        from localwispr.modes import MODES

        for mode_type, mode in MODES.items():
            assert mode.mode_type == mode_type
            assert len(mode.name) > 0
            assert len(mode.description) > 0
            assert len(mode.prompt_file) > 0


class TestMODE_CYCLE_ORDER:
    """Tests for MODE_CYCLE_ORDER list."""

    def test_cycle_order_contains_all_types(self):
        """Test that MODE_CYCLE_ORDER contains all mode types."""
        from localwispr.modes import MODE_CYCLE_ORDER, ModeType

        for mode_type in ModeType:
            assert mode_type in MODE_CYCLE_ORDER

    def test_cycle_order_has_no_duplicates(self):
        """Test that MODE_CYCLE_ORDER has no duplicates."""
        from localwispr.modes import MODE_CYCLE_ORDER

        assert len(MODE_CYCLE_ORDER) == len(set(MODE_CYCLE_ORDER))


class TestModeManager:
    """Tests for ModeManager class."""

    def test_mode_manager_initialization(self, reset_mode_manager):
        """Test ModeManager initialization."""
        from localwispr.modes import ModeManager, ModeType

        manager = ModeManager()

        assert manager.current_mode_type == ModeType.DICTATION  # Default
        assert manager.is_manual_override is False

    def test_mode_manager_set_mode(self, reset_mode_manager):
        """Test setting mode."""
        from localwispr.modes import ModeManager, ModeType

        manager = ModeManager()
        result = manager.set_mode(ModeType.CODE)

        assert manager.current_mode_type == ModeType.CODE
        assert manager.is_manual_override is True
        assert result.mode_type == ModeType.CODE

    def test_mode_manager_cycle_mode(self, reset_mode_manager):
        """Test cycling through modes."""
        from localwispr.modes import MODE_CYCLE_ORDER, ModeManager, ModeType

        manager = ModeManager()
        manager.set_mode(ModeType.CODE)  # First in cycle

        # Cycle through all modes
        visited = [ModeType.CODE]
        for _ in range(len(MODE_CYCLE_ORDER) - 1):
            mode = manager.cycle_mode()
            visited.append(mode.mode_type)

        # Should have cycled through all modes
        assert len(visited) == len(MODE_CYCLE_ORDER)

        # Next cycle should wrap around
        mode = manager.cycle_mode()
        assert mode.mode_type == ModeType.CODE

    def test_mode_manager_reset_override(self, reset_mode_manager):
        """Test resetting manual override."""
        from localwispr.modes import ModeManager, ModeType

        manager = ModeManager()
        manager.set_mode(ModeType.CODE)

        assert manager.is_manual_override is True

        manager.reset_override()

        assert manager.is_manual_override is False

    def test_mode_manager_on_mode_change_callback(self, reset_mode_manager):
        """Test that mode change callback is invoked."""
        from localwispr.modes import ModeManager, ModeType

        callback = MagicMock()
        manager = ModeManager(on_mode_change=callback)

        manager.set_mode(ModeType.CODE)

        callback.assert_called_once()
        called_mode = callback.call_args[0][0]
        assert called_mode.mode_type == ModeType.CODE

    def test_mode_manager_callback_not_called_same_mode(self, reset_mode_manager):
        """Test that callback is not called when setting same mode."""
        from localwispr.modes import ModeManager, ModeType

        callback = MagicMock()
        manager = ModeManager(on_mode_change=callback)

        manager.set_mode(ModeType.CODE)
        callback.reset_mock()

        manager.set_mode(ModeType.CODE)

        callback.assert_not_called()

    def test_mode_manager_auto_reset(self, reset_mode_manager, mocker):
        """Test auto-reset after inactivity."""
        from localwispr.modes import ModeManager, ModeType

        manager = ModeManager(auto_reset=True)
        manager.AUTO_RESET_DELAY = 0.1  # Short delay for test

        manager.set_mode(ModeType.CODE)
        assert manager.is_manual_override is True

        # Wait for auto-reset
        time.sleep(0.2)

        # Access current_mode to trigger check
        _ = manager.current_mode

        assert manager.is_manual_override is False

    def test_mode_manager_record_activity(self, reset_mode_manager):
        """Test recording activity resets timer."""
        from localwispr.modes import ModeManager, ModeType

        manager = ModeManager(auto_reset=True)
        manager.AUTO_RESET_DELAY = 0.1

        manager.set_mode(ModeType.CODE)
        time.sleep(0.05)
        manager.record_activity()
        time.sleep(0.05)

        # Should not have reset yet due to activity
        _ = manager.current_mode
        assert manager.is_manual_override is True

    def test_mode_manager_get_prompt(self, reset_mode_manager, mocker):
        """Test getting prompt for current mode."""
        mocker.patch(
            "localwispr.modes.definitions.load_prompt",
            return_value="test prompt",
        )

        from localwispr.modes import ModeManager

        manager = ModeManager()
        prompt = manager.get_prompt()

        assert prompt == "test prompt"


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_mode_manager(self, reset_mode_manager):
        """Test get_mode_manager returns singleton."""
        from localwispr.modes import get_mode_manager

        manager1 = get_mode_manager()
        manager2 = get_mode_manager()

        assert manager1 is manager2

    def test_get_current_mode(self, reset_mode_manager):
        """Test get_current_mode function."""
        from localwispr.modes import ModeType, get_current_mode

        mode = get_current_mode()

        assert mode.mode_type == ModeType.DICTATION

    def test_set_mode_function(self, reset_mode_manager):
        """Test set_mode function."""
        from localwispr.modes import ModeType, get_current_mode, set_mode

        set_mode(ModeType.EMAIL)

        assert get_current_mode().mode_type == ModeType.EMAIL

    def test_cycle_mode_function(self, reset_mode_manager):
        """Test cycle_mode function."""
        from localwispr.modes import ModeType, cycle_mode, set_mode

        set_mode(ModeType.CODE)
        new_mode = cycle_mode()

        assert new_mode.mode_type != ModeType.CODE

    def test_get_mode_prompt_function(self, reset_mode_manager, mocker):
        """Test get_mode_prompt function."""
        mocker.patch(
            "localwispr.modes.definitions.load_prompt",
            return_value="prompt content",
        )

        from localwispr.modes import get_mode_prompt

        prompt = get_mode_prompt()

        assert prompt == "prompt content"

    def test_get_all_modes(self, reset_mode_manager):
        """Test get_all_modes function."""
        from localwispr.modes import MODE_CYCLE_ORDER, get_all_modes

        modes = get_all_modes()

        assert len(modes) == len(MODE_CYCLE_ORDER)
        for i, mode in enumerate(modes):
            assert mode.mode_type == MODE_CYCLE_ORDER[i]

    def test_get_mode_by_name(self, reset_mode_manager):
        """Test get_mode_by_name function."""
        from localwispr.modes import ModeType, get_mode_by_name

        mode = get_mode_by_name("Code")
        assert mode is not None
        assert mode.mode_type == ModeType.CODE

        # Case insensitive
        mode = get_mode_by_name("code")
        assert mode is not None
        assert mode.mode_type == ModeType.CODE

        # Non-existent
        mode = get_mode_by_name("NonExistent")
        assert mode is None
