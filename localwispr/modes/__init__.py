"""Transcription modes for LocalWispr.

This package provides user-facing named transcription modes, inspired by HyperWhisper.
Modes are user-friendly wrappers around the existing context system with:
- Named modes (Code, Notes, Dictation, Email, Chat)
- Persistence across recordings
- Mode cycling via hotkey
- Manual override with auto-reset

Modes map to prompt files that help Whisper recognize domain-specific vocabulary.

This module re-exports all public symbols from the submodules for backward
compatibility. Existing imports like `from localwispr.modes import ModeType`
will continue to work.
"""

# Re-export from definitions
from localwispr.modes.definitions import (
    Mode,
    ModeType,
    MODES,
    MODE_CYCLE_ORDER,
)

# Re-export from manager
from localwispr.modes.manager import (
    ModeManager,
    get_mode_manager,
    get_current_mode,
    set_mode,
    cycle_mode,
    get_mode_prompt,
    get_all_modes,
    get_mode_by_name,
)

__all__ = [
    # Definitions
    "Mode",
    "ModeType",
    "MODES",
    "MODE_CYCLE_ORDER",
    # Manager
    "ModeManager",
    "get_mode_manager",
    "get_current_mode",
    "set_mode",
    "cycle_mode",
    "get_mode_prompt",
    "get_all_modes",
    "get_mode_by_name",
]
