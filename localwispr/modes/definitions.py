"""Mode definitions for LocalWispr.

This module contains the core mode data structures:
- ModeType: Enum of available transcription modes
- Mode: Dataclass defining a transcription mode
- MODES: Dictionary mapping ModeType to Mode
- MODE_CYCLE_ORDER: List defining the order for mode cycling
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from localwispr.prompts import load_prompt


class ModeType(Enum):
    """Available transcription modes."""

    CODE = "code"
    NOTES = "notes"
    DICTATION = "dictation"  # Maps to "general" prompt
    EMAIL = "email"
    CHAT = "chat"


@dataclass
class Mode:
    """Transcription mode definition.

    Attributes:
        mode_type: The ModeType enum value.
        name: Display name for the mode.
        description: Brief description of when to use this mode.
        prompt_file: Name of the prompt file (without .txt extension).
        icon: Optional icon/emoji for display.
    """

    mode_type: ModeType
    name: str
    description: str
    prompt_file: str
    icon: str = ""

    def load_prompt(self) -> str:
        """Load the prompt text for this mode.

        Returns:
            The prompt text content.
        """
        return load_prompt(self.prompt_file)


# Define available modes
MODES: dict[ModeType, Mode] = {
    ModeType.CODE: Mode(
        mode_type=ModeType.CODE,
        name="Code",
        description="Programming and technical dictation",
        prompt_file="coding",
        icon="</>",
    ),
    ModeType.NOTES: Mode(
        mode_type=ModeType.NOTES,
        name="Notes",
        description="Bullet points, tasks, meeting notes",
        prompt_file="planning",
        icon="*",
    ),
    ModeType.DICTATION: Mode(
        mode_type=ModeType.DICTATION,
        name="Dictation",
        description="Natural prose, documents",
        prompt_file="general",
        icon="Aa",
    ),
    ModeType.EMAIL: Mode(
        mode_type=ModeType.EMAIL,
        name="Email",
        description="Greeting/sign-off formatting",
        prompt_file="email",
        icon="@",
    ),
    ModeType.CHAT: Mode(
        mode_type=ModeType.CHAT,
        name="Chat",
        description="Casual, brief responses",
        prompt_file="chat",
        icon="...",
    ),
}

# Mode cycle order for hotkey cycling
MODE_CYCLE_ORDER: list[ModeType] = [
    ModeType.CODE,
    ModeType.NOTES,
    ModeType.DICTATION,
    ModeType.EMAIL,
    ModeType.CHAT,
]


__all__ = [
    "Mode",
    "ModeType",
    "MODES",
    "MODE_CYCLE_ORDER",
]
