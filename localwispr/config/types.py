"""Configuration type definitions and defaults for LocalWispr."""

from typing import TypedDict


class ModelConfig(TypedDict, total=False):
    """Model configuration settings."""

    name: str
    device: str
    compute_type: str
    language: str  # Optional language code for transcription


class HotkeyConfig(TypedDict):
    """Hotkey configuration settings."""

    mode: str  # "push-to-talk" or "toggle"
    modifiers: list[str]  # ["win", "ctrl", "shift"]
    audio_feedback: bool
    mute_system: bool  # Mute system audio during recording


class ContextConfig(TypedDict):
    """Context detection configuration settings."""

    coding_apps: list[str]
    planning_apps: list[str]
    coding_keywords: list[str]
    planning_keywords: list[str]


class OutputConfig(TypedDict):
    """Output configuration settings."""

    auto_paste: bool  # Whether to auto-paste or clipboard-only
    paste_delay_ms: int  # Delay before paste to ensure focus


class VocabularyConfig(TypedDict):
    """Vocabulary configuration settings."""

    words: list[str]  # Custom words for better transcription


class StreamingConfig(TypedDict, total=False):
    """Streaming transcription configuration settings."""

    enabled: bool  # Enable streaming/chunked transcription
    min_silence_ms: int  # Minimum silence to trigger segment boundary
    max_segment_duration: float  # Maximum segment duration before forced split
    min_segment_duration: float  # Minimum segment duration to transcribe
    overlap_ms: int  # Audio overlap between segments for context


class Config(TypedDict, total=False):
    """Full application configuration."""

    model: ModelConfig
    hotkeys: HotkeyConfig
    context: ContextConfig
    output: OutputConfig
    vocabulary: VocabularyConfig
    streaming: StreamingConfig


DEFAULT_CONFIG: Config = {
    "model": {
        "name": "large-v3",
        "device": "auto",
        "compute_type": "auto",
        "language": "auto",
    },
    "hotkeys": {
        "mode": "push-to-talk",
        "modifiers": ["win", "ctrl", "shift"],
        "audio_feedback": True,
        "mute_system": False,
    },
    "vocabulary": {
        "words": [],
    },
    "context": {
        "coding_apps": [
            "code",
            "pycharm",
            "intellij",
            "vim",
            "neovim",
            "visual studio",
            "sublime",
            "atom",
            "emacs",
        ],
        "planning_apps": [
            "notion",
            "obsidian",
            "todoist",
            "jira",
            "asana",
            "trello",
            "linear",
        ],
        "coding_keywords": [
            "function",
            "variable",
            "import",
            "class",
            "def",
            "return",
            "async",
            "await",
            "const",
            "let",
            "var",
            "public",
            "private",
            "interface",
            "type",
            "null",
            "undefined",
        ],
        "planning_keywords": [
            "task",
            "project",
            "milestone",
            "deadline",
            "goal",
            "plan",
            "schedule",
            "priority",
            "action",
            "item",
            "todo",
            "complete",
            "review",
        ],
    },
    "output": {
        "auto_paste": True,  # Auto-paste after transcription
        "paste_delay_ms": 50,  # Small delay before paste
    },
    "streaming": {
        "enabled": False,  # Disabled by default (opt-in feature)
        "min_silence_ms": 800,  # Higher = more accurate, fewer segments
        "max_segment_duration": 20.0,  # Force split after 20s
        "min_segment_duration": 2.0,  # Avoid tiny fragments
        "overlap_ms": 100,  # Context overlap between segments
    },
}
