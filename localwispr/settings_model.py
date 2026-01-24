"""Settings model for LocalWispr.

Provides immutable settings snapshots and validation for the settings system.
This module is GUI-agnostic and can be used with any frontend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SettingsSnapshot:
    """Immutable settings state - thread-safe, hashable, easy to diff.

    All settings are stored as flat fields for easy access and comparison.
    Use with_changes() to create modified copies (immutable pattern).

    Example:
        >>> snap = SettingsSnapshot.from_config_dict(config)
        >>> new_snap = snap.with_changes(model_name="small")
        >>> changed = snap.diff(new_snap)  # {"model_name"}
    """

    # Model settings
    model_name: str = "large-v3"
    model_device: str = "cuda"
    model_compute_type: str = "float16"
    model_language: str = "auto"

    # Hotkey settings
    hotkey_mode: str = "push-to-talk"
    hotkey_modifiers: tuple[str, ...] = ("win", "ctrl", "shift")
    audio_feedback: bool = True
    mute_system: bool = False

    # Output settings
    auto_paste: bool = True
    paste_delay_ms: int = 50

    # Vocabulary
    vocabulary_words: tuple[str, ...] = ()

    # Streaming settings
    streaming_enabled: bool = False
    streaming_min_silence_ms: int = 800
    streaming_max_segment_duration: float = 20.0
    streaming_min_segment_duration: float = 2.0
    streaming_overlap_ms: int = 100

    @classmethod
    def from_config_dict(cls, config: dict[str, Any]) -> SettingsSnapshot:
        """Create snapshot from legacy config dictionary.

        Args:
            config: Configuration dictionary (from config.py).

        Returns:
            New SettingsSnapshot with values from config.
        """
        model = config.get("model", {})
        hotkeys = config.get("hotkeys", {})
        output = config.get("output", {})
        vocabulary = config.get("vocabulary", {})
        streaming = config.get("streaming", {})

        return cls(
            # Model
            model_name=model.get("name", "large-v3"),
            model_device=model.get("device", "cuda"),
            model_compute_type=model.get("compute_type", "float16"),
            model_language=model.get("language", "auto"),
            # Hotkeys
            hotkey_mode=hotkeys.get("mode", "push-to-talk"),
            hotkey_modifiers=tuple(hotkeys.get("modifiers", ["win", "ctrl", "shift"])),
            audio_feedback=hotkeys.get("audio_feedback", True),
            mute_system=hotkeys.get("mute_system", False),
            # Output
            auto_paste=output.get("auto_paste", True),
            paste_delay_ms=output.get("paste_delay_ms", 50),
            # Vocabulary
            vocabulary_words=tuple(vocabulary.get("words", [])),
            # Streaming
            streaming_enabled=streaming.get("enabled", False),
            streaming_min_silence_ms=streaming.get("min_silence_ms", 800),
            streaming_max_segment_duration=streaming.get("max_segment_duration", 20.0),
            streaming_min_segment_duration=streaming.get("min_segment_duration", 2.0),
            streaming_overlap_ms=streaming.get("overlap_ms", 100),
        )

    def to_config_dict(self) -> dict[str, Any]:
        """Convert snapshot to legacy config dictionary.

        Returns:
            Configuration dictionary compatible with save_config().
        """
        return {
            "model": {
                "name": self.model_name,
                "device": self.model_device,
                "compute_type": self.model_compute_type,
                "language": self.model_language,
            },
            "hotkeys": {
                "mode": self.hotkey_mode,
                "modifiers": list(self.hotkey_modifiers),
                "audio_feedback": self.audio_feedback,
                "mute_system": self.mute_system,
            },
            "output": {
                "auto_paste": self.auto_paste,
                "paste_delay_ms": self.paste_delay_ms,
            },
            "vocabulary": {
                "words": list(self.vocabulary_words),
            },
            "streaming": {
                "enabled": self.streaming_enabled,
                "min_silence_ms": self.streaming_min_silence_ms,
                "max_segment_duration": self.streaming_max_segment_duration,
                "min_segment_duration": self.streaming_min_segment_duration,
                "overlap_ms": self.streaming_overlap_ms,
            },
        }

    def with_changes(self, **kwargs: Any) -> SettingsSnapshot:
        """Create a new snapshot with specified fields changed.

        Args:
            **kwargs: Field names and new values.

        Returns:
            New SettingsSnapshot with changes applied.

        Example:
            >>> new_snap = snap.with_changes(model_name="small", audio_feedback=False)
        """
        # Get current values as dict
        current = {
            "model_name": self.model_name,
            "model_device": self.model_device,
            "model_compute_type": self.model_compute_type,
            "model_language": self.model_language,
            "hotkey_mode": self.hotkey_mode,
            "hotkey_modifiers": self.hotkey_modifiers,
            "audio_feedback": self.audio_feedback,
            "mute_system": self.mute_system,
            "auto_paste": self.auto_paste,
            "paste_delay_ms": self.paste_delay_ms,
            "vocabulary_words": self.vocabulary_words,
            "streaming_enabled": self.streaming_enabled,
            "streaming_min_silence_ms": self.streaming_min_silence_ms,
            "streaming_max_segment_duration": self.streaming_max_segment_duration,
            "streaming_min_segment_duration": self.streaming_min_segment_duration,
            "streaming_overlap_ms": self.streaming_overlap_ms,
        }
        # Apply changes
        current.update(kwargs)
        return SettingsSnapshot(**current)

    def diff(self, other: SettingsSnapshot) -> set[str]:
        """Find fields that differ between this snapshot and another.

        Args:
            other: Another SettingsSnapshot to compare against.

        Returns:
            Set of field names that have different values.
        """
        changed = set()
        fields = [
            "model_name", "model_device", "model_compute_type", "model_language",
            "hotkey_mode", "hotkey_modifiers", "audio_feedback", "mute_system",
            "auto_paste", "paste_delay_ms", "vocabulary_words",
            "streaming_enabled", "streaming_min_silence_ms",
            "streaming_max_segment_duration", "streaming_min_segment_duration",
            "streaming_overlap_ms",
        ]
        for field_name in fields:
            if getattr(self, field_name) != getattr(other, field_name):
                changed.add(field_name)
        return changed


@dataclass
class ValidationResult:
    """Result of settings validation.

    Attributes:
        is_valid: True if all settings are valid.
        errors: Dict mapping field names to error messages.
    """

    is_valid: bool = True
    errors: dict[str, str] = field(default_factory=dict)

    def add_error(self, field: str, message: str) -> None:
        """Add a validation error.

        Args:
            field: Name of the invalid field.
            message: Human-readable error message.
        """
        self.errors[field] = message
        self.is_valid = False


# Valid model options
VALID_MODEL_NAMES = {"tiny", "base", "small", "medium", "large-v2", "large-v3"}
VALID_DEVICES = {"cuda", "cpu"}
VALID_COMPUTE_TYPES = {"float16", "int8", "float32"}
VALID_LANGUAGES = {
    "auto", "en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja", "ko"
}
VALID_HOTKEY_MODES = {"push-to-talk", "toggle"}
VALID_MODIFIERS = {"win", "ctrl", "shift", "alt"}


class SettingsValidator:
    """Validates settings before save - blocks save if ANY setting invalid.

    All validation is strict: any invalid field causes the entire save to fail.
    This prevents partial saves that could leave settings in an inconsistent state.

    Example:
        >>> result = SettingsValidator.validate(snapshot)
        >>> if not result.is_valid:
        ...     for field, error in result.errors.items():
        ...         print(f"{field}: {error}")
    """

    @classmethod
    def validate(cls, settings: SettingsSnapshot) -> ValidationResult:
        """Validate all settings.

        Args:
            settings: Settings snapshot to validate.

        Returns:
            ValidationResult with is_valid=False if any field is invalid.
        """
        result = ValidationResult()

        # Model validation
        if settings.model_name not in VALID_MODEL_NAMES:
            result.add_error(
                "model_name",
                f"Invalid model: {settings.model_name}. "
                f"Valid: {', '.join(sorted(VALID_MODEL_NAMES))}"
            )

        if settings.model_device not in VALID_DEVICES:
            result.add_error(
                "model_device",
                f"Invalid device: {settings.model_device}. "
                f"Valid: {', '.join(sorted(VALID_DEVICES))}"
            )

        if settings.model_compute_type not in VALID_COMPUTE_TYPES:
            result.add_error(
                "model_compute_type",
                f"Invalid compute type: {settings.model_compute_type}. "
                f"Valid: {', '.join(sorted(VALID_COMPUTE_TYPES))}"
            )

        if settings.model_language not in VALID_LANGUAGES:
            result.add_error(
                "model_language",
                f"Invalid language: {settings.model_language}. "
                f"Valid: {', '.join(sorted(VALID_LANGUAGES))}"
            )

        # Hotkey validation
        if settings.hotkey_mode not in VALID_HOTKEY_MODES:
            result.add_error(
                "hotkey_mode",
                f"Invalid mode: {settings.hotkey_mode}. "
                f"Valid: {', '.join(sorted(VALID_HOTKEY_MODES))}"
            )

        invalid_mods = set(settings.hotkey_modifiers) - VALID_MODIFIERS
        if invalid_mods:
            result.add_error(
                "hotkey_modifiers",
                f"Invalid modifiers: {', '.join(sorted(invalid_mods))}. "
                f"Valid: {', '.join(sorted(VALID_MODIFIERS))}"
            )

        if len(settings.hotkey_modifiers) < 2:
            result.add_error(
                "hotkey_modifiers",
                "At least 2 modifier keys required"
            )

        # Output validation
        if not isinstance(settings.paste_delay_ms, int) or settings.paste_delay_ms < 0:
            result.add_error(
                "paste_delay_ms",
                "Paste delay must be a non-negative integer"
            )

        if settings.paste_delay_ms > 1000:
            result.add_error(
                "paste_delay_ms",
                "Paste delay cannot exceed 1000ms"
            )

        # Vocabulary validation
        cls._validate_vocabulary(settings, result)

        # Streaming validation
        if settings.streaming_min_silence_ms < 100:
            result.add_error(
                "streaming_min_silence_ms",
                "Minimum silence must be at least 100ms"
            )

        if settings.streaming_max_segment_duration < 1.0:
            result.add_error(
                "streaming_max_segment_duration",
                "Maximum segment duration must be at least 1 second"
            )

        if settings.streaming_min_segment_duration < 0.5:
            result.add_error(
                "streaming_min_segment_duration",
                "Minimum segment duration must be at least 0.5 seconds"
            )

        if settings.streaming_min_segment_duration >= settings.streaming_max_segment_duration:
            result.add_error(
                "streaming_min_segment_duration",
                "Minimum segment duration must be less than maximum"
            )

        return result

    @classmethod
    def _validate_vocabulary(
        cls, settings: SettingsSnapshot, result: ValidationResult
    ) -> None:
        """Validate vocabulary words for TOML safety.

        Args:
            settings: Settings to validate.
            result: ValidationResult to add errors to.
        """
        unsafe_chars = {'"', '\\', '\n', '\r', '\t'}

        for i, word in enumerate(settings.vocabulary_words):
            if not word.strip():
                result.add_error(
                    f"vocabulary_words[{i}]",
                    "Empty vocabulary word"
                )
                continue

            if len(word) > 100:
                result.add_error(
                    f"vocabulary_words[{i}]",
                    f"Word too long (max 100 chars): {word[:20]}..."
                )

            found_unsafe = [c for c in word if c in unsafe_chars]
            if found_unsafe:
                result.add_error(
                    f"vocabulary_words[{i}]",
                    f"Word contains unsafe characters: {word[:20]}"
                )
