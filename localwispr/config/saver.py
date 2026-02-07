"""Configuration saving for LocalWispr."""

import logging
from pathlib import Path

from localwispr.config.types import Config
from localwispr.config.loader import _get_appdata_config_path

logger = logging.getLogger(__name__)


def save_config(config: Config, config_path: Path | None = None) -> None:
    """Save configuration to TOML file.

    Args:
        config: Configuration dictionary to save.
        config_path: Path to config file. Defaults to AppData user-settings.toml
                     (when frozen) or project root (when running as script).
    """
    if config_path is None:
        config_path = _get_appdata_config_path()

    # Create parent directory if needed
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Build TOML content with comments
    lines = ["# LocalWispr Configuration", ""]

    # Model section
    lines.append("[model]")
    lines.append("# Whisper model to use: tiny, base, small, medium, large-v2, large-v3")
    lines.append(f'name = "{config["model"]["name"]}"')
    lines.append("")
    lines.append("# Device: auto (auto-detect), cuda (GPU), or cpu")
    lines.append(f'device = "{config["model"]["device"]}"')
    lines.append("")
    lines.append("# Compute type: auto, float16 (GPU), int8 (CPU), or float32")
    lines.append(f'compute_type = "{config["model"]["compute_type"]}"')
    lines.append("")
    lines.append("# Language: auto, en, es, fr, de, it, pt, nl, ru, zh, ja, ko")
    language = config.get("model", {}).get("language", "auto")
    lines.append(f'language = "{language}"')
    lines.append("")

    # Hotkeys section
    lines.append("[hotkeys]")
    lines.append("# Recording activation mode:")
    lines.append('#   "push-to-talk" - Hold keys to record, release to stop and transcribe')
    lines.append('#   "toggle" - Press once to start recording, press again to stop and transcribe')
    lines.append(f'mode = "{config["hotkeys"]["mode"]}"')
    lines.append("")
    lines.append("# Modifier key combination to activate recording")
    lines.append('# Available modifiers: "win", "ctrl", "shift", "alt"')
    modifiers = config["hotkeys"]["modifiers"]
    modifiers_str = ", ".join(f'"{m}"' for m in modifiers)
    lines.append(f"modifiers = [{modifiers_str}]")
    lines.append("")
    lines.append("# Play audio feedback sounds when recording starts/stops")
    audio_fb = "true" if config["hotkeys"]["audio_feedback"] else "false"
    lines.append(f"audio_feedback = {audio_fb}")
    lines.append("")
    lines.append("# Mute system audio during recording (prevents feedback)")
    mute_sys = "true" if config["hotkeys"].get("mute_system", False) else "false"
    lines.append(f"mute_system = {mute_sys}")
    lines.append("")

    # Output section
    lines.append("[output]")
    lines.append("# Auto-paste after transcription (or clipboard-only)")
    auto_paste = "true" if config["output"]["auto_paste"] else "false"
    lines.append(f"auto_paste = {auto_paste}")
    lines.append("")
    lines.append("# Delay before paste to ensure focus (milliseconds)")
    lines.append(f"paste_delay_ms = {config['output']['paste_delay_ms']}")
    lines.append("")

    # Vocabulary section
    vocab = config.get("vocabulary", {}).get("words", [])
    if vocab:
        lines.append("[vocabulary]")
        lines.append("# Custom words for better transcription accuracy")
        vocab_str = ", ".join(f'"{w}"' for w in vocab)
        lines.append(f"words = [{vocab_str}]")
        lines.append("")

    # Streaming section
    streaming = config.get("streaming", {})
    lines.append("[streaming]")
    lines.append("# Enable streaming transcription for faster processing of long recordings")
    lines.append("# When enabled, audio is transcribed in segments during recording")
    streaming_enabled = "true" if streaming.get("enabled", False) else "false"
    lines.append(f"enabled = {streaming_enabled}")
    lines.append("")
    lines.append("# Minimum silence duration (ms) before triggering segment transcription")
    lines.append("# Higher = more accurate (fewer segments, more context per transcription)")
    lines.append(f"min_silence_ms = {streaming.get('min_silence_ms', 800)}")
    lines.append("")
    lines.append("# Maximum segment duration (seconds) before forced transcription")
    lines.append(f"max_segment_duration = {streaming.get('max_segment_duration', 20.0)}")
    lines.append("")
    lines.append("# Minimum segment duration (seconds) - avoid tiny fragments")
    lines.append(f"min_segment_duration = {streaming.get('min_segment_duration', 2.0)}")
    lines.append("")
    lines.append("# Audio overlap between segments (ms) for better context")
    lines.append(f"overlap_ms = {streaming.get('overlap_ms', 100)}")
    lines.append("")

    # Write to file
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
