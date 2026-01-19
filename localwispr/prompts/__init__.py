"""
Context-specific prompt files for Whisper transcription.

This module provides initial_prompt text files that help Whisper
better recognize domain-specific vocabulary during transcription.
"""

from pathlib import Path


# Directory containing prompt files
PROMPTS_DIR = Path(__file__).parent


def load_prompt(context: str) -> str:
    """
    Load the initial_prompt text for a given context.

    Args:
        context: Context name as lowercase string ("coding", "planning", "general")

    Returns:
        The prompt text content for the specified context.
        Falls back to general.txt if the requested context file is missing.
    """
    # Normalize context name to lowercase
    context = context.lower().strip()

    # Build path to the prompt file
    prompt_file = PROMPTS_DIR / f"{context}.txt"

    # Fall back to general.txt if file doesn't exist
    if not prompt_file.exists():
        prompt_file = PROMPTS_DIR / "general.txt"

    # Read and return the prompt content
    try:
        return prompt_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        # Ultimate fallback if even general.txt is missing
        return ""


def get_available_contexts() -> list[str]:
    """
    Get list of available context names based on .txt files in prompts directory.

    Returns:
        List of context names (without .txt extension)
    """
    return [f.stem for f in PROMPTS_DIR.glob("*.txt")]
