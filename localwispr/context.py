"""Context detection for LocalWispr.

Detects the current context (CODING, PLANNING, GENERAL) based on:
- Active window title (pre-detection)
- Transcribed text keywords (post-detection)

Keyword and app lists are loaded from config.py (single source of truth).
"""

from enum import Enum


class ContextType(Enum):
    """Types of context that can be detected."""

    CODING = "coding"
    PLANNING = "planning"
    GENERAL = "general"


class ContextDetector:
    """Detects context from window titles and transcribed text."""

    def __init__(
        self,
        coding_apps: list[str] | None = None,
        planning_apps: list[str] | None = None,
        coding_keywords: list[str] | None = None,
        planning_keywords: list[str] | None = None,
    ) -> None:
        """Initialize the context detector.

        If lists are not provided, loads from config (single source of truth).

        Args:
            coding_apps: List of app name patterns for coding context.
            planning_apps: List of app name patterns for planning context.
            coding_keywords: List of keywords indicating coding context.
            planning_keywords: List of keywords indicating planning context.
        """
        # Load from config if any argument is None
        if any(arg is None for arg in [coding_apps, planning_apps, coding_keywords, planning_keywords]):
            from localwispr.config import get_config
            config = get_config()
            ctx = config["context"]

        self.coding_apps = coding_apps if coding_apps is not None else ctx["coding_apps"]
        self.planning_apps = planning_apps if planning_apps is not None else ctx["planning_apps"]
        self.coding_keywords = coding_keywords if coding_keywords is not None else ctx["coding_keywords"]
        self.planning_keywords = planning_keywords if planning_keywords is not None else ctx["planning_keywords"]

    def detect_from_window(self) -> ContextType:
        """Detect context from the currently focused window title.

        Uses pygetwindow to get the active window title and matches against
        configured app patterns.

        Returns:
            ContextType based on window title, or GENERAL if detection fails.
        """
        try:
            import pygetwindow as gw

            active_window = gw.getActiveWindow()
            if active_window is None:
                return ContextType.GENERAL

            title = active_window.title.lower()

            # Check for coding apps
            for app in self.coding_apps:
                if app.lower() in title:
                    return ContextType.CODING

            # Check for planning apps
            for app in self.planning_apps:
                if app.lower() in title:
                    return ContextType.PLANNING

            return ContextType.GENERAL

        except Exception:
            # Platform fallback - return GENERAL on any error
            return ContextType.GENERAL

    def detect_from_text(self, text: str) -> ContextType:
        """Detect context from transcribed text using keyword analysis.

        Args:
            text: The transcribed text to analyze.

        Returns:
            ContextType based on keyword matches.
        """
        text_lower = text.lower()
        words = set(text_lower.split())

        coding_count = sum(1 for kw in self.coding_keywords if kw.lower() in words)
        planning_count = sum(1 for kw in self.planning_keywords if kw.lower() in words)

        if coding_count > planning_count and coding_count > 0:
            return ContextType.CODING
        elif planning_count > coding_count and planning_count > 0:
            return ContextType.PLANNING
        else:
            return ContextType.GENERAL

    def get_context(self, text: str | None = None) -> ContextType:
        """Get the current context using hybrid detection.

        Uses window detection first, then refines with text analysis if text
        is provided.

        Args:
            text: Optional transcribed text for keyword analysis.

        Returns:
            ContextType determined by hybrid detection.
        """
        window_context = self.detect_from_window()

        if text is None:
            return window_context

        text_context = self.detect_from_text(text)

        # If both agree or window is GENERAL, use text context
        if window_context == ContextType.GENERAL:
            return text_context

        # If text provides a specific context that differs, prefer text
        # (post-detection can refine pre-detection)
        if text_context != ContextType.GENERAL:
            return text_context

        return window_context
