"""Tests for context detection module."""

import pytest

from localwispr.context import ContextDetector, ContextType
from localwispr.prompts import load_prompt, get_available_contexts


@pytest.mark.usefixtures("isolated_config_cache")
class TestContextDetector:
    """Tests for ContextDetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ContextDetector()

    # Tests for detect_from_text()

    def test_detect_from_text_coding_keywords(self):
        """Test that coding keywords are correctly detected."""
        # Single keyword
        assert self.detector.detect_from_text("add a function here") == ContextType.CODING
        assert self.detector.detect_from_text("create a variable") == ContextType.CODING
        assert self.detector.detect_from_text("import the module") == ContextType.CODING
        assert self.detector.detect_from_text("define a class") == ContextType.CODING

    def test_detect_from_text_multiple_coding_keywords(self):
        """Test that multiple coding keywords strengthen detection."""
        text = "create a function with a variable and return the value"
        assert self.detector.detect_from_text(text) == ContextType.CODING

    def test_detect_from_text_planning_keywords(self):
        """Test that planning keywords are correctly detected."""
        assert self.detector.detect_from_text("create a task") == ContextType.PLANNING
        assert self.detector.detect_from_text("set the deadline") == ContextType.PLANNING
        assert self.detector.detect_from_text("this is a milestone") == ContextType.PLANNING
        assert self.detector.detect_from_text("schedule the review") == ContextType.PLANNING

    def test_detect_from_text_multiple_planning_keywords(self):
        """Test that multiple planning keywords strengthen detection."""
        text = "create a task for the project milestone with a deadline"
        assert self.detector.detect_from_text(text) == ContextType.PLANNING

    def test_detect_from_text_neutral_returns_general(self):
        """Test that neutral text returns GENERAL context."""
        assert self.detector.detect_from_text("hello world") == ContextType.GENERAL
        assert self.detector.detect_from_text("the quick brown fox") == ContextType.GENERAL
        assert self.detector.detect_from_text("I need to buy groceries") == ContextType.GENERAL

    def test_detect_from_text_case_insensitive(self):
        """Test that keyword detection is case insensitive."""
        assert self.detector.detect_from_text("FUNCTION") == ContextType.CODING
        assert self.detector.detect_from_text("Function") == ContextType.CODING
        assert self.detector.detect_from_text("TASK") == ContextType.PLANNING
        assert self.detector.detect_from_text("Task") == ContextType.PLANNING

    def test_detect_from_text_coding_wins_when_more(self):
        """Test that coding context wins when more coding keywords present."""
        text = "define a function class variable to track the task"
        # 3 coding keywords (function, class, variable) vs 1 planning (task)
        assert self.detector.detect_from_text(text) == ContextType.CODING

    def test_detect_from_text_planning_wins_when_more(self):
        """Test that planning context wins when more planning keywords present."""
        text = "create task project milestone for the function deadline"
        # 4 planning keywords vs 1 coding (function)
        assert self.detector.detect_from_text(text) == ContextType.PLANNING

    def test_detect_from_text_empty_string(self):
        """Test that empty string returns GENERAL context."""
        assert self.detector.detect_from_text("") == ContextType.GENERAL

    # Tests for detect_from_window() - limited testing due to platform dependency

    def test_detect_from_window_graceful_fallback(self):
        """Test that window detection returns GENERAL on any error."""
        # This test verifies the fallback behavior
        # The actual window detection is platform-dependent
        result = self.detector.detect_from_window()
        # Should return a valid ContextType (either detected or GENERAL fallback)
        assert isinstance(result, ContextType)

    # Tests for get_context()

    def test_get_context_without_text(self):
        """Test that get_context without text uses window detection."""
        result = self.detector.get_context()
        assert isinstance(result, ContextType)

    def test_get_context_with_text(self):
        """Test that get_context with text uses hybrid detection."""
        result = self.detector.get_context("create a function variable")
        assert isinstance(result, ContextType)
        # With coding keywords, should likely be CODING
        # (unless window says otherwise)


class TestLoadPrompt:
    """Tests for load_prompt function."""

    def test_load_prompt_coding(self):
        """Test that coding prompt loads correctly."""
        prompt = load_prompt("coding")
        assert len(prompt) > 0
        # Should contain some technical terms
        assert "function" in prompt.lower() or "api" in prompt.lower()

    def test_load_prompt_planning(self):
        """Test that planning prompt loads correctly."""
        prompt = load_prompt("planning")
        assert len(prompt) > 0
        # Should contain planning terms
        assert "task" in prompt.lower() or "milestone" in prompt.lower() or "sprint" in prompt.lower()

    def test_load_prompt_general(self):
        """Test that general prompt loads correctly."""
        prompt = load_prompt("general")
        assert len(prompt) > 0

    def test_load_prompt_fallback(self):
        """Test that unknown context falls back to general."""
        prompt = load_prompt("unknown_nonexistent_context")
        general_prompt = load_prompt("general")
        # Should fallback to general.txt content
        assert prompt == general_prompt

    def test_load_prompt_case_insensitive(self):
        """Test that context names are case insensitive."""
        coding1 = load_prompt("coding")
        coding2 = load_prompt("CODING")
        coding3 = load_prompt("Coding")
        assert coding1 == coding2 == coding3


class TestGetAvailableContexts:
    """Tests for get_available_contexts function."""

    def test_get_available_contexts_returns_list(self):
        """Test that get_available_contexts returns a list."""
        contexts = get_available_contexts()
        assert isinstance(contexts, list)

    def test_get_available_contexts_has_expected(self):
        """Test that expected contexts are available."""
        contexts = get_available_contexts()
        assert "coding" in contexts
        assert "planning" in contexts
        assert "general" in contexts
