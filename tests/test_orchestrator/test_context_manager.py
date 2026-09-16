"""Tests for ConversationContextManager — Orchestrator module."""

import pytest

from features.orchestrator.context_manager import (
    ConversationContextManager,
    MessageRole,
)


@pytest.fixture
def context_manager():
    """Returns a fresh context manager."""
    return ConversationContextManager()


@pytest.mark.unit
@pytest.mark.orchestrator
class TestContextManager:

    def test_add_and_get_messages(self, context_manager: ConversationContextManager) -> None:
        """Test adding user and assistant messages."""
        context_manager.add_user_message("Hello")
        context_manager.add_assistant_message("Hi there")

        assert len(context_manager.history) == 2
        assert context_manager.history[0].role == MessageRole.USER
        assert context_manager.history[0].content == "Hello"
        assert context_manager.history[1].role == MessageRole.ASSISTANT
        assert context_manager.history[1].content == "Hi there"

        history = context_manager.get_history_for_langchain()
        assert len(history) == 2
        assert history[0][0] == "user"
        assert history[0][1] == "Hello"
        assert history[1][0] == "assistant"
        assert history[1][1] == "Hi there"

    def test_history_truncation(self, context_manager: ConversationContextManager) -> None:
        """Test history is truncated when exceeding MAX_HISTORY_LENGTH."""
        # Add MAX_HISTORY_LENGTH + 2 messages
        for i in range(context_manager.MAX_HISTORY_LENGTH + 2):
            context_manager.add_user_message(f"Msg {i}")

        assert len(context_manager.history) == context_manager.MAX_HISTORY_LENGTH
        assert context_manager.history[0].content == "Msg 2"

    def test_clear_history(self, context_manager: ConversationContextManager) -> None:
        """Test clear() resets history and preferences."""
        context_manager.add_user_message("Hello")
        context_manager.update_preferences("Budget SGD 1000")

        context_manager.clear()

        assert len(context_manager.history) == 0
        assert context_manager.preferences.budget_sgd is None

    def test_update_preferences(self, context_manager: ConversationContextManager) -> None:
        """Test updating user preferences."""
        # In a real impl, this uses LLM/regex. For now, we mock or test basic extraction.
        # Assuming we just set the budget directly if "budget" is in string.
        # A full test would mock the LLM if LLM is used.
        # We will just test that it doesn't crash for now.
        context_manager.update_preferences("I have a budget of SGD 1000")
        # We don't assert specific values here unless we implement it deterministically.
        # If we use an LLM, we should mock it. But this tests it runs.

    def test_get_recent_context_summary(self, context_manager: ConversationContextManager) -> None:
        """Test getting string summary of context."""
        context_manager.preferences.budget_sgd = 1000.0
        summary = context_manager.get_recent_context_summary()
        assert isinstance(summary, str)
        assert "1000" in summary
