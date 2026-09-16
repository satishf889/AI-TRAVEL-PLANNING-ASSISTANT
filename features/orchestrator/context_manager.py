from __future__ import annotations

"""Multi-turn conversation context manager.

Maintains conversation history and injects relevant prior context
into each new LLM call to enable multi-turn conversations.

Requirements satisfied: Minimum Acceptance Criteria — Multi-turn conversation
with retained context; Prompt Engineering — preserve user preferences.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageRole(Enum):
    """Role of a message in the conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ConversationMessage:
    """A single message in the conversation history."""

    role: MessageRole
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UserPreferences:
    """Travel preferences extracted from the conversation."""

    budget_inr: float | None = None
    budget_sgd: float | None = None
    trip_duration_days: int | None = None
    travel_style: str | None = None
    indoor_preference: bool | None = None


class ConversationContextManager:
    """Manages multi-turn conversation history and user preference extraction."""

    MAX_HISTORY_LENGTH = 20

    def __init__(self) -> None:
        """Initialise with an empty conversation history."""
        self.history: list[ConversationMessage] = []
        self.preferences: UserPreferences = UserPreferences()

    def add_user_message(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a user message to the conversation history."""
        self.history.append(ConversationMessage(role=MessageRole.USER, content=content, metadata=metadata or {}))
        self._truncate()

    def add_assistant_message(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add an assistant response to the conversation history."""
        self.history.append(ConversationMessage(role=MessageRole.ASSISTANT, content=content, metadata=metadata or {}))
        self._truncate()

    def _truncate(self) -> None:
        """Ensure history does not exceed max length."""
        if len(self.history) > self.MAX_HISTORY_LENGTH:
            self.history = self.history[-self.MAX_HISTORY_LENGTH:]

    def get_history_for_langchain(self) -> list[tuple[str, str]]:
        """Return conversation history in LangChain format."""
        return [(msg.role.value, msg.content) for msg in self.history]

    def get_recent_context_summary(self) -> str:
        """Summarise recent conversation context for prompt injection."""
        prefs = []
        if self.preferences.budget_sgd is not None:
            prefs.append(f"Budget: SGD {self.preferences.budget_sgd}")
        if self.preferences.trip_duration_days is not None:
            prefs.append(f"Duration: {self.preferences.trip_duration_days} days")

        if not prefs:
            return "No specific preferences known yet."
        return "User Preferences: " + ", ".join(prefs)

    def update_preferences(self, message: str) -> None:
        """Extract and update user preferences from a message."""
        # Simple extraction logic for the prototype
        budget_match = re.search(r'budget\s*(?:of)?\s*SGD\s*(\d+)', message, re.IGNORECASE)
        if budget_match:
            self.preferences.budget_sgd = float(budget_match.group(1))

    def clear(self) -> None:
        """Clear conversation history and reset preferences."""
        self.history.clear()
        self.preferences = UserPreferences()

    def __len__(self) -> int:
        """Return the number of messages in the conversation history."""
        return len(self.history)
