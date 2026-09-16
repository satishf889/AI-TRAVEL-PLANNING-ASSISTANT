"""Multi-turn conversation context manager.

Maintains conversation history and injects relevant prior context
into each new LLM call to enable multi-turn conversations.

Requirements satisfied: Minimum Acceptance Criteria — Multi-turn conversation
with retained context; Prompt Engineering — preserve user preferences.
"""

from dataclasses import dataclass, field
from enum import Enum


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
    metadata: dict = field(default_factory=dict)
    # Metadata can include: sources_used, mcp_tools_used, kb_chunks_used


@dataclass
class UserPreferences:
    """Travel preferences extracted from the conversation."""

    budget_inr: float | None = None
    budget_sgd: float | None = None
    trip_duration_days: int | None = None
    travel_style: str | None = None  # e.g., "family", "cultural", "adventure"
    indoor_preference: bool | None = None


class ConversationContextManager:
    """Manages multi-turn conversation history and user preference extraction.

    Tracks the full conversation history and extracts user preferences
    from past messages to inject into future prompts (e.g., remembered budget,
    trip style, indoor/outdoor preference).
    """

    MAX_HISTORY_LENGTH = 20  # Maximum turns before truncating old messages

    def __init__(self) -> None:
        """Initialise with an empty conversation history."""
        self.history: list[ConversationMessage] = []
        self.preferences: UserPreferences = UserPreferences()

    def add_user_message(self, content: str, metadata: dict | None = None) -> None:
        """Add a user message to the conversation history.

        Args:
            content: The user's message text.
            metadata: Optional metadata (e.g., detected intent).
        """
        raise NotImplementedError("Implement in TDD cycle")

    def add_assistant_message(self, content: str, metadata: dict | None = None) -> None:
        """Add an assistant response to the conversation history.

        Args:
            content: The assistant's response text.
            metadata: Optional metadata (e.g., sources used, MCP tools called).
        """
        raise NotImplementedError("Implement in TDD cycle")

    def get_history_for_langchain(self) -> list[tuple[str, str]]:
        """Return conversation history in LangChain format.

        Returns:
            List of (role, content) tuples for LangChain memory input.
        """
        raise NotImplementedError("Implement in TDD cycle")

    def get_recent_context_summary(self) -> str:
        """Summarise recent conversation context for prompt injection.

        Returns:
            A concise string summarising recent user preferences and context.
        """
        raise NotImplementedError("Implement in TDD cycle")

    def update_preferences(self, message: str) -> None:
        """Extract and update user preferences from a message.

        Args:
            message: User message to parse for preference signals.
        """
        raise NotImplementedError("Implement in TDD cycle")

    def clear(self) -> None:
        """Clear conversation history and reset preferences."""
        raise NotImplementedError("Implement in TDD cycle")

    def __len__(self) -> int:
        """Return the number of messages in the conversation history."""
        return len(self.history)
