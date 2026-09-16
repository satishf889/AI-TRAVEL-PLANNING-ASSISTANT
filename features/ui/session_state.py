from __future__ import annotations

"""Streamlit session state management for the AI Travel Planning Assistant.

Centralises all Streamlit session state initialisation and access
to avoid scattered st.session_state references throughout the codebase.
"""

import streamlit as st

from features.orchestrator.agent import TravelAgent


def initialise_session_state() -> None:
    """Initialise all required Streamlit session state variables.

    Must be called at the top of the Streamlit app before accessing any state.
    Idempotent — safe to call on every rerun.
    """
    raise NotImplementedError("Implement in TDD cycle")


def get_agent() -> TravelAgent:
    """Get the TravelAgent singleton from session state.

    Returns:
        The initialised TravelAgent instance.

    Raises:
        RuntimeError: If session state has not been initialised.
    """
    raise NotImplementedError("Implement in TDD cycle")


def get_chat_history() -> list[dict]:
    """Get the UI-facing chat history from session state.

    Returns:
        List of {"role": str, "content": str, "metadata": dict} dicts.
    """
    raise NotImplementedError("Implement in TDD cycle")


def add_to_chat_history(role: str, content: str, metadata: dict | None = None) -> None:
    """Append a message to the UI chat history.

    Args:
        role: "user" or "assistant"
        content: The message text.
        metadata: Optional metadata (sources, MCP tools used).
    """
    raise NotImplementedError("Implement in TDD cycle")


def clear_chat_history() -> None:
    """Clear all chat history and reset the conversation context."""
    raise NotImplementedError("Implement in TDD cycle")
