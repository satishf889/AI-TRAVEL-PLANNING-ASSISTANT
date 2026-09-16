from __future__ import annotations

"""Reusable Streamlit UI components for the AI Travel Planning Assistant."""



def render_chat_message(role: str, content: str, metadata: dict | None = None) -> None:
    """Render a single chat message with role-appropriate styling.

    Args:
        role: "user" or "assistant"
        content: The message text (supports Markdown).
        metadata: Optional metadata containing sources and MCP info.
    """
    raise NotImplementedError("Implement in TDD cycle")


def render_source_citations(sources: list[dict]) -> None:
    """Render a collapsible source citation section.

    Args:
        sources: List of {"title": str, "url": str} dicts from KB retrieval.
    """
    raise NotImplementedError("Implement in TDD cycle")


def render_mcp_tool_badge(tool_names: list[str]) -> None:
    """Render a badge indicating which MCP tools were used in a response.

    Args:
        tool_names: List of MCP tool names used (e.g., ["get_weather_forecast"]).
    """
    raise NotImplementedError("Implement in TDD cycle")


def render_sidebar() -> None:
    """Render the application sidebar with settings and info."""
    raise NotImplementedError("Implement in TDD cycle")


def render_example_queries() -> list[str]:
    """Render example query buttons and return the selected query if clicked.

    Returns:
        The selected example query string, or empty string if none clicked.
    """
    raise NotImplementedError("Implement in TDD cycle")
