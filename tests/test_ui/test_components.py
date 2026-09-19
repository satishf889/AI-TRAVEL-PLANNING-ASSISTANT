from __future__ import annotations

from unittest.mock import MagicMock, patch

from features.ui.components import (
    render_chat_message,
    render_example_queries,
    render_mcp_tool_badge,
    render_sidebar,
    should_show_save_options,
)


def test_should_show_save_options_denial() -> None:
    """Should return False when message is an out-of-scope refusal."""
    content = "I can only help for Singapore travel, no other thing."
    assert not should_show_save_options("assistant", content)


def test_should_show_save_options_weather_intent() -> None:
    """Should return False when message intent is mcp_weather."""
    content = "The current weather in Singapore is 30°C and sunny."
    metadata = {"intent": "mcp_weather", "mcp_tools_used": ["get_weather_forecast"]}
    assert not should_show_save_options("assistant", content, metadata)


def test_should_show_save_options_currency_intent() -> None:
    """Should return False when message intent is mcp_currency."""
    content = "100 USD is equivalent to 134.50 SGD."
    metadata = {"intent": "mcp_currency", "mcp_tools_used": ["convert_currency"]}
    assert not should_show_save_options("assistant", content, metadata)


def test_should_show_save_options_user_role() -> None:
    """Should return False for user messages."""
    assert not should_show_save_options("user", "Hello")


def test_should_show_save_options_valid_itinerary() -> None:
    """Should return True for Singapore travel itineraries and facts."""
    content = "🏛️ Destination Facts (Knowledge Base)\n- Gardens by the Bay\n💡 AI Suggestion: Visit at night."
    metadata = {"intent": "kb_only", "kb_sources_used": [{"title": "Wiki", "url": "http"}]}
    assert should_show_save_options("assistant", content, metadata)


@patch("features.ui.components.st")
@patch("features.ui.components.render_mcp_tool_badge")
def test_render_chat_message(
    mock_badge: MagicMock,
    mock_st: MagicMock,
) -> None:
    """Test rendering a chat message with metadata."""
    # Setup mock context manager for st.chat_message
    mock_chat = MagicMock()
    mock_st.chat_message.return_value.__enter__.return_value = mock_chat
    mock_st.columns.return_value = [MagicMock(), MagicMock()]

    metadata = {
        "mcp_tools_used": ["weather"],
        "kb_sources_used": [{"title": "Wiki", "url": "http"}],
    }

    render_chat_message("assistant", "Hello", metadata)

    mock_st.chat_message.assert_called_with("assistant", avatar="✈️")
    mock_st.markdown.assert_called_with("Hello")
    mock_badge.assert_called_with(["weather"])


@patch("features.ui.components.st")
def test_render_mcp_tool_badge(mock_st: MagicMock) -> None:
    """Test rendering MCP tool badges."""
    tools = ["get_weather_forecast"]

    render_mcp_tool_badge(tools)

    mock_st.caption.assert_called_once_with("🛠️ **Tools used:** get_weather_forecast")


@patch("features.ui.components.st")
@patch("features.ui.components.clear_chat_history")
def test_render_sidebar(mock_clear: MagicMock, mock_st: MagicMock) -> None:
    """Test sidebar rendering with clear and stop buttons."""
    mock_sidebar = MagicMock()
    mock_st.sidebar = mock_sidebar
    mock_st.sidebar.button.side_effect = [False] * 5 + [True] + [False]

    render_sidebar()

    assert mock_st.sidebar.button.call_count == 7
    mock_clear.assert_called_once()


@patch("features.ui.components.st")
def test_render_example_queries(mock_st: MagicMock) -> None:
    """Test rendering example query buttons."""
    # Return True for the first button
    mock_st.button.side_effect = [True, False, False]

    result = render_example_queries()

    # 3 buttons should be created (columns)
    assert mock_st.columns.called
    assert result != ""  # First query string
