from __future__ import annotations

from unittest.mock import MagicMock, patch

from features.ui.components import (
    render_chat_message,
    render_example_queries,
    render_mcp_tool_badge,
    render_sidebar,
    render_source_citations,
)


@patch("features.ui.components.render_source_citations")
@patch("features.ui.components.render_mcp_tool_badge")
@patch("features.ui.components.st")
def test_render_chat_message(
    mock_st: MagicMock,
    mock_badge: MagicMock,
    mock_citations: MagicMock,
) -> None:
    """Test rendering a chat message with metadata."""
    # Setup mock context manager for st.chat_message
    mock_chat = MagicMock()
    mock_st.chat_message.return_value.__enter__.return_value = mock_chat

    metadata = {
        "mcp_tools_used": ["weather"],
        "kb_sources_used": [{"title": "Wiki", "url": "http"}]
    }

    render_chat_message("assistant", "Hello", metadata)

    mock_st.chat_message.assert_called_with("assistant")
    mock_st.markdown.assert_called_with("Hello")
    mock_badge.assert_called_with(["weather"])
    mock_citations.assert_called_with([{"title": "Wiki", "url": "http"}])

@patch("features.ui.components.st")
def test_render_source_citations(mock_st: MagicMock) -> None:
    """Test rendering source citations."""
    mock_expander = MagicMock()
    mock_st.expander.return_value.__enter__.return_value = mock_expander

    sources = [{"title": "Test Docs", "url": "http://test"}]

    render_source_citations(sources)

    mock_st.expander.assert_called_once_with("📚 View Sources")
    mock_st.markdown.assert_called_once_with("- [Test Docs](http://test)")

@patch("features.ui.components.st")
def test_render_mcp_tool_badge(mock_st: MagicMock) -> None:
    """Test rendering MCP tool badges."""
    tools = ["get_weather_forecast"]

    render_mcp_tool_badge(tools)

    mock_st.caption.assert_called_once_with("🛠️ **Tools used:** get_weather_forecast")

@patch("features.ui.components.st")
@patch("features.ui.components.clear_chat_history")
def test_render_sidebar(mock_clear: MagicMock, mock_st: MagicMock) -> None:
    """Test sidebar rendering and clear button."""
    mock_sidebar = MagicMock()
    mock_st.sidebar = mock_sidebar
    mock_st.sidebar.button.return_value = True

    render_sidebar()

    mock_st.sidebar.button.assert_called_once_with("🗑️ Clear Chat History", use_container_width=True)
    mock_clear.assert_called_once()

@patch("features.ui.components.st")
def test_render_example_queries(mock_st: MagicMock) -> None:
    """Test rendering example query buttons."""
    # Return True for the first button
    mock_st.button.side_effect = [True, False, False]

    result = render_example_queries()

    # 3 buttons should be created (columns)
    assert mock_st.columns.called
    assert result != "" # First query string
