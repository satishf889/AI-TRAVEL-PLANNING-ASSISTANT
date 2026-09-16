from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch, ANY

from features.ui.app import main, render_chat_page


@patch("features.ui.app.st")
@patch("features.ui.app.initialise_session_state")
@patch("features.ui.app.render_sidebar")
@patch("features.ui.app.render_chat_page")
@patch("features.ui.app.trip_planning_form")
def test_main_calls_initialise(
    mock_trip_form: MagicMock,
    mock_render_chat_page: MagicMock,
    mock_sidebar: MagicMock,
    mock_init: MagicMock,
    mock_st: MagicMock,
) -> None:
    """Test that main initializes state and renders the base layout."""
    mock_st.session_state = {"current_page": "Chat"}
    mock_st.columns.return_value = [MagicMock(), MagicMock()]

    main()

    mock_st.set_page_config.assert_called_once()
    mock_init.assert_called_once()
    mock_sidebar.assert_called_once()
    mock_render_chat_page.assert_called_once()


@patch("features.ui.app.st")
@patch("features.ui.app.get_chat_history")
@patch("features.ui.app.add_to_chat_history")
@patch("features.ui.app.render_chat_message")
@patch("features.ui.app.get_agent")
def test_render_chat_page_processes_user_input(
    mock_get_agent: MagicMock,
    mock_render_msg: MagicMock,
    mock_add_history: MagicMock,
    mock_history: MagicMock,
    mock_st: MagicMock,
) -> None:
    """Test that render_chat_page processes user chat input and renders responses."""
    mock_history.return_value = [{"role": "user", "content": "prev"}]
    mock_st.chat_input.return_value = "What is the weather?"
    mock_st.session_state = {}

    mock_agent = MagicMock()
    def fake_stream(q: str, metadata: dict[str, Any]) -> Any:
        metadata["kb_sources_used"] = [{"title": "Visit SG", "url": "http://sg"}]
        metadata["mcp_tools_used"] = ["get_weather_forecast"]
        yield "Sunny 31C"

    mock_agent.stream_query.side_effect = fake_stream
    mock_get_agent.return_value = mock_agent

    render_chat_page()

    assert mock_add_history.call_count == 2
    assert mock_render_msg.call_count == 2  # 1 from history + 1 user


@patch("features.ui.app.st")
@patch("features.ui.app.get_chat_history")
@patch("features.ui.app.add_to_chat_history")
@patch("features.ui.app.render_chat_message")
@patch("features.ui.app.get_agent")
def test_render_chat_page_handles_agent_exception(
    mock_get_agent: MagicMock,
    mock_render_msg: MagicMock,
    mock_add_history: MagicMock,
    mock_history: MagicMock,
    mock_st: MagicMock,
) -> None:
    """Test that render_chat_page handles agent runtime errors gracefully."""
    mock_history.return_value = []
    mock_st.chat_input.return_value = "Test Error"
    mock_st.session_state = {}

    mock_agent = MagicMock()
    mock_agent.stream_query.side_effect = RuntimeError("API failed")
    mock_get_agent.return_value = mock_agent

    render_chat_page()

    mock_st.error.assert_called_once()
