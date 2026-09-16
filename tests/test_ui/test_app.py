from __future__ import annotations

from unittest.mock import MagicMock, patch

from features.ui.app import main


@patch("features.ui.app.st")
@patch("features.ui.app.initialise_session_state")
@patch("features.ui.app.render_sidebar")
@patch("features.ui.app.render_example_queries")
@patch("features.ui.app.get_chat_history")
def test_main_calls_initialise(
    mock_history: MagicMock,
    mock_examples: MagicMock,
    mock_sidebar: MagicMock,
    mock_init: MagicMock,
    mock_st: MagicMock,
) -> None:
    """Test that main initializes state and renders the base layout."""
    # Setup
    mock_history.return_value = []
    mock_examples.return_value = ""
    mock_st.chat_input.return_value = None

    # Execute
    main()

    # Assert
    mock_st.set_page_config.assert_called_once()
    mock_init.assert_called_once()
    mock_sidebar.assert_called_once()
    mock_history.assert_called_once()
    mock_st.chat_input.assert_called_once()


@patch("features.ui.app.st")
@patch("features.ui.app.initialise_session_state")
@patch("features.ui.app.render_sidebar")
@patch("features.ui.app.get_chat_history")
@patch("features.ui.app.add_to_chat_history")
@patch("features.ui.app.render_chat_message")
@patch("features.ui.app.get_agent")
def test_main_processes_user_input(
    mock_get_agent: MagicMock,
    mock_render_msg: MagicMock,
    mock_add_history: MagicMock,
    mock_history: MagicMock,
    mock_sidebar: MagicMock,
    mock_init: MagicMock,
    mock_st: MagicMock,
) -> None:
    """Test that main processes user chat input and renders responses."""
    # Setup
    mock_history.return_value = [{"role": "user", "content": "prev"}]
    mock_st.chat_input.return_value = "What is the weather?"

    mock_agent = MagicMock()
    mock_response = MagicMock()
    mock_response.answer = "Sunny 31C"
    mock_response.kb_sources_used = [{"title": "Visit SG", "url": "http://sg"}]
    mock_response.mcp_tools_used = ["get_weather_forecast"]
    mock_response.has_fallback = False
    mock_response.fallback_message = None
    mock_agent.process_query.return_value = mock_response
    mock_get_agent.return_value = mock_agent

    # Execute
    main()

    # Assert
    mock_agent.process_query.assert_called_once_with("What is the weather?")
    assert mock_add_history.call_count == 2
    assert mock_render_msg.call_count == 3  # 1 from history + 1 user + 1 assistant


@patch("features.ui.app.st")
@patch("features.ui.app.initialise_session_state")
@patch("features.ui.app.render_sidebar")
@patch("features.ui.app.get_chat_history")
@patch("features.ui.app.add_to_chat_history")
@patch("features.ui.app.render_chat_message")
@patch("features.ui.app.get_agent")
def test_main_handles_agent_exception(
    mock_get_agent: MagicMock,
    mock_render_msg: MagicMock,
    mock_add_history: MagicMock,
    mock_history: MagicMock,
    mock_sidebar: MagicMock,
    mock_init: MagicMock,
    mock_st: MagicMock,
) -> None:
    """Test that main handles agent runtime errors gracefully."""
    mock_history.return_value = []
    mock_st.chat_input.return_value = "Test Error"

    mock_agent = MagicMock()
    mock_agent.process_query.side_effect = RuntimeError("API failed")
    mock_get_agent.return_value = mock_agent

    main()

    mock_st.error.assert_called_once()

