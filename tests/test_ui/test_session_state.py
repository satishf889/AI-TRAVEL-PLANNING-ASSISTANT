from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from features.ui.session_state import (
    add_to_chat_history,
    clear_chat_history,
    get_agent,
    get_chat_history,
    initialise_session_state,
)


@patch("features.ui.session_state.st")
@patch("features.ui.session_state.get_agent_dependencies")
def test_initialise_session_state(mock_deps: MagicMock, mock_st: MagicMock) -> None:
    """Test session state initialization when empty."""
    # Setup
    mock_st.session_state = {}
    mock_deps.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

    # Execute
    initialise_session_state()

    # Assert
    assert "messages" in mock_st.session_state
    assert "agent" in mock_st.session_state
    assert isinstance(mock_st.session_state["messages"], list)
    assert len(mock_st.session_state["messages"]) == 0


@patch("features.ui.session_state.st")
def test_initialise_session_state_idempotent(mock_st: MagicMock) -> None:
    """Test that existing state is not overwritten."""
    mock_st.session_state = {
        "messages": [{"role": "user", "content": "hello"}],
        "agent": MagicMock(),
    }

    initialise_session_state()

    assert len(mock_st.session_state["messages"]) == 1
    assert mock_st.session_state["messages"][0]["content"] == "hello"

@patch("features.ui.session_state.st")
def test_get_agent_success(mock_st: MagicMock) -> None:
    """Test retrieving agent successfully."""
    mock_agent = MagicMock()
    mock_st.session_state = {"agent": mock_agent}

    agent = get_agent()

    assert agent == mock_agent

@patch("features.ui.session_state.st")
def test_get_agent_raises_error(mock_st: MagicMock) -> None:
    """Test retrieving agent when not initialized raises error."""
    mock_st.session_state = {}

    with pytest.raises(RuntimeError):
        get_agent()

@patch("features.ui.session_state.st")
def test_get_chat_history(mock_st: MagicMock) -> None:
    """Test retrieving chat history."""
    mock_st.session_state = {"messages": [{"role": "assistant", "content": "hi"}]}

    history = get_chat_history()

    assert history == [{"role": "assistant", "content": "hi"}]

@patch("features.ui.session_state.st")
def test_add_to_chat_history(mock_st: MagicMock) -> None:
    """Test adding a message to history."""
    mock_st.session_state = {"messages": []}

    add_to_chat_history("user", "my query", {"source": "test"})

    assert len(mock_st.session_state["messages"]) == 1
    assert mock_st.session_state["messages"][0]["role"] == "user"
    assert mock_st.session_state["messages"][0]["content"] == "my query"
    assert mock_st.session_state["messages"][0]["metadata"] == {"source": "test"}

@patch("features.ui.session_state.st")
def test_clear_chat_history(mock_st: MagicMock) -> None:
    """Test clearing history and resetting agent context."""
    mock_agent = MagicMock()
    mock_st.session_state = {
        "messages": [{"role": "user", "content": "foo"}],
        "agent": mock_agent,
    }

    clear_chat_history()

    assert len(mock_st.session_state["messages"]) == 0
    mock_agent.context_manager.clear.assert_called_once()


def test_get_agent_dependencies_preview() -> None:
    """Test get_agent_dependencies returns configured interactive preview stubs."""
    from features.ui.session_state import get_agent_dependencies

    retriever, mcp_client, context_manager, llm = get_agent_dependencies()

    # Verify retriever
    docs = retriever.invoke("Singapore attractions")
    assert len(docs) > 0
    assert "Visit Singapore" in docs[0].metadata["title"]

    # Verify MCP client
    weather = mcp_client.get_weather_forecast("Singapore")
    assert "Singapore" in weather
    currency = mcp_client.convert_currency("50000 INR")
    assert "SGD" in currency

    # Verify Bound LLM tool calls
    bound = llm.bind_tools([])
    weather_resp = bound.invoke("What is the weather like?")
    assert any(c["name"] == "get_weather_forecast" for c in weather_resp.tool_calls)

    currency_resp = bound.invoke("Convert 50000 INR to SGD")
    assert any(c["name"] == "convert_currency" for c in currency_resp.tool_calls)

    itinerary_resp = bound.invoke("Plan a 3-day itinerary")
    assert any(c["name"] == "kb_search" for c in itinerary_resp.tool_calls)

    # Verify LLM invoke responses
    assert "Weather" in llm.invoke("live weather data: Singapore").content
    assert "Currency" in llm.invoke("live conversion data: SGD").content
    assert "Itinerary" in llm.invoke("3-day itinerary").content
    assert "Singapore" in llm.invoke("general question").content

