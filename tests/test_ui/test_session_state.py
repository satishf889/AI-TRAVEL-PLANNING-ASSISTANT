"""Tests for Streamlit UI session state management."""

from unittest.mock import MagicMock, patch

import pytest
import streamlit as st

from features.ui.session_state import (
    add_to_chat_history,
    clear_chat_history,
    get_agent,
    get_agent_dependencies,
    get_chat_history,
    initialise_session_state,
)


@pytest.fixture(autouse=True)
def reset_session_state():
    """Clear Streamlit session state before each test."""
    st.session_state.clear()


@pytest.mark.unit
@pytest.mark.ui
class TestSessionStateInit:

    @patch("features.ui.session_state.get_agent_dependencies")
    def test_initialise_creates_empty_messages(self, mock_get_deps):
        mock_get_deps.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        initialise_session_state()
        assert "messages" in st.session_state
        assert st.session_state["messages"] == []

    @patch("features.ui.session_state.get_agent_dependencies")
    def test_initialise_creates_agent(self, mock_get_deps):
        mock_get_deps.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        initialise_session_state()
        assert "agent" in st.session_state
        assert st.session_state["agent"] is not None

    def test_get_agent_raises_if_not_initialized(self):
        with pytest.raises(RuntimeError):
            get_agent()


@pytest.mark.unit
@pytest.mark.ui
class TestChatHistory:

    def test_add_to_chat_history_creates_messages_if_missing(self):
        add_to_chat_history("user", "Hello")
        assert len(st.session_state["messages"]) == 1
        assert st.session_state["messages"][0]["role"] == "user"

    def test_clear_chat_history_resets_messages(self):
        st.session_state["messages"] = [{"role": "user", "content": "hi"}]
        clear_chat_history()
        assert st.session_state["messages"] == []

    def test_clear_chat_history_clears_context_manager(self):
        mock_ctx = MagicMock()
        mock_agent = MagicMock()
        mock_agent.context_manager = mock_ctx
        st.session_state["agent"] = mock_agent
        
        clear_chat_history()
        mock_ctx.clear.assert_called_once()


@pytest.mark.unit
@pytest.mark.ui
class TestAgentDependencies:

    @patch("features.ui.session_state.settings")
    @patch("features.ui.session_state.VectorStoreManager")
    @patch("features.ui.session_state.ChatGoogleGenerativeAI")
    def test_get_agent_dependencies_returns_real_objects(self, mock_llm, mock_vsm, mock_settings):
        # We need to mock settings so it doesn't fail trying to read env vars
        mock_settings.chroma_persist_directory = MagicMock()
        mock_settings.chroma_collection_name = "test"
        mock_settings.embedding_model_name = "test"
        mock_settings.google_api_key = "test_key"
        
        # Test the function returns real types
        retriever, mcp_client, context_manager, llm = get_agent_dependencies()
        
        from features.rag.retriever import KnowledgeRetriever
        from features.mcp.mcp_client import MCPClient
        from features.orchestrator.context_manager import ConversationContextManager
        
        assert isinstance(retriever, KnowledgeRetriever)
        assert isinstance(mcp_client, MCPClient)
        assert isinstance(context_manager, ConversationContextManager)
        # LLM is mocked
