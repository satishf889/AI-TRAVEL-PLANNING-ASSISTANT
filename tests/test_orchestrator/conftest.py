from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_retriever():
    """Mock for RAG Retriever."""
    retriever = MagicMock()
    retriever.invoke.return_value = []
    return retriever

@pytest.fixture
def mock_mcp_client():
    """Mock for MCP Client."""
    client = MagicMock()
    client.get_weather_forecast.return_value = "Sunny, 30C"
    client.convert_currency.return_value = "SGD 100"
    return client

@pytest.fixture
def context_manager():
    """Mock for ConversationContextManager."""
    cm = MagicMock()
    cm.get_recent_context_summary.return_value = "No preferences."
    return cm

@pytest.fixture
def mock_llm():
    """Mock for LangChain LLM."""
    llm = MagicMock()
    llm.invoke.return_value.content = "Mocked LLM response"
    return llm
