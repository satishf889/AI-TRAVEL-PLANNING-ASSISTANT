"""Tests for TravelAgent and intent classification — Orchestrator module.

TDD: These tests drive the implementation of agent.py.
All external calls (LLM, RAG, MCP) are mocked.
"""

from unittest.mock import MagicMock

import pytest

from features.orchestrator.agent import AgentResponse, QueryIntent, TravelAgent


@pytest.fixture
def travel_agent(
    mock_retriever: MagicMock,
    mock_mcp_client: MagicMock,
    context_manager: MagicMock,
    mock_llm: MagicMock,
) -> TravelAgent:
    """A TravelAgent instance with all dependencies mocked."""
    return TravelAgent(
        retriever=mock_retriever,
        mcp_client=mock_mcp_client,
        context_manager=context_manager,
        llm=mock_llm,
    )


@pytest.mark.unit
@pytest.mark.orchestrator
class TestIntentClassification:
    """Tests for query intent classification."""

    def test_destination_query_classified_as_kb_only(
        self, travel_agent: TravelAgent, mock_llm: MagicMock
    ) -> None:
        """Destination questions are classified as KB_ONLY."""
        mock_message = MagicMock()
        mock_message.tool_calls = []
        mock_message.content = "mcp is not needed"
        mock_llm.bind_tools.return_value.invoke.return_value = mock_message

        query = "What are the must-visit attractions in Singapore?"
        intent = travel_agent.classify_intent(query)
        assert intent == QueryIntent.KB_ONLY

    def test_weather_query_classified_as_mcp_weather(
        self, travel_agent: TravelAgent, mock_llm: MagicMock
    ) -> None:
        """Weather questions are classified as MCP_WEATHER."""
        mock_message = MagicMock()
        mock_message.tool_calls = [{"name": "get_weather_forecast"}]
        mock_llm.bind_tools.return_value.invoke.return_value = mock_message

        query = "What is the weather forecast for Singapore tomorrow?"
        intent = travel_agent.classify_intent(query)
        assert intent == QueryIntent.MCP_WEATHER

    def test_currency_query_classified_as_mcp_currency(
        self, travel_agent: TravelAgent, mock_llm: MagicMock
    ) -> None:
        """Currency questions are classified as MCP_CURRENCY."""
        mock_message = MagicMock()
        mock_message.tool_calls = [{"name": "convert_currency"}]
        mock_llm.bind_tools.return_value.invoke.return_value = mock_message

        query = "Convert INR 50000 to SGD"
        intent = travel_agent.classify_intent(query)
        assert intent == QueryIntent.MCP_CURRENCY

    def test_combined_query_classified_as_combined(
        self, travel_agent: TravelAgent, mock_llm: MagicMock
    ) -> None:
        """Combined itinerary+weather questions are classified as COMBINED."""
        mock_message = MagicMock()
        mock_message.tool_calls = [{"name": "get_weather_forecast"}, {"name": "kb_search"}]
        mock_llm.bind_tools.return_value.invoke.return_value = mock_message

        query = "Create a 3-day itinerary for Singapore adjusted for the weather forecast"
        intent = travel_agent.classify_intent(query)
        assert intent == QueryIntent.COMBINED


@pytest.mark.unit
@pytest.mark.orchestrator
class TestAgentResponse:
    """Tests for AgentResponse structure."""

    def test_process_query_kb_only(
        self, travel_agent: TravelAgent, mock_retriever: MagicMock, mock_llm: MagicMock
    ) -> None:
        """process_query handles KB_ONLY routing."""
        # Setup mocks
        travel_agent.classify_intent = MagicMock(return_value=QueryIntent.KB_ONLY)
        mock_doc = MagicMock()
        mock_doc.page_content = "Gardens by the Bay is a nature park."
        mock_doc.metadata = {"title": "Visit Singapore", "source": "https://visitsingapore.com"}
        mock_retriever.invoke.return_value = [mock_doc]
        mock_llm.invoke.return_value.content = "Gardens by the Bay is a great park."

        response = travel_agent.process_query("What is Gardens by the Bay?")

        assert isinstance(response, AgentResponse)
        assert response.intent == QueryIntent.KB_ONLY
        assert len(response.kb_sources_used) == 1
        assert response.kb_sources_used[0]["title"] == "Visit Singapore"
        assert "Gardens by the Bay" in response.answer
        mock_retriever.invoke.assert_called_once()

    def test_process_query_combined(
        self, travel_agent: TravelAgent, mock_retriever: MagicMock, mock_mcp_client: MagicMock, mock_llm: MagicMock
    ) -> None:
        """process_query handles COMBINED routing."""
        travel_agent.classify_intent = MagicMock(return_value=QueryIntent.COMBINED)
        mock_doc = MagicMock()
        mock_doc.page_content = "Some content"
        mock_doc.metadata = {"title": "Visit Singapore", "source": "url"}
        mock_retriever.invoke.return_value = [mock_doc]
        mock_mcp_client.get_weather_forecast.return_value = "Sunny"
        mock_llm.invoke.return_value.content = "Here is your sunny itinerary."

        response = travel_agent.process_query("Give me a 3-day itinerary adjusted for weather")

        assert response.intent == QueryIntent.COMBINED
        assert len(response.kb_sources_used) == 1
        assert "get_weather_forecast" in response.mcp_tools_used
        mock_retriever.invoke.assert_called_once()
        mock_mcp_client.get_weather_forecast.assert_called_once()

    def test_agent_response_has_kb_sources(self) -> None:
        """AgentResponse.kb_sources_used is a list of dicts."""
        response = AgentResponse(
            answer="Test answer",
            intent=QueryIntent.KB_ONLY,
            kb_sources_used=[{"title": "Test", "url": "https://example.com"}],
            mcp_tools_used=[],
            has_fallback=False,
            fallback_message=None,
        )
        assert len(response.kb_sources_used) == 1
        assert "title" in response.kb_sources_used[0]
        assert "url" in response.kb_sources_used[0]

    def test_agent_response_mcp_only_has_empty_kb_sources(self) -> None:
        """A pure MCP response has empty kb_sources_used."""
        response = AgentResponse(
            answer="Weather: sunny, 31°C",
            intent=QueryIntent.MCP_WEATHER,
            kb_sources_used=[],
            mcp_tools_used=["get_weather_forecast"],
            has_fallback=False,
            fallback_message=None,
        )
        assert response.kb_sources_used == []
        assert "get_weather_forecast" in response.mcp_tools_used


@pytest.mark.unit
@pytest.mark.orchestrator
class TestFallbackBehavior:
    """Tests for fallback behavior when KB or MCP data is unavailable."""

    def test_fallback_response_has_fallback_flag(self) -> None:
        """A fallback response has has_fallback=True and a fallback message."""
        response = AgentResponse(
            answer="I could not find that information in the knowledge base.",
            intent=QueryIntent.KB_ONLY,
            kb_sources_used=[],
            mcp_tools_used=[],
            has_fallback=True,
            fallback_message="Please check visitsingapore.com for accurate information.",
        )
        assert response.has_fallback is True
        assert response.fallback_message is not None

    def test_fallback_no_kb_data(self, travel_agent: TravelAgent, mock_retriever: MagicMock) -> None:
        """When retriever returns empty, response has_fallback=True."""
        travel_agent.classify_intent = MagicMock(return_value=QueryIntent.KB_ONLY)
        mock_retriever.invoke.return_value = []

        response = travel_agent.process_query("Tell me about an unknown place")

        assert response.has_fallback is True
        assert "searched my Singapore travel knowledge base" in response.fallback_message

    def test_fallback_mcp_failure_combined(self, travel_agent: TravelAgent, mock_retriever: MagicMock, mock_mcp_client: MagicMock, mock_llm: MagicMock) -> None:
        """When MCP fails on COMBINED query, return KB response anyway without total failure."""
        travel_agent.classify_intent = MagicMock(return_value=QueryIntent.COMBINED)
        mock_doc = MagicMock()
        mock_doc.page_content = "Some content"
        mock_doc.metadata = {"title": "Visit Singapore", "source": "url"}
        mock_retriever.invoke.return_value = [mock_doc]

        mock_mcp_client.get_weather_forecast.side_effect = Exception("API Down")
        mock_llm.invoke.return_value.content = "Here is your itinerary without weather."

        response = travel_agent.process_query("Give me 3-day itinerary and adjust for weather")

        assert response.intent == QueryIntent.COMBINED
        assert len(response.kb_sources_used) == 1
        assert response.has_fallback is True
        assert "get_weather_forecast" not in response.mcp_tools_used
        assert "Here is your itinerary" in response.answer

    def test_fallback_mcp_failure_only(self, travel_agent: TravelAgent, mock_mcp_client: MagicMock) -> None:
        """When MCP fails on MCP-only query, return FALLBACK_MCP_TOOL_FAILURE."""
        travel_agent.classify_intent = MagicMock(return_value=QueryIntent.MCP_WEATHER)
        mock_mcp_client.get_weather_forecast.side_effect = Exception("API Down")

        response = travel_agent.process_query("What is the weather?")

        assert response.has_fallback is True
        assert response.fallback_message is not None
        assert "weather" in response.fallback_message.lower()
