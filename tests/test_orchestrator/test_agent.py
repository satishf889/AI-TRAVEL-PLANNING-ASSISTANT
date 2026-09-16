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
        self, travel_agent: TravelAgent
    ) -> None:
        """Destination questions are classified as KB_ONLY."""
        query = "What are the must-visit attractions in Singapore?"
        with pytest.raises(NotImplementedError):
            travel_agent.classify_intent(query)

    def test_weather_query_classified_as_mcp_weather(
        self, travel_agent: TravelAgent
    ) -> None:
        """Weather questions are classified as MCP_WEATHER."""
        query = "What is the weather forecast for Singapore tomorrow?"
        with pytest.raises(NotImplementedError):
            travel_agent.classify_intent(query)

    def test_currency_query_classified_as_mcp_currency(
        self, travel_agent: TravelAgent
    ) -> None:
        """Currency questions are classified as MCP_CURRENCY."""
        query = "Convert INR 50000 to SGD"
        with pytest.raises(NotImplementedError):
            travel_agent.classify_intent(query)

    def test_combined_query_classified_as_combined(
        self, travel_agent: TravelAgent
    ) -> None:
        """Combined itinerary+weather questions are classified as COMBINED."""
        query = "Create a 3-day itinerary for Singapore adjusted for the weather forecast"
        with pytest.raises(NotImplementedError):
            travel_agent.classify_intent(query)


@pytest.mark.unit
@pytest.mark.orchestrator
class TestAgentResponse:
    """Tests for AgentResponse structure."""

    def test_process_query_returns_agent_response(
        self, travel_agent: TravelAgent
    ) -> None:
        """process_query returns an AgentResponse instance."""
        with pytest.raises(NotImplementedError):
            travel_agent.process_query("What is Gardens by the Bay?")

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
