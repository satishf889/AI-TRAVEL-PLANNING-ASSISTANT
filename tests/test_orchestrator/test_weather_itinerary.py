"""Tests for weather-based day selection and source origin tagging in trip planning itineraries.

Requirements satisfied:
- Requirement 4.2 (Weather MCP Tool)
- Requirement 4.3 (Combined RAG + MCP Response: weather-adjusted itineraries with source attribution)
- Section 5 (Prompt Engineering Requirements: distinguish KB, MCP, and AI suggestions)
"""

from unittest.mock import MagicMock
import pytest

from features.orchestrator.agent import QueryIntent, TravelAgent
from features.orchestrator.prompt_templates import (
    COMBINED_RAG_MCP_PROMPT_TEMPLATE,
    RAG_QA_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)


@pytest.mark.unit
@pytest.mark.orchestrator
class TestWeatherItineraryPlanning:
    """Tests for intent routing and prompt guidelines when planning trips based on weather."""

    def test_trip_planning_with_weather_classified_as_combined(
        self, mock_retriever: MagicMock, mock_mcp_client: MagicMock, context_manager: MagicMock, mock_llm: MagicMock
    ) -> None:
        """Queries asking to plan a trip based on weather should be classified as COMBINED intent."""
        agent = TravelAgent(
            retriever=mock_retriever,
            mcp_client=mock_mcp_client,
            context_manager=context_manager,
            llm=mock_llm,
        )
        mock_message = MagicMock()
        mock_message.tool_calls = [{"name": "get_weather_forecast"}]
        mock_llm.bind_tools.return_value.invoke.return_value = mock_message

        query = "Plan my trip and consider the days which would help to decide based on weather"
        intent = agent.classify_intent(query)
        assert intent == QueryIntent.COMBINED

    def test_prompt_template_instructs_weather_day_decisions(self) -> None:
        """Prompt templates must instruct the LLM to make weather-based day decisions for activities."""
        combined_lower = COMBINED_RAG_MCP_PROMPT_TEMPLATE.lower()
        system_lower = SYSTEM_PROMPT.lower()

        assert "weather decision" in combined_lower or "weather-based" in combined_lower
        assert "outdoor" in combined_lower and "indoor" in combined_lower

    def test_prompt_template_instructs_source_origin_tagging(self) -> None:
        """Prompt templates must instruct tagging items with [KB], [MCP], and [AI Suggestion]."""
        combined_text = COMBINED_RAG_MCP_PROMPT_TEMPLATE
        system_text = SYSTEM_PROMPT

        assert "[KB]" in combined_text or "Knowledge Base" in combined_text
        assert "[MCP]" in combined_text or "Live Data" in combined_text
        assert "[AI Suggestion]" in combined_text or "AI Suggestion" in combined_text
