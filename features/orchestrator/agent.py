"""Main LangChain agent orchestrating RAG and MCP tools.

The TravelAgent is the central coordinator that:
1. Analyses the user's query to determine the required information sources
2. Retrieves KB content via RAG for destination questions
3. Invokes MCP tools for real-time weather and currency data
4. Synthesises a grounded, attributed response using the LLM

Requirements satisfied: All of Section 4 (Core Features), Section 5 (Prompt Engineering).
"""

from dataclasses import dataclass
from enum import Enum


class QueryIntent(Enum):
    """Classified intent of a user query."""

    KB_ONLY = "kb_only"           # Only knowledge base needed (destination facts)
    MCP_WEATHER = "mcp_weather"   # Only weather MCP tool needed
    MCP_CURRENCY = "mcp_currency" # Only currency MCP tool needed
    COMBINED = "combined"          # Both KB and one or more MCP tools needed
    UNKNOWN = "unknown"            # Cannot classify — agent decides


@dataclass
class AgentResponse:
    """Structured response from the TravelAgent."""

    answer: str
    intent: QueryIntent
    kb_sources_used: list[dict]       # List of {"title": str, "url": str}
    mcp_tools_used: list[str]         # List of tool names used (e.g., ["get_weather_forecast"])
    has_fallback: bool                # True if KB or MCP had insufficient data
    fallback_message: str | None      # Fallback message if data was insufficient


class TravelAgent:
    """The main LangChain-powered travel planning agent.

    Orchestrates RAG retrieval, MCP tool calls, and LLM synthesis
    to answer user queries with grounded, attributed responses.

    Design principles (from prompt engineering requirements):
    - KB content is used for ALL destination facts
    - MCP tools are used for ALL real-time data
    - Responses clearly distinguish KB facts, MCP data, and AI suggestions
    - Missing information is stated clearly — never fabricated
    - User preferences from conversation history are preserved
    """

    def __init__(
        self,
        retriever: object,
        mcp_client: object,
        context_manager: object,
        llm: object,
    ) -> None:
        """Initialise the travel agent with all required components.

        Args:
            retriever: KnowledgeRetriever instance for semantic KB search.
            mcp_client: MCPClient instance with registered weather and currency tools.
            context_manager: ConversationContextManager for multi-turn context.
            llm: LangChain-compatible LLM instance (Google Gemini Pro).
        """
        self.retriever = retriever
        self.mcp_client = mcp_client
        self.context_manager = context_manager
        self.llm = llm

    def classify_intent(self, query: str) -> QueryIntent:
        """Classify the user's query to determine required information sources.

        Uses keyword heuristics + LLM classification to determine whether
        the query needs KB only, MCP only, or a combination.

        Args:
            query: The user's natural language question.

        Returns:
            QueryIntent enum value.
        """
        raise NotImplementedError("Implement in TDD cycle")

    def process_query(self, query: str) -> AgentResponse:
        """Process a user query and return a grounded, attributed response.

        This is the main entry point for the agent. It:
        1. Classifies the query intent
        2. Retrieves KB content if needed
        3. Invokes MCP tools if needed
        4. Synthesises the response with proper attribution
        5. Updates conversation history

        Args:
            query: The user's natural language question.

        Returns:
            AgentResponse with the answer and attribution metadata.
        """
        raise NotImplementedError("Implement in TDD cycle")

    def _handle_kb_query(self, query: str) -> AgentResponse:
        """Handle a query that only requires knowledge base retrieval.

        Args:
            query: The user's question about destination facts.

        Returns:
            AgentResponse grounded in KB content with source citations.
        """
        raise NotImplementedError("Implement in TDD cycle")

    def _handle_weather_query(self, query: str) -> AgentResponse:
        """Handle a query that requires weather MCP tool data.

        Args:
            query: The user's weather-related question.

        Returns:
            AgentResponse with weather data clearly labeled as MCP-sourced.
        """
        raise NotImplementedError("Implement in TDD cycle")

    def _handle_currency_query(self, query: str) -> AgentResponse:
        """Handle a query that requires currency conversion via MCP tool.

        Args:
            query: The user's currency conversion question.

        Returns:
            AgentResponse with conversion data clearly labeled as MCP-sourced.
        """
        raise NotImplementedError("Implement in TDD cycle")

    def _handle_combined_query(self, query: str) -> AgentResponse:
        """Handle a query requiring both KB retrieval and MCP tool calls.

        The primary combined scenario: weather-aware itinerary planning.

        Args:
            query: The user's combined destination + real-time question.

        Returns:
            AgentResponse combining KB facts and MCP data with clear attribution.
        """
        raise NotImplementedError("Implement in TDD cycle")
