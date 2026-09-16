from __future__ import annotations

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
from typing import Any

from features.orchestrator.prompt_templates import (
    COMBINED_RAG_MCP_PROMPT_TEMPLATE,
    FALLBACK_MCP_TOOL_FAILURE,
    FALLBACK_NO_KB_CONTENT,
    RAG_QA_PROMPT_TEMPLATE,
)


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
    kb_sources_used: list[dict[str, Any]]       # List of {"title": str, "url": str}
    mcp_tools_used: list[str]         # List of tool names used (e.g., ["get_weather_forecast"])
    has_fallback: bool                # True if KB or MCP had insufficient data
    fallback_message: str | None      # Fallback message if data was insufficient


class TravelAgent:
    """The main LangChain-powered travel planning agent."""

    def __init__(
        self,
        retriever: Any,
        mcp_client: Any,
        context_manager: Any,
        llm: Any,
    ) -> None:
        self.retriever = retriever
        self.mcp_client = mcp_client
        self.context_manager = context_manager
        self.llm = llm

    def _extract_text(self, content: str | list[dict[str, Any]] | list[Any]) -> str:
        """Extract text from LLM content which might be a string or a list of dicts."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts)
        return str(content)

    def classify_intent(self, query: str) -> QueryIntent:
        """Classify the user's query to determine required information sources."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather_forecast",
                    "description": "Get the current weather forecast for Singapore"
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "convert_currency",
                    "description": "Convert money between currencies"
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "kb_search",
                    "description": "Search the knowledge base for destination facts and itineraries"
                }
            }
        ]

        # In a real LangChain setup, bind_tools returns a runnable.
        llm_with_tools = self.llm.bind_tools(tools)
        response = llm_with_tools.invoke(query)

        tool_calls = getattr(response, "tool_calls", [])
        tool_names = [call.get("name", call) if isinstance(call, dict) else call.name for call in tool_calls]

        if not tool_names:
            return QueryIntent.KB_ONLY

        has_weather = "get_weather_forecast" in tool_names
        has_currency = "convert_currency" in tool_names
        has_kb = "kb_search" in tool_names

        if has_weather and has_kb:
            return QueryIntent.COMBINED
        elif has_currency and has_kb:
            return QueryIntent.COMBINED
        elif has_weather and not has_currency:
            return QueryIntent.MCP_WEATHER
        elif has_currency and not has_weather:
            return QueryIntent.MCP_CURRENCY

        return QueryIntent.COMBINED

    def process_query(self, query: str) -> AgentResponse:
        """Process a user query and return a grounded, attributed response."""
        self.context_manager.update_preferences(query)
        self.context_manager.add_user_message(query)

        intent = self.classify_intent(query)

        if intent == QueryIntent.KB_ONLY:
            response = self._handle_kb_query(query)
        elif intent == QueryIntent.MCP_WEATHER:
            response = self._handle_weather_query(query)
        elif intent == QueryIntent.MCP_CURRENCY:
            response = self._handle_currency_query(query)
        else:
            response = self._handle_combined_query(query)

        self.context_manager.add_assistant_message(response.answer)
        return response

    def _extract_doc_content(self, doc: Any) -> str:
        """Extract text content from a document or retrieval result."""
        if hasattr(doc, "page_content"):
            return str(doc.page_content)
        elif hasattr(doc, "content"):
            return str(doc.content)
        return str(doc)

    def _extract_doc_source(self, doc: Any) -> dict[str, str]:
        """Extract source metadata from a document or retrieval result."""
        if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
            title = doc.metadata.get("title") or doc.metadata.get("source_title") or "Unknown"
            url = doc.metadata.get("source") or doc.metadata.get("source_url") or ""
            return {"title": str(title), "url": str(url)}
        elif hasattr(doc, "source_title") and hasattr(doc, "source_url"):
            return {"title": str(doc.source_title), "url": str(doc.source_url)}
        return {"title": "Unknown", "url": ""}

    def _retrieve_docs(self, query: str) -> list[Any]:
        """Retrieve documents from the retriever using invoke or retrieve."""
        if hasattr(self.retriever, "invoke"):
            return self.retriever.invoke(query)  # type: ignore[no-any-return]
        elif hasattr(self.retriever, "retrieve"):
            return self.retriever.retrieve(query)  # type: ignore[no-any-return]
        return []

    def _handle_kb_query(self, query: str) -> AgentResponse:
        """Handle a query that only requires knowledge base retrieval."""
        docs = self._retrieve_docs(query)
        if not docs:
            return AgentResponse(
                answer="",
                intent=QueryIntent.KB_ONLY,
                kb_sources_used=[],
                mcp_tools_used=[],
                has_fallback=True,
                fallback_message=FALLBACK_NO_KB_CONTENT
            )

        context = "\n\n".join(self._extract_doc_content(doc) for doc in docs)
        sources = [self._extract_doc_source(doc) for doc in docs]

        prompt = RAG_QA_PROMPT_TEMPLATE.format(context=context, question=query)
        answer = self._extract_text(self.llm.invoke(prompt).content)

        return AgentResponse(
            answer=answer,
            intent=QueryIntent.KB_ONLY,
            kb_sources_used=sources,
            mcp_tools_used=[],
            has_fallback=False,
            fallback_message=None
        )

    def _handle_weather_query(self, query: str) -> AgentResponse:
        """Handle a query that requires weather MCP tool data."""
        try:
            weather = self.mcp_client.get_weather_forecast("Singapore")
            prompt = f"Answer using this live weather data: {weather}\nUser: {query}"
            answer = self._extract_text(self.llm.invoke(prompt).content)
            return AgentResponse(
                answer=answer,
                intent=QueryIntent.MCP_WEATHER,
                kb_sources_used=[],
                mcp_tools_used=["get_weather_forecast"],
                has_fallback=False,
                fallback_message=None
            )
        except Exception:
            return AgentResponse(
                answer="",
                intent=QueryIntent.MCP_WEATHER,
                kb_sources_used=[],
                mcp_tools_used=[],
                has_fallback=True,
                fallback_message=FALLBACK_MCP_TOOL_FAILURE.format(tool_type="weather", fallback_source="Open-Meteo")
            )

    def _handle_currency_query(self, query: str) -> AgentResponse:
        """Handle a query that requires currency conversion via MCP tool."""
        try:
            conversion = self.mcp_client.convert_currency(query)
            prompt = f"Answer using this live conversion data: {conversion}\nUser: {query}"
            answer = self._extract_text(self.llm.invoke(prompt).content)
            return AgentResponse(
                answer=answer,
                intent=QueryIntent.MCP_CURRENCY,
                kb_sources_used=[],
                mcp_tools_used=["convert_currency"],
                has_fallback=False,
                fallback_message=None
            )
        except Exception:
            return AgentResponse(
                answer="",
                intent=QueryIntent.MCP_CURRENCY,
                kb_sources_used=[],
                mcp_tools_used=[],
                has_fallback=True,
                fallback_message=FALLBACK_MCP_TOOL_FAILURE.format(tool_type="currency", fallback_source="Frankfurter")
            )

    def _handle_combined_query(self, query: str) -> AgentResponse:
        """Handle a query requiring both KB retrieval and MCP tool calls."""
        docs = self._retrieve_docs(query)
        context = "\n\n".join(self._extract_doc_content(doc) for doc in docs) if docs else "No specific destination facts found."
        sources = [self._extract_doc_source(doc) for doc in docs]

        mcp_data = []
        tools_used = []
        has_fallback = False

        try:
            weather = self.mcp_client.get_weather_forecast("Singapore")
            mcp_data.append(f"Weather: {weather}")
            tools_used.append("get_weather_forecast")
        except Exception:
            has_fallback = True

        mcp_context = "\n".join(mcp_data)
        prompt = COMBINED_RAG_MCP_PROMPT_TEMPLATE.format(
            kb_context=context,
            mcp_data=mcp_context,
            user_request=query
        )
        answer = self._extract_text(self.llm.invoke(prompt).content)

        return AgentResponse(
            answer=answer,
            intent=QueryIntent.COMBINED,
            kb_sources_used=sources,
            mcp_tools_used=tools_used,
            has_fallback=has_fallback,
            fallback_message=None
        )

