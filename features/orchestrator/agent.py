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

from features.cache.response_cache import ResponseCache
from features.orchestrator.pii_sanitizer import PIISanitizer
from features.orchestrator.prompt_templates import (
    COMBINED_RAG_MCP_PROMPT_TEMPLATE,
    CONVERSATIONAL_PROMPT_TEMPLATE,
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
    CONVERSATIONAL = "conversational"  # Greeting, small talk, or general introduction
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

    GREETING_KEYWORDS = {
        "hi", "hello", "hey", "hola", "greetings", "good morning",
        "good afternoon", "good evening", "how are you", "who are you",
        "what can you do", "help", "thanks", "thank you", "bye", "goodbye"
    }

    def __init__(
        self,
        retriever: Any,
        mcp_client: Any,
        context_manager: Any,
        llm: Any,
        cache: Any = None,
    ) -> None:
        self.retriever = retriever
        self.mcp_client = mcp_client
        self.context_manager = context_manager
        self.llm = llm
        self.cache = cache or ResponseCache(redis_enabled=False)


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
        cleaned_query = query.strip().lower()
        # Fast check for simple conversational greetings
        if cleaned_query in self.GREETING_KEYWORDS or any(
            cleaned_query.startswith(g + " ") or cleaned_query.endswith(" " + g)
            for g in ["hi", "hello", "hey"]
        ):
            return QueryIntent.CONVERSATIONAL

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

        if not hasattr(self.llm, "bind_tools"):
            return QueryIntent.KB_ONLY

        try:
            llm_with_tools = self.llm.bind_tools(tools)
            response = llm_with_tools.invoke(query)

            tool_calls = getattr(response, "tool_calls", [])
            tool_names = [
                call.get("name", call) if isinstance(call, dict) else getattr(call, "name", str(call))
                for call in tool_calls
            ]

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
        except Exception:
            return QueryIntent.KB_ONLY

    def _get_history_text(self) -> str:
        """Format recent conversation history as a string."""
        if hasattr(self.context_manager, "get_history_for_langchain"):
            history = self.context_manager.get_history_for_langchain()
            if not history:
                return "No previous conversation history."
            lines = [f"{role.capitalize()}: {content}" for role, content in history[-6:]]
            return "\n".join(lines)
        return "No previous conversation history."

    def process_query(self, query: str) -> AgentResponse:
        """Process a user query synchronously (consumes stream)."""
        metadata: dict[str, Any] = {}
        stream = self.stream_query(query, metadata)
        answer = "".join(list(stream))

        # Add to history
        self.context_manager.add_assistant_message(answer)

        return AgentResponse(
            answer=answer,
            intent=metadata.get("intent", QueryIntent.UNKNOWN),
            kb_sources_used=metadata.get("kb_sources_used", []),
            mcp_tools_used=metadata.get("mcp_tools_used", []),
            has_fallback=metadata.get("has_fallback", False),
            fallback_message=metadata.get("fallback_message")
        )

    def stream_query(self, query: str, metadata_out: dict[str, Any]) -> Any:
        """Stream a user query and populate metadata_out with sources and intent."""
        # 1. Sanitize query to scrub any PII before processing or storing
        clean_query = PIISanitizer.sanitize(query)

        self.context_manager.update_preferences(clean_query)
        self.context_manager.add_user_message(clean_query)

        # 2. Check response cache for existing answer
        if hasattr(self, "cache") and self.cache is not None:
            cached_item = self.cache.get(clean_query)
            if cached_item and isinstance(cached_item, dict):
                metadata_out["intent"] = QueryIntent(cached_item.get("intent", QueryIntent.KB_ONLY.value))
                metadata_out["kb_sources_used"] = cached_item.get("kb_sources_used", [])
                metadata_out["mcp_tools_used"] = cached_item.get("mcp_tools_used", [])
                metadata_out["has_fallback"] = cached_item.get("has_fallback", False)
                metadata_out["fallback_message"] = cached_item.get("fallback_message")
                yield cached_item.get("answer", "")
                return

        intent = self.classify_intent(clean_query)
        metadata_out["intent"] = intent
        metadata_out["kb_sources_used"] = []
        metadata_out["mcp_tools_used"] = []
        metadata_out["has_fallback"] = False
        metadata_out["fallback_message"] = None

        prompt = ""

        if intent == QueryIntent.CONVERSATIONAL:
            prompt = self._prepare_conversational_query(clean_query)
        elif intent == QueryIntent.KB_ONLY:
            prompt = self._prepare_kb_query(clean_query, metadata_out)
        elif intent == QueryIntent.MCP_WEATHER:
            prompt = self._prepare_weather_query(clean_query, metadata_out)
        elif intent == QueryIntent.MCP_CURRENCY:
            prompt = self._prepare_currency_query(clean_query, metadata_out)
        else:
            prompt = self._prepare_combined_query(clean_query, metadata_out)

        if metadata_out.get("has_fallback") and metadata_out.get("fallback_message"):
            yield metadata_out["fallback_message"]
            return

        if not prompt:
            return

        # Check if LLM supports streaming
        handled = False
        accumulated_answer: list[str] = []

        if hasattr(self.llm, "stream"):
            try:
                for chunk in self.llm.stream(prompt):
                    handled = True
                    text = self._extract_text(chunk.content) if hasattr(chunk, "content") else str(chunk)
                    if text:
                        accumulated_answer.append(text)
                        yield text
            except Exception:
                handled = False

        if not handled and hasattr(self.llm, "invoke"):
            response = self.llm.invoke(prompt)
            text = self._extract_text(response.content) if hasattr(response, "content") else str(response)
            if text:
                accumulated_answer.append(text)
                yield text

        # Cache completed response
        full_text = "".join(accumulated_answer)
        if full_text and hasattr(self, "cache") and self.cache is not None:
            self.cache.set(
                clean_query,
                {
                    "answer": full_text,
                    "intent": intent.value,
                    "kb_sources_used": metadata_out.get("kb_sources_used", []),
                    "mcp_tools_used": metadata_out.get("mcp_tools_used", []),
                    "has_fallback": metadata_out.get("has_fallback", False),
                    "fallback_message": metadata_out.get("fallback_message"),
                },
            )


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

    def _prepare_conversational_query(self, query: str) -> str:
        history = self._get_history_text()
        return CONVERSATIONAL_PROMPT_TEMPLATE.format(history=history, message=query)

    def _prepare_kb_query(self, query: str, metadata_out: dict[str, Any]) -> str:
        docs = self._retrieve_docs(query)
        if not docs:
            metadata_out["has_fallback"] = True
            metadata_out["fallback_message"] = FALLBACK_NO_KB_CONTENT
            return ""

        context = "\n\n".join(self._extract_doc_content(doc) for doc in docs)
        history = self._get_history_text()
        metadata_out["kb_sources_used"] = [self._extract_doc_source(doc) for doc in docs]
        return RAG_QA_PROMPT_TEMPLATE.format(context=context, history=history, question=query)

    def _prepare_weather_query(self, query: str, metadata_out: dict[str, Any]) -> str:
        try:
            weather = self.mcp_client.get_weather_forecast("Singapore")
            metadata_out["mcp_tools_used"] = ["get_weather_forecast"]
            return f"Answer using this live weather data: {weather}\nUser: {query}"
        except Exception:
            metadata_out["has_fallback"] = True
            metadata_out["fallback_message"] = FALLBACK_MCP_TOOL_FAILURE.format(
                tool_type="weather", fallback_source="Open-Meteo"
            )
            return ""

    def _prepare_currency_query(self, query: str, metadata_out: dict[str, Any]) -> str:
        try:
            conversion = self.mcp_client.convert_currency(query)
            metadata_out["mcp_tools_used"] = ["convert_currency"]
            return f"Answer using this live conversion data: {conversion}\nUser: {query}"
        except Exception:
            metadata_out["has_fallback"] = True
            metadata_out["fallback_message"] = FALLBACK_MCP_TOOL_FAILURE.format(
                tool_type="currency", fallback_source="Frankfurter"
            )
            return ""

    def _prepare_combined_query(self, query: str, metadata_out: dict[str, Any]) -> str:
        docs = self._retrieve_docs(query)
        context = "\n\n".join(self._extract_doc_content(doc) for doc in docs) if docs else "No specific destination facts found."
        metadata_out["kb_sources_used"] = [self._extract_doc_source(doc) for doc in docs]

        mcp_data = []
        try:
            weather = self.mcp_client.get_weather_forecast("Singapore")
            mcp_data.append(f"Weather: {weather}")
            metadata_out["mcp_tools_used"].append("get_weather_forecast")
        except Exception:
            metadata_out["has_fallback"] = True

        mcp_context = "\n".join(mcp_data)
        history = self._get_history_text()
        return COMBINED_RAG_MCP_PROMPT_TEMPLATE.format(
            kb_context=context,
            mcp_data=mcp_context,
            history=history,
            user_request=query
        )


