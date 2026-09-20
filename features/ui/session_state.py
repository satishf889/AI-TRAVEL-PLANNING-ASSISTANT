from __future__ import annotations

from typing import Any

import streamlit as st
from langchain_openai import AzureChatOpenAI

from features.config.settings import settings
from features.orchestrator.agent import TravelAgent
from features.rag.vector_store import VectorStoreManager


def get_agent_dependencies() -> tuple[Any, Any, Any, Any]:
    """Factory function for initializing agent dependencies with interactive preview stubs."""
    from features.mcp.currency_tool import CurrencyTool
    from features.mcp.mcp_client import MCPClient
    from features.mcp.weather_tool import WeatherTool
    from features.orchestrator.context_manager import ConversationContextManager
    from features.rag.retriever import KnowledgeRetriever

    # 1. RAG Retriever
    vector_store_manager = VectorStoreManager(
        persist_directory=settings.get_chroma_persist_path(),
        collection_name=settings.chroma_collection_name,
    )
    # Attempt to load if possible. If not initialized, retriever raises errors later
    try:
        vector_store_manager.load()
    except Exception:
        pass
    retriever = KnowledgeRetriever(
        vector_store_manager=vector_store_manager,
        top_k=settings.retrieval_top_k,
    )

    # 2. MCP Client
    weather_tool = WeatherTool(
        base_url=settings.weather_api_base_url,
        latitude=settings.destination_latitude,
        longitude=settings.destination_longitude,
        city=settings.destination_city,
    )
    currency_tool = CurrencyTool(base_url=settings.currency_api_base_url)
    mcp_client = MCPClient(weather_tool=weather_tool, currency_tool=currency_tool)

    # 3. Context Manager
    context_manager = ConversationContextManager()

    # 4. LLM
    from pydantic import SecretStr

    llm = AzureChatOpenAI(
        azure_deployment=settings.azure_openai_deployment_name,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        api_key=SecretStr(settings.azure_openai_api_key),
        temperature=settings.azure_openai_temperature,
        max_tokens=settings.azure_openai_max_tokens,
        streaming=True,
    )

    # 5. Cache
    from features.cache.response_cache import ResponseCache
    cache = ResponseCache(
        redis_enabled=settings.redis_enabled,
        redis_host=settings.redis_host,
        redis_port=settings.redis_port,
        ttl_seconds=settings.cache_ttl_seconds,
    )

    return retriever, mcp_client, context_manager, llm, cache


def initialise_session_state() -> None:
    """Initialise all required Streamlit session state variables.

    Must be called at the top of the Streamlit app before accessing any state.
    Idempotent — safe to call on every rerun.
    """
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "agent" not in st.session_state:
        deps = get_agent_dependencies()
        st.session_state["agent"] = TravelAgent(*deps)

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Chat"

    if "my_trips" not in st.session_state:
        st.session_state["my_trips"] = []

    if "saved_places" not in st.session_state:
        st.session_state["saved_places"] = []

    if "show_trip_form" not in st.session_state:
        st.session_state["show_trip_form"] = False

def get_agent() -> TravelAgent:
    """Get the TravelAgent singleton from session state."""
    if "agent" not in st.session_state:
        raise RuntimeError("Agent not initialized in session state")
    return st.session_state["agent"]  # type: ignore[no-any-return]

def get_chat_history() -> list[dict[str, Any]]:
    """Get the UI-facing chat history from session state."""
    return st.session_state.get("messages", [])  # type: ignore[no-any-return]

def add_to_chat_history(role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
    """Append a message to the UI chat history."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    st.session_state["messages"].append({
        "role": role,
        "content": content,
        "metadata": metadata
    })

def clear_chat_history() -> None:
    """Clear all chat history and reset the conversation context."""
    if "messages" in st.session_state:
        st.session_state["messages"] = []

    if "agent" in st.session_state and hasattr(st.session_state["agent"], "context_manager"):
        ctx = st.session_state["agent"].context_manager
        if ctx is not None:
            if hasattr(ctx, "clear"):
                ctx.clear()
            elif hasattr(ctx, "reset_context"):
                ctx.reset_context()
