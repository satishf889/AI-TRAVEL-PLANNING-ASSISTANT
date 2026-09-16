from __future__ import annotations

from typing import Any

import streamlit as st

from features.orchestrator.agent import TravelAgent


def get_agent_dependencies() -> tuple[Any, Any, Any, Any]:
    """Factory function for initializing agent dependencies with interactive preview stubs."""
    from features.orchestrator.context_manager import ConversationContextManager

    class DummyDocument:
        def __init__(self, page_content: str, metadata: dict[str, Any]) -> None:
            self.page_content = page_content
            self.metadata = metadata

    class DummyRetriever:
        def invoke(self, query: str) -> list[DummyDocument]:
            return [
                DummyDocument(
                    page_content=(
                        "Singapore offers diverse attractions: Marina Bay Sands, "
                        "Gardens by the Bay with Supertrees and Flower Dome, Sentosa Island, "
                        "and culinary hubs across Chinatown and Little India."
                    ),
                    metadata={
                        "title": "Visit Singapore Travel Guide",
                        "source": "https://www.visitsingapore.com",
                    },
                ),
                DummyDocument(
                    page_content=(
                        "The MRT (Mass Rapid Transit) is the most efficient and cost-effective "
                        "way to get around Singapore. EZ-Link cards are accepted."
                    ),
                    metadata={
                        "title": "Wikivoyage Singapore",
                        "source": "https://en.wikivoyage.org/wiki/Singapore",
                    },
                ),
            ]

    class DummyMCPClient:
        def get_weather_forecast(self, location: str) -> str:
            return (
                "Singapore: 31°C / 26°C, Partly Cloudy, 20% chance of showers. "
                "Warm & tropical."
            )

        def convert_currency(self, amount: str) -> str:
            return "50,000 INR = ~805.50 SGD (Exchange rate: 0.0161)"

    class BoundResponse:
        def __init__(self, calls: list[dict[str, Any]]) -> None:
            self.tool_calls = calls
            self.content = ""

    class DummyBoundLLM:
        def invoke(self, query: str) -> BoundResponse:
            q = query.lower()
            tool_calls: list[dict[str, Any]] = []
            if any(w in q for w in ["weather", "rain", "temperature", "forecast", "climate"]):
                tool_calls.append({"name": "get_weather_forecast"})
            if any(w in q for w in ["convert", "currency", "sgd", "inr", "usd", "rate"]):
                tool_calls.append({"name": "convert_currency"})
            if any(w in q for w in ["itinerary", "plan", "attraction", "visit", "hotel", "food"]):
                tool_calls.append({"name": "kb_search"})
                if "weather" in q or "plan" in q or "itinerary" in q:
                    tool_calls.append({"name": "get_weather_forecast"})

            # Default to kb_search if no specific tool matched
            if not tool_calls:
                tool_calls.append({"name": "kb_search"})

            return BoundResponse(tool_calls)

    class LLMResponse:
        def __init__(self, text: str) -> None:
            self.content = text
            self.tool_calls: list[Any] = []

    class DummyLLM:
        def bind_tools(self, tools: list[Any]) -> DummyBoundLLM:
            return DummyBoundLLM()

        def invoke(self, prompt: str) -> LLMResponse:
            p = prompt.lower()
            if "live weather data" in p or (
                "weather" in p and "singapore" in p and "itinerary" not in p
            ):
                content = (
                    "### ☀️ Singapore Weather Forecast\n\n"
                    "- **Current Conditions:** 31°C, Partly Cloudy, Light tropical breeze\n"
                    "- **Precipitation:** 20% chance of brief afternoon showers\n"
                    "- **Recommendation:** Outdoor attractions (like Gardens by the Bay or "
                    "Merlion Park) are ideal in the morning. Carry a light umbrella or visit "
                    "indoor domes during the afternoon heat!"
                )
            elif "live conversion data" in p or "currency" in p or "convert" in p:
                content = (
                    "### 💱 Currency Conversion\n\n"
                    "Based on real-time foreign exchange data:\n"
                    "- **50,000 INR** ≈ **805.50 SGD** (Rate: 1 SGD = ~62.07 INR)\n\n"
                    "Most merchants in Singapore accept contactless cards and Apple/Google Pay."
                )
            elif "itinerary" in p or "plan" in p:
                content = (
                    "### 🗺️ 3-Day Singapore Travel Itinerary\n\n"
                    "**Day 1: Marina Bay & Iconic Landmarks**\n"
                    "- **Morning:** Walk around Marina Bay Sands, Merlion Park, and Helix Bridge.\n"
                    "- **Afternoon:** Visit Gardens by the Bay (Flower Dome & Cloud Forest).\n"
                    "- **Evening:** Enjoy Spectra Light Show and dinner at Lau Pa Sat.\n\n"
                    "**Day 2: Cultural Heritage & Street Food**\n"
                    "- **Morning:** Explore Chinatown (Buddha Tooth Relic Temple).\n"
                    "- **Afternoon:** Stroll through Little India and Kampong Glam / Haji Lane.\n"
                    "- **Evening:** Night Safari or Clarke Quay riverfront dining.\n\n"
                    "**Day 3: Sentosa Island & Relaxation**\n"
                    "- **Morning:** Cable car to Sentosa, visit S.E.A. Aquarium.\n"
                    "- **Afternoon:** Relax at Palawan Beach or Siloso Beach.\n"
                    "- **Evening:** Sunset dining along the beach."
                )
            else:
                content = (
                    "Singapore is a vibrant global hub known for its lush green spaces, "
                    "world-class transit, and diverse culinary culture. The MRT system makes "
                    "getting between attractions quick and convenient."
                )

            return LLMResponse(content)

    retriever = DummyRetriever()
    mcp_client = DummyMCPClient()
    context_manager = ConversationContextManager()
    llm = DummyLLM()

    return retriever, mcp_client, context_manager, llm

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
