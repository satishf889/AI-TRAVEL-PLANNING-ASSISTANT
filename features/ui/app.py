"""Main Streamlit application entry point."""
from __future__ import annotations

from typing import Any
from datetime import date, timedelta

import streamlit as st

from features.ui.components import (
    inject_custom_css,
    render_chat_message,
    render_example_queries,
    render_mcp_tool_badge,
    render_sidebar,
)
from features.ui.session_state import (
    add_to_chat_history,
    clear_chat_history,
    get_agent,
    get_chat_history,
    initialise_session_state,
)

@st.dialog("Plan New Trip")
def trip_planning_form():
    """Form to generate a structured itinerary request."""
    st.write("Fill in your travel details to get a tailored Singapore itinerary!")
    
    max_date = date.today() + timedelta(days=180)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", min_value=date.today(), max_value=max_date, value=date.today())
    with col2:
        end_date = st.date_input("End Date", min_value=start_date, max_value=max_date, value=start_date + timedelta(days=2))
        
    days = (end_date - start_date).days + 1
    
    if days > 15:
        st.error("Trip duration cannot exceed 15 days.")
        st.stop()
        
    specification = st.selectbox("Who are you travelling with?", ["Couples", "Family with Kids", "Corporate/Business", "Solo", "Friends"])
    
    if st.button("Generate Itinerary", type="primary"):
        st.session_state["show_trip_form"] = False
        st.session_state["pending_query"] = f"I am planning a {days}-day trip to Singapore for {specification}. Please provide a tailored itinerary."
        clear_chat_history()
        st.rerun()

def render_chat_page() -> None:
    history = get_chat_history()

    for msg in history:
        render_chat_message(msg["role"], msg["content"], msg.get("metadata"))

    if not history:
        query = render_example_queries()
    else:
        query = None

    if "pending_query" in st.session_state:
        query = st.session_state.pop("pending_query")

    user_input = st.chat_input("💬 Type your message here...")

    final_query = query if query else user_input

    if final_query:
        if st.session_state.get("stop_generation"):
            st.session_state["stop_generation"] = False

        add_to_chat_history("user", final_query)
        render_chat_message("user", final_query)

        with st.chat_message("assistant", avatar="✈️"):
            agent = get_agent()
            metadata: dict[str, Any] = {}

            try:
                with st.spinner("Agent is thinking..."):
                    stream = agent.stream_query(final_query, metadata)
                    try:
                        # Extract the first chunk to hold the spinner until the TTFT
                        first_chunk = next(stream)
                    except StopIteration:
                        first_chunk = ""

                response_container = st.empty()
                full_response = first_chunk
                
                if first_chunk:
                    response_container.markdown(full_response + "▌")

                for chunk in stream:
                    if st.session_state.get("stop_generation", False):
                        full_response += "\n\n*(Generation stopped by user)*"
                        st.session_state["stop_generation"] = False
                        break

                    full_response += chunk
                    response_container.markdown(full_response + "▌")

                response_container.markdown(full_response)

                if metadata.get("mcp_tools_used"):
                    render_mcp_tool_badge(metadata["mcp_tools_used"])

                if metadata.get("has_fallback") and metadata.get("fallback_message") and not full_response:
                    st.warning(metadata["fallback_message"])
                    full_response = metadata["fallback_message"]

                clean_metadata: dict[str, Any] = {}
                if metadata.get("mcp_tools_used"):
                    clean_metadata["mcp_tools_used"] = metadata["mcp_tools_used"]
                if metadata.get("intent"):
                    intent_val = metadata["intent"]
                    clean_metadata["intent"] = (
                        intent_val.value if hasattr(intent_val, "value") else str(intent_val)
                    )
                if metadata.get("kb_sources_used"):
                    clean_metadata["kb_sources_used"] = metadata["kb_sources_used"]

                add_to_chat_history("assistant", full_response, clean_metadata)
                st.rerun()

            except Exception as e:
                st.error(f"Error processing query: {e}")

def main() -> None:
    """Main Streamlit application function."""
    st.set_page_config(
        page_title="TripMate — Your AI Travel Planner",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialise_session_state()
    inject_custom_css()
    render_sidebar()

    # Top Navigation Bar layout
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("✈️ TripMate")
    with col2:
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("➕ Plan New Trip", use_container_width=True):
            trip_planning_form()
        st.markdown('</div>', unsafe_allow_html=True)

    st.caption("Powered by Azure OpenAI (gpt-4o-mini) + LangChain + RAG + MCP Tools")
    st.markdown("---")

    current_page = st.session_state.get("current_page", "Chat")

    if current_page == "Chat":
        render_chat_page()
    elif current_page == "My Trips":
        st.subheader("🧳 My Trips")
        if not st.session_state.get("my_trips"):
            st.info("No trips saved yet. Plan a new trip and save the itinerary!")
        else:
            for i, trip in enumerate(st.session_state["my_trips"]):
                with st.expander(f"Saved Trip #{i+1}"):
                    st.markdown(trip)
    elif current_page == "Saved Places":
        st.subheader("🤍 Saved Places")
        if not st.session_state.get("saved_places"):
            st.info("No places saved yet. Ask the assistant about places and save them!")
        else:
            for i, place in enumerate(st.session_state["saved_places"]):
                with st.expander(f"Saved Place #{i+1}"):
                    st.markdown(place)
    else:
        st.info(f"{current_page} - Coming Soon!")

if __name__ == "__main__":
    main()
