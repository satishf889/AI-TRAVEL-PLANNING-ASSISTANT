"""Main Streamlit application entry point."""
from __future__ import annotations

from typing import Any

import streamlit as st

from features.ui.components import (
    render_chat_message,
    render_example_queries,
    render_sidebar,
)
from features.ui.session_state import (
    add_to_chat_history,
    get_agent,
    get_chat_history,
    initialise_session_state,
)


def main() -> None:
    """Main Streamlit application function."""
    st.set_page_config(
        page_title="AI Travel Planning Assistant — Singapore",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialise_session_state()
    render_sidebar()

    st.title("✈️ AI Travel Planning Assistant")
    st.caption("Powered by Google Gemini Pro + LangChain + RAG + MCP Tools")

    history = get_chat_history()

    # Render chat history
    for msg in history:
        render_chat_message(msg["role"], msg["content"], msg.get("metadata"))

    if not history:
        query = render_example_queries()
    else:
        query = None

    user_input = st.chat_input("💬 Type your message here...")

    # Use example query if clicked, otherwise use chat input
    final_query = query if query else user_input

    if final_query:
        # 1. Add user message to UI state
        add_to_chat_history("user", final_query)
        render_chat_message("user", final_query)

        # 2. Get agent and process
        with st.spinner("Agent is thinking..."):
            try:
                agent = get_agent()
                response = agent.process_query(final_query)

                # Extract metadata
                metadata: dict[str, Any] = {}
                if response.kb_sources_used:
                    metadata["kb_sources_used"] = response.kb_sources_used
                if response.mcp_tools_used:
                    metadata["mcp_tools_used"] = response.mcp_tools_used

                # Add assistant response to UI state
                add_to_chat_history("assistant", response.answer, metadata)
                render_chat_message("assistant", response.answer, metadata)

                if response.has_fallback and response.fallback_message:
                    st.warning(response.fallback_message)

            except Exception as e:
                st.error(f"Error processing query: {e}")


if __name__ == "__main__":
    main()

