from __future__ import annotations

from typing import Any

import streamlit as st

from features.ui.session_state import clear_chat_history


def render_chat_message(role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
    """Render a single chat message with role-appropriate styling."""
    with st.chat_message(role):
        st.markdown(content)

        if metadata:
            if metadata.get("mcp_tools_used"):
                render_mcp_tool_badge(metadata["mcp_tools_used"])
            if metadata.get("kb_sources_used"):
                render_source_citations(metadata["kb_sources_used"])

def render_source_citations(sources: list[dict[str, Any]]) -> None:
    """Render a collapsible source citation section."""
    if not sources:
        return
    with st.expander("📚 View Sources"):
        for source in sources:
            st.markdown(f"- [{source['title']}]({source['url']})")

def render_mcp_tool_badge(tool_names: list[str]) -> None:
    """Render a badge indicating which MCP tools were used in a response."""
    if not tool_names:
        return
    for tool in tool_names:
        st.caption(f"🛠️ **Tools used:** {tool}")

def render_sidebar() -> None:
    """Render the application sidebar with settings and info."""
    if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
        clear_chat_history()

def render_example_queries() -> str:
    """Render example query buttons and return the selected query if clicked."""
    cols = st.columns(3)
    examples = [
        "Plan a 3-day itinerary",
        "What is the weather like?",
        "Convert $100 USD to SGD"
    ]

    for i, ex in enumerate(examples):
        if cols[i].button(ex):
            return ex
    return ""
