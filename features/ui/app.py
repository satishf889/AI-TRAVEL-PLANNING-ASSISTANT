"""Main Streamlit application entry point.

Run with: streamlit run features/ui/app.py

This is the main entry point for the AI Travel Planning Assistant UI.
It wires together the orchestrator, session state, and UI components.
"""

import streamlit as st

from features.ui.session_state import initialise_session_state


def main() -> None:
    """Main Streamlit application function."""
    st.set_page_config(
        page_title="AI Travel Planning Assistant — Singapore",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialise session state on first load
    initialise_session_state()

    # TODO: Render full UI — implemented in TDD cycle
    st.title("✈️ AI Travel Planning Assistant")
    st.caption("Powered by Google Gemini Pro + LangChain + RAG + MCP Tools")
    st.info("🚧 UI implementation in progress — following TDD cycle.")


if __name__ == "__main__":
    main()
