from __future__ import annotations

from typing import Any

import streamlit as st

from features.ui.session_state import clear_chat_history


def inject_custom_css() -> None:
    """Inject custom CSS for a modern light-themed TripMate UI."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Chat Bubbles */
    [data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* User Chat Bubble specifically */
    [data-testid="stChatMessage"][data-baseweb="box"]:has(div:contains("user")) {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
    }
    
    /* Inputs */
    .stChatInputContainer {
        border-radius: 12px !important;
        border: 1px solid #cbd5e1 !important;
        background: #ffffff !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px !important;
        transition: all 0.2s ease;
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        color: #334155;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15);
        border-color: #6366f1 !important;
        color: #6366f1 !important;
    }
    
    /* Primary Button override (e.g. + Plan New Trip) */
    .primary-btn .stButton>button {
        background-color: #6366f1;
        color: white;
        border: none;
    }
    .primary-btn .stButton>button:hover {
        background-color: #4f46e5;
        color: white !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def should_show_save_options(
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Determine if a message should show the Save Options expander."""
    if role != "assistant" or not content.strip():
        return False

    # 1. Hide for boundary refusal message
    if "I can only help for Singapore travel, no other thing." in content:
        return False

    # 2. Hide for fallback messages
    if "I searched my Singapore travel knowledge base" in content:
        return False
    if "service is currently unavailable" in content:
        return False

    # 3. Check metadata intent
    if metadata:
        intent = metadata.get("intent")
        if intent in ("mcp_weather", "mcp_currency", "conversational"):
            return False

        # If purely MCP tools used without KB sources and intent is not combined
        mcp_tools = metadata.get("mcp_tools_used", [])
        kb_sources = metadata.get("kb_sources_used", [])
        if mcp_tools and not kb_sources and intent != "combined":
            return False

    # 4. Fallback heuristics if metadata is omitted
    trimmed_lower = content.strip().lower()
    if (
        trimmed_lower.startswith("answer using this live")
        or "current weather in singapore" in trimmed_lower
        or ("sgd" in trimmed_lower and "exchange rate" in trimmed_lower)
    ):
        if "destination facts" not in trimmed_lower and "itinerary" not in trimmed_lower:
            return False

    return True


def render_chat_message(
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Render a single chat message with role-appropriate styling."""
    avatar = "🧑‍💻" if role == "user" else "✈️"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)

        if metadata and metadata.get("mcp_tools_used"):
            render_mcp_tool_badge(metadata["mcp_tools_used"])

        # Conditionally show save options for assistant messages
        if should_show_save_options(role, content, metadata):
            with st.expander("💾 Save Options"):
                import hashlib
                content_hash = hashlib.md5(content.encode()).hexdigest()[:8]

                col1, col2 = st.columns(2)

                with col1:
                    save_type = st.selectbox(
                        "What to save?",
                        ["Full Response", "KB Facts Only", "Agent Suggestions Only"],
                        key=f"save_type_{content_hash}",
                    )
                with col2:
                    st.write("")  # spacing
                    st.write("")  # spacing
                    if st.button("Save to My Trips", key=f"save_trips_{content_hash}"):
                        text_to_save = content
                        if save_type == "KB Facts Only":
                            kb_lines = [
                                line for line in content.split("\n")
                                if "Knowledge Base" in line or "Destination Facts" in line
                                or line.startswith("-") or line.startswith("*")
                            ]
                            text_to_save = (
                                "Extracting KB facts from response: \n\n" + "\n".join(kb_lines)
                            )
                        elif save_type == "Agent Suggestions Only":
                            agent_lines = [
                                line for line in content.split("\n")
                                if "AI Suggestion" in line or "Recommendations" in line
                                or line.startswith("-") or line.startswith("*")
                            ]
                            text_to_save = (
                                "Extracting Agent suggestions from response: \n\n"
                                + "\n".join(agent_lines)
                            )

                        st.session_state["my_trips"].append(text_to_save)
                        st.toast("Saved to My Trips! 🧳")

                    if st.button("Save to Saved Places", key=f"save_places_{content_hash}"):
                        st.session_state["saved_places"].append(content)
                        st.toast("Saved to Places! 🤍")

def render_mcp_tool_badge(tool_names: list[str]) -> None:
    """Render a badge indicating which MCP tools were used in a response."""
    if not tool_names:
        return
    for tool in tool_names:
        st.caption(f"🛠️ **Tools used:** {tool}")

def render_sidebar() -> None:
    """Render the application sidebar with navigation and settings."""
    st.sidebar.title("TripMate")
    st.sidebar.caption("Your AI Travel Planner")
    
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### Menu")
    
    # Use session state to handle navigation
    current = st.session_state.get("current_page", "Chat")
    
    if st.sidebar.button("💬 Chat", use_container_width=True, type="primary" if current == "Chat" else "secondary"):
        st.session_state["current_page"] = "Chat"
        st.rerun()
    if st.sidebar.button("🧳 My Trips", use_container_width=True, type="primary" if current == "My Trips" else "secondary"):
        st.session_state["current_page"] = "My Trips"
        st.rerun()
    if st.sidebar.button("🤍 Saved Places", use_container_width=True, type="primary" if current == "Saved Places" else "secondary"):
        st.session_state["current_page"] = "Saved Places"
        st.rerun()
    if st.sidebar.button("🧭 Explore", use_container_width=True, type="primary" if current == "Explore" else "secondary"):
        st.session_state["current_page"] = "Explore"
        st.rerun()
    if st.sidebar.button("❓ How to Use", use_container_width=True, type="primary" if current == "How to Use" else "secondary"):
        st.session_state["current_page"] = "How to Use"
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Settings")

    if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
        clear_chat_history()
        st.rerun()

    if st.sidebar.button("🛑 Cancel Generation", use_container_width=True):
        st.session_state["stop_generation"] = True
        st.rerun()

def render_how_to_use_page() -> None:
    """Render the How to Use guidance page explaining chatbot capabilities."""
    st.markdown("## ❓ How to Use TripMate")
    st.caption("Your AI-powered Singapore travel planning assistant")
    st.markdown("---")

    # --- What TripMate can do ---
    st.markdown("### 🎯 What TripMate Can Do For You")
    col1, col2 = st.columns(2)
    with col1:
        st.info(
            "🏙️ **Plan Itineraries**\n\n"
            "Get personalised day-by-day Singapore travel plans tailored to your travel style, "
            "duration, and group type."
        )
        st.info(
            "🌦️ **Live Weather**\n\n"
            "Check real-time Singapore weather forecasts sourced directly from Open-Meteo. "
            "Includes indoor/outdoor activity recommendations."
        )
    with col2:
        st.info(
            "🏰 **Attractions & Tips**\n\n"
            "Discover top attractions, neighbourhoods, local food, transport options, "
            "and cultural tips from our verified knowledge base."
        )
        st.info(
            "💱 **Live Currency Conversion**\n\n"
            "Convert any amount between currencies (INR, USD, EUR, GBP, JPY and more) using "
            "live rates from Frankfurter API."
        )

    st.markdown("---")

    # --- Step-by-step guide ---
    st.markdown("### 📍 Step-by-Step Guide")
    steps = [
        (
            "1️⃣",
            "Start a conversation",
            'Type any Singapore travel question in the chat box at the bottom, e.g. \'What are the top things to do in Singapore?\'',
        ),
        (
            "2️⃣",
            "Plan a trip instantly",
            "Click the **➕ Plan New Trip** button at the top right. Fill in your dates and travel group — TripMate generates a full itinerary.",
        ),
        (
            "3️⃣",
            "Ask for live data",
            "For weather or currency queries, TripMate automatically calls live MCP tools. Try: *\'What is the weather in Singapore this week?\' or \'Convert 500 INR to SGD\'*",
        ),
        (
            "4️⃣",
            "Save what you love",
            "After any assistant response, expand **💾 Save Options** to save itineraries to **My Trips** or places to **Saved Places**.",
        ),
        (
            "5️⃣",
            "Review saved content",
            "Use **🧳 My Trips** and **🤍 Saved Places** in the sidebar to revisit anything you’ve bookmarked.",
        ),
    ]
    for icon, title, desc in steps:
        with st.container():
            st.markdown(f"**{icon} {title}**")
            st.caption(desc)
            st.markdown("")

    st.markdown("---")

    # --- Example queries table ---
    st.markdown("### 💬 Example Questions to Try")
    examples = [
        ("Itinerary", "🗓️", "Plan a 3-day Singapore itinerary for a couple"),
        ("Itinerary", "🗓️", "Family-friendly 5-day Singapore trip with kids"),
        ("Attractions", "🏛️", "What are the must-visit attractions in Singapore?"),
        ("Attractions", "🏛️", "Best hawker centres for local food in Singapore"),
        ("Weather", "🌦️", "What is the weather forecast for Singapore this week?"),
        ("Weather", "🌦️", "Is it going to rain in Singapore tomorrow?"),
        ("Currency", "💱", "Convert 10000 INR to SGD"),
        ("Currency", "💱", "What is the exchange rate from USD to SGD?"),
        ("Transport", "🚇", "How do I get around Singapore using public transport?"),
        ("Culture", "🌏", "What are the cultural etiquette tips for Singapore?"),
    ]
    st.table(
        {
            "Category": [f"{icon} {cat}" for cat, icon, _ in examples],
            "Example Question": [q for _, _, q in examples],
        }
    )

    st.markdown("---")

    # --- Limitations ---
    st.markdown("### ⚠️ Limitations & Boundaries")
    st.warning(
        "🇬🇧 **Singapore Only** — TripMate is specialised for Singapore travel. "
        "Questions about other countries or cities will not be answered.\n\n"
        "🚫 **No Personal Advice** — TripMate does not provide medical, legal, financial, "
        "or personal contact information.\n\n"
        "📊 **Grounded Responses** — All destination facts come from a verified knowledge base. "
        "If information is unavailable, TripMate will tell you clearly rather than guess.\n\n"
        "🔄 **Live Data Availability** — Weather and currency data require an internet connection. "
        "If a live data service is temporarily unavailable, TripMate will let you know."
    )


def render_example_queries() -> str:
    """Render example query buttons and return the selected query if clicked."""
    st.markdown("### Welcome to TripMate! 👋")
    st.caption("We currently specialize in Singapore travel planning, with more destinations coming soon!")
    
    cols = st.columns(4)
    examples = [
        "🌴 Singapore 3-Day Trip",
        "🌦️ Singapore Weather",
        "💱 Convert Currency",
        "🏛️ Top Attractions"
    ]

    for i, ex in enumerate(examples):
        if cols[i].button(ex, use_container_width=True):
            return ex
    return ""
