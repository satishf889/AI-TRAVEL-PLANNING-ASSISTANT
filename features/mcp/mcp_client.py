from __future__ import annotations

"""MCP client setup and tool registry.

Initialises the MCP client and registers all available MCP tools
so they can be discovered and used by the LangChain agent.

Requirements satisfied: MCP Requirements 8, 9 (connect to and expose MCP tools).
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class MCPToolResult:
    """Represents the result from an MCP tool call."""

    tool_name: str
    success: bool
    data: dict[str, Any]
    error_message: str | None = None
    source_label: str = ""  # e.g., "Open-Meteo Weather API", "Frankfurter Currency API"


class MCPClient:
    """Manages MCP tool connections and provides a unified tool interface.

    Registers the weather and currency MCP tools and exposes them to
    the LangChain agent via the LangChain MCP adapter interface.

    Requirements: MCP tools must be clearly labeled in responses (Req 13).
    Failed tool calls must be handled gracefully — no fabrication (Req 14).
    """

    def __init__(self, weather_tool: object, currency_tool: object) -> None:
        """Initialise the MCP client with registered tools.

        Args:
            weather_tool: An instance of WeatherTool.
            currency_tool: An instance of CurrencyTool.
        """
        self.weather_tool = weather_tool
        self.currency_tool = currency_tool

    def get_langchain_tools(self) -> list[object]:
        """Return all MCP tools as LangChain Tool objects for agent binding.

        Returns:
            List of LangChain-compatible tool objects.
        """
        tools = []
        if self.weather_tool:
            tools.append(self.weather_tool)
        if self.currency_tool:
            tools.append(self.currency_tool)
        return tools

    def select_tool(self, user_query: str) -> str | None:
        """Suggest the appropriate MCP tool based on the user query.

        This provides a hint to the agent about which tool to use,
        but final selection is performed by the LangChain agent.

        Args:
            user_query: The user's natural language question.

        Returns:
            Tool name ("weather" or "currency") or None if no MCP tool is needed.
        """
        query_lower = user_query.lower()

        WEATHER_KEYWORDS = {"weather", "rain", "forecast", "temperature", "humid", "sunny", "wind"}
        CURRENCY_KEYWORDS = {"currency", "convert", "exchange", "rate", "sgd", "inr", "usd", "eur", "gbp", "jpy", "money"}

        if any(keyword in query_lower for keyword in WEATHER_KEYWORDS):
            return "weather"
        if any(keyword in query_lower for keyword in CURRENCY_KEYWORDS):
            return "currency"

        return None

    def is_tool_available(self, tool_name: str) -> bool:
        """Check whether a named MCP tool is available and responding.

        Args:
            tool_name: Name of the tool to check ("weather" or "currency").

        Returns:
            True if the tool is available, False if it is unavailable.
        """
        if tool_name == "weather":
            return self.weather_tool is not None
        elif tool_name == "currency":
            return self.currency_tool is not None
        return False
