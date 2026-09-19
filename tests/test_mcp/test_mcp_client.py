"""Tests for MCPClient — MCP tool registry and routing.

TDD: All tool dependencies are mocked. No real HTTP calls are made.
Requirements satisfied: MCP Requirements 8, 9, 10, 11, 14.
"""

from unittest.mock import MagicMock

import pytest

from features.mcp.mcp_client import MCPClient


@pytest.mark.unit
@pytest.mark.mcp
class TestMCPClientInit:
    """Tests for MCPClient initialisation and tool registration."""

    def test_init_stores_weather_tool(
        self, mock_weather_tool: MagicMock, mock_currency_tool: MagicMock
    ) -> None:
        """MCPClient stores the weather tool after init."""
        client = MCPClient(
            weather_tool=mock_weather_tool, currency_tool=mock_currency_tool
        )
        assert client.weather_tool is mock_weather_tool

    def test_init_stores_currency_tool(
        self, mock_weather_tool: MagicMock, mock_currency_tool: MagicMock
    ) -> None:
        """MCPClient stores the currency tool after init."""
        client = MCPClient(
            weather_tool=mock_weather_tool, currency_tool=mock_currency_tool
        )
        assert client.currency_tool is mock_currency_tool

    def test_get_langchain_tools_returns_list(
        self, mock_weather_tool: MagicMock, mock_currency_tool: MagicMock
    ) -> None:
        """get_langchain_tools returns a list containing both registered tools."""
        client = MCPClient(
            weather_tool=mock_weather_tool, currency_tool=mock_currency_tool
        )
        tools = client.get_langchain_tools()
        assert isinstance(tools, list)
        assert len(tools) == 2

    def test_get_langchain_tools_contains_weather_and_currency(
        self, mock_weather_tool: MagicMock, mock_currency_tool: MagicMock
    ) -> None:
        """get_langchain_tools list contains both the weather and currency tools."""
        client = MCPClient(
            weather_tool=mock_weather_tool, currency_tool=mock_currency_tool
        )
        tools = client.get_langchain_tools()
        assert mock_weather_tool in tools
        assert mock_currency_tool in tools


@pytest.mark.unit
@pytest.mark.mcp
class TestSelectTool:
    """Tests for MCPClient.select_tool — keyword-based routing.

    Requirements: Req 10 (select appropriate tool based on user request).
    """

    @pytest.fixture
    def client(
        self, mock_weather_tool: MagicMock, mock_currency_tool: MagicMock
    ) -> MCPClient:
        """Provide a real MCPClient with mock tools for routing tests."""
        return MCPClient(
            weather_tool=mock_weather_tool, currency_tool=mock_currency_tool
        )

    def test_weather_keyword_routes_to_weather(self, client: MCPClient) -> None:
        """Query containing 'weather' returns 'weather'."""
        assert client.select_tool("What is the weather today?") == "weather"

    def test_forecast_keyword_routes_to_weather(self, client: MCPClient) -> None:
        """Query containing 'forecast' returns 'weather'."""
        assert client.select_tool("Give me a 3-day forecast") == "weather"

    def test_rain_keyword_routes_to_weather(self, client: MCPClient) -> None:
        """Query containing 'rain' returns 'weather'."""
        assert client.select_tool("Will it rain tomorrow?") == "weather"

    def test_temperature_keyword_routes_to_weather(self, client: MCPClient) -> None:
        """Query containing 'temperature' returns 'weather'."""
        assert client.select_tool("What is the temperature in Singapore?") == "weather"

    def test_currency_keyword_routes_to_currency(self, client: MCPClient) -> None:
        """Query containing 'currency' returns 'currency'."""
        assert client.select_tool("What is the local currency?") == "currency"

    def test_convert_keyword_routes_to_currency(self, client: MCPClient) -> None:
        """Query containing 'convert' returns 'currency'."""
        assert client.select_tool("Convert 500 INR to SGD") == "currency"

    def test_exchange_keyword_routes_to_currency(self, client: MCPClient) -> None:
        """Query containing 'exchange' returns 'currency'."""
        assert client.select_tool("What is the exchange rate for SGD?") == "currency"

    def test_sgd_keyword_routes_to_currency(self, client: MCPClient) -> None:
        """Query containing currency code 'SGD' returns 'currency'."""
        assert client.select_tool("How much is 200 SGD in INR?") == "currency"

    def test_inr_keyword_routes_to_currency(self, client: MCPClient) -> None:
        """Query containing currency code 'INR' returns 'currency'."""
        assert client.select_tool("I have 10000 INR, how much is that in USD?") == "currency"

    def test_unrelated_query_returns_none(self, client: MCPClient) -> None:
        """Query with no weather or currency keywords returns None."""
        assert client.select_tool("Tell me about Gardens by the Bay") is None

    def test_destination_query_returns_none(self, client: MCPClient) -> None:
        """Destination-knowledge questions should not route to any MCP tool."""
        assert client.select_tool("What are the best hawker centres in Singapore?") is None

    def test_select_tool_is_case_insensitive(self, client: MCPClient) -> None:
        """Keyword matching is case-insensitive."""
        assert client.select_tool("WEATHER forecast for tomorrow") == "weather"
        assert client.select_tool("CONVERT my money") == "currency"

    def test_empty_query_returns_none(self, client: MCPClient) -> None:
        """An empty query string returns None without raising."""
        assert client.select_tool("") is None


@pytest.mark.unit
@pytest.mark.mcp
class TestIsToolAvailable:
    """Tests for MCPClient.is_tool_available.

    Requirements: Req 14 (handle unavailable tools without fabricating answers).
    """

    def test_weather_tool_is_available(
        self, mock_weather_tool: MagicMock, mock_currency_tool: MagicMock
    ) -> None:
        """is_tool_available returns True for 'weather' when tool is registered."""
        client = MCPClient(
            weather_tool=mock_weather_tool, currency_tool=mock_currency_tool
        )
        assert client.is_tool_available("weather") is True

    def test_currency_tool_is_available(
        self, mock_weather_tool: MagicMock, mock_currency_tool: MagicMock
    ) -> None:
        """is_tool_available returns True for 'currency' when tool is registered."""
        client = MCPClient(
            weather_tool=mock_weather_tool, currency_tool=mock_currency_tool
        )
        assert client.is_tool_available("currency") is True

    def test_unknown_tool_returns_false(
        self, mock_weather_tool: MagicMock, mock_currency_tool: MagicMock
    ) -> None:
        """is_tool_available returns False for an unknown tool name."""
        client = MCPClient(
            weather_tool=mock_weather_tool, currency_tool=mock_currency_tool
        )
        assert client.is_tool_available("flights") is False

    def test_none_weather_tool_returns_false(
        self, mock_currency_tool: MagicMock
    ) -> None:
        """is_tool_available returns False for 'weather' when tool is None."""
        client = MCPClient(weather_tool=None, currency_tool=mock_currency_tool)  # type: ignore[arg-type]
        assert client.is_tool_available("weather") is False

    def test_none_currency_tool_returns_false(
        self, mock_weather_tool: MagicMock
    ) -> None:
        """is_tool_available returns False for 'currency' when tool is None."""
        client = MCPClient(weather_tool=mock_weather_tool, currency_tool=None)  # type: ignore[arg-type]
        assert client.is_tool_available("currency") is False
