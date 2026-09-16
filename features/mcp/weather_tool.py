"""Weather MCP Tool using Open-Meteo API.

Retrieves current weather conditions and multi-day forecasts for Singapore.
Open-Meteo is completely free — no API key required.

Requirements satisfied: MCP Tool 1 (weather information), MCP Requirements 10–14.

API Documentation: https://open-meteo.com/en/docs
"""

from dataclasses import dataclass
from datetime import date


@dataclass
class WeatherCondition:
    """Current or forecasted weather conditions for a location."""

    date: date
    temperature_max_c: float
    temperature_min_c: float
    precipitation_mm: float
    wind_speed_kmh: float
    weather_description: str
    is_rainy: bool
    indoor_recommended: bool


@dataclass
class WeatherForecast:
    """Multi-day weather forecast for a destination."""

    city: str
    latitude: float
    longitude: float
    days: list[WeatherCondition]
    source: str = "Open-Meteo (https://open-meteo.com)"
    retrieved_at: str = ""


class WeatherTool:
    """MCP Tool: Retrieves weather data from Open-Meteo for Singapore.

    This tool is registered with the MCP client and made available to
    the LangChain agent. It is ONLY used for weather questions, never
    for destination knowledge already in the knowledge base.
    """

    TOOL_NAME = "get_weather_forecast"
    TOOL_DESCRIPTION = (
        "Get current weather conditions and multi-day forecast for Singapore. "
        "Use this tool ONLY for weather questions (current conditions, rain forecast, "
        "indoor vs outdoor recommendations). Do NOT use for destination knowledge."
    )
    SOURCE_LABEL = "Open-Meteo Weather API (https://open-meteo.com)"

    def __init__(
        self,
        base_url: str = "https://api.open-meteo.com/v1",
        latitude: float = 1.3521,
        longitude: float = 103.8198,
        city: str = "Singapore",
    ) -> None:
        """Initialise the weather tool with location settings.

        Args:
            base_url: Open-Meteo API base URL.
            latitude: Destination latitude (default: Singapore).
            longitude: Destination longitude (default: Singapore).
            city: Destination city name for display purposes.
        """
        self.base_url = base_url
        self.latitude = latitude
        self.longitude = longitude
        self.city = city

    def get_forecast(self, days: int = 7) -> WeatherForecast:
        """Fetch a multi-day weather forecast for the destination.

        Args:
            days: Number of forecast days (1–16).

        Returns:
            WeatherForecast with per-day conditions including rain flags.

        Raises:
            ConnectionError: If the Open-Meteo API is unreachable.
            ValueError: If days is outside the valid range.
        """
        raise NotImplementedError("Implement in TDD cycle")

    def get_current_conditions(self) -> WeatherCondition:
        """Fetch today's current weather conditions.

        Returns:
            WeatherCondition for today.

        Raises:
            ConnectionError: If the Open-Meteo API is unreachable.
        """
        raise NotImplementedError("Implement in TDD cycle")

    def as_langchain_tool(self) -> object:
        """Return this tool wrapped as a LangChain Tool object.

        Returns:
            LangChain Tool instance with name, description, and invocation function.
        """
        raise NotImplementedError("Implement in TDD cycle")

    @staticmethod
    def _wmo_code_to_description(wmo_code: int) -> tuple[str, bool]:
        """Convert a WMO weather code to a human-readable description.

        Args:
            wmo_code: WMO weather interpretation code from Open-Meteo.

        Returns:
            Tuple of (description string, is_rainy boolean).
        """
        raise NotImplementedError("Implement in TDD cycle")
