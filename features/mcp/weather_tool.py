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
        if not (1 <= days <= 16):
            raise ValueError(f"Days must be between 1 and 16, got {days}")

        import requests
        url = f"{self.base_url}/forecast"
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,weather_code",
            "forecast_days": days,
            "timezone": "Asia/Singapore",
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to fetch weather from Open-Meteo: {e}") from e

        daily = data.get("daily", {})
        times = daily.get("time", [])
        temp_max = daily.get("temperature_2m_max", [])
        temp_min = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])
        wind = daily.get("wind_speed_10m_max", [])
        codes = daily.get("weather_code", [])

        if not times:
            raise ValueError("Unexpected JSON shape from Open-Meteo")

        conditions = []
        for i in range(min(days, len(times))):
            desc, is_rainy = self._wmo_code_to_description(codes[i])
            conditions.append(
                WeatherCondition(
                    date=date.fromisoformat(times[i]),
                    temperature_max_c=temp_max[i],
                    temperature_min_c=temp_min[i],
                    precipitation_mm=precip[i],
                    wind_speed_kmh=wind[i],
                    weather_description=desc,
                    is_rainy=is_rainy,
                    indoor_recommended=is_rainy,
                )
            )

        from datetime import datetime
        return WeatherForecast(
            city=self.city,
            latitude=self.latitude,
            longitude=self.longitude,
            days=conditions,
            source=self.SOURCE_LABEL,
            retrieved_at=datetime.now().isoformat(),
        )

    def get_current_conditions(self) -> WeatherCondition:
        """Fetch today's current weather conditions.

        Returns:
            WeatherCondition for today.

        Raises:
            ConnectionError: If the Open-Meteo API is unreachable.
        """
        forecast = self.get_forecast(days=1)
        if not forecast.days:
            raise ValueError("No forecast data returned")
        return forecast.days[0]

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
        if wmo_code == 0:
            return "Clear sky", False
        elif wmo_code in [1, 2, 3]:
            return "Mainly clear, partly cloudy, and overcast", False
        elif wmo_code in [45, 48]:
            return "Fog and depositing rime fog", False
        elif wmo_code in [51, 53, 55, 56, 57]:
            return "Drizzle", True
        elif wmo_code in [61, 63, 65, 66, 67]:
            return "Rain", True
        elif wmo_code in [71, 73, 75, 77]:
            return "Snow fall", False
        elif wmo_code in [80, 81, 82]:
            return "Rain showers", True
        elif wmo_code in [85, 86]:
            return "Snow showers", False
        elif wmo_code in [95, 96, 99]:
            return "Thunderstorm", True
        else:
            return f"Unknown weather code {wmo_code}", False
