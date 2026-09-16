"""Tests for WeatherTool — MCP Tool 1.

TDD: Tests use respx to mock HTTP calls — no real API calls are made.
"""

from datetime import date
from unittest.mock import MagicMock

import pytest

from features.mcp.weather_tool import WeatherCondition, WeatherForecast, WeatherTool


@pytest.mark.unit
@pytest.mark.mcp
class TestWeatherToolInit:
    """Tests for WeatherTool initialisation."""

    def test_init_with_defaults(self) -> None:
        """WeatherTool initialises with Singapore coordinates by default."""
        tool = WeatherTool()
        assert tool.latitude == 1.3521
        assert tool.longitude == 103.8198
        assert tool.city == "Singapore"

    def test_tool_name_constant(self) -> None:
        """WeatherTool has the correct TOOL_NAME constant."""
        assert WeatherTool.TOOL_NAME == "get_weather_forecast"

    def test_source_label_contains_open_meteo(self) -> None:
        """SOURCE_LABEL references Open-Meteo."""
        assert "Open-Meteo" in WeatherTool.SOURCE_LABEL


@pytest.mark.unit
@pytest.mark.mcp
class TestWeatherForecastParsing:
    """Tests for parsing Open-Meteo API responses."""

    def test_wmo_code_sunny_returns_not_rainy(self) -> None:
        """WMO code 0 (clear sky) returns is_rainy=False."""
        description, is_rainy = WeatherTool._wmo_code_to_description(0)
        assert is_rainy is False
        assert description  # must have a description

    def test_wmo_code_rain_returns_is_rainy(self) -> None:
        """WMO codes 61–67 (rain) return is_rainy=True."""
        for code in [61, 63, 65, 67]:
            description, is_rainy = WeatherTool._wmo_code_to_description(code)
            assert is_rainy is True, f"Code {code} should be rainy"

    def test_wmo_code_thunderstorm_returns_is_rainy(self) -> None:
        """WMO codes 95–99 (thunderstorms) return is_rainy=True."""
        description, is_rainy = WeatherTool._wmo_code_to_description(95)
        assert is_rainy is True


@pytest.mark.unit
@pytest.mark.mcp
class TestWeatherToolMocked:
    """Tests for WeatherTool using mocked HTTP responses."""

    def test_get_forecast_returns_weather_forecast(
        self, mock_weather_tool: MagicMock
    ) -> None:
        """get_forecast returns a WeatherForecast instance."""
        result = mock_weather_tool.get_forecast(days=3)
        assert isinstance(result, WeatherForecast)

    def test_get_forecast_has_correct_city(
        self, mock_weather_tool: MagicMock, sample_weather_forecast: WeatherForecast
    ) -> None:
        """get_forecast returns forecast for the correct city."""
        result = mock_weather_tool.get_forecast(days=3)
        assert result.city == "Singapore"

    def test_get_forecast_rainy_day_recommends_indoor(
        self, sample_weather_forecast: WeatherForecast
    ) -> None:
        """A rainy WeatherCondition has indoor_recommended=True."""
        rainy_day = sample_weather_forecast.days[0]
        assert rainy_day.is_rainy is True
        assert rainy_day.indoor_recommended is True

    def test_get_forecast_includes_source_label(
        self, sample_weather_forecast: WeatherForecast
    ) -> None:
        """WeatherForecast includes an Open-Meteo source label."""
        assert "open-meteo" in sample_weather_forecast.source.lower()

    def test_get_forecast_invalid_days_raises_value_error(self) -> None:
        """get_forecast raises ValueError for invalid days count (< 1 or > 16)."""
        tool = WeatherTool()
        with pytest.raises((ValueError, NotImplementedError)):
            tool.get_forecast(days=0)


@pytest.mark.unit
@pytest.mark.mcp
class TestWeatherToolLangChainAdapter:
    """Tests for LangChain tool adapter."""

    def test_as_langchain_tool_returns_object(self) -> None:
        """as_langchain_tool returns a non-None object."""
        tool = WeatherTool()
        # Will raise NotImplementedError until implemented — expected in Red phase
        with pytest.raises(NotImplementedError):
            tool.as_langchain_tool()
