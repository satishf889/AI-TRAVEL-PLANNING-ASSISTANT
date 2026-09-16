"""Tests for WeatherTool — MCP Tool 1.

TDD: Tests use respx to mock HTTP calls — no real API calls are made.
"""

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

    def test_wmo_code_unknown_fallback_does_not_raise(self) -> None:
        """An unknown WMO code returns a non-empty string and False without raising."""
        description, is_rainy = WeatherTool._wmo_code_to_description(999)
        assert isinstance(description, str)
        assert len(description) > 0
        assert is_rainy is False


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


@pytest.mark.unit
@pytest.mark.mcp
class TestWeatherToolHTTP:
    """Tests for WeatherTool HTTP data-fetching with mocked requests.get.

    All tests patch requests.get so no real network calls are made.
    These tests verify the full parsing pipeline from raw API JSON
    to WeatherForecast / WeatherCondition objects.
    """

    # Representative Open-Meteo /forecast JSON response (3-day)
    _MOCK_API_RESPONSE: dict = {
        "latitude": 1.3521,
        "longitude": 103.8198,
        "daily": {
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "temperature_2m_max": [31.0, 30.0, 32.0],
            "temperature_2m_min": [25.0, 24.0, 26.0],
            "precipitation_sum": [0.0, 5.2, 0.0],
            "wind_speed_10m_max": [12.0, 15.0, 10.0],
            "weather_code": [0, 63, 1],
        },
    }

    def _mock_http_response(self) -> MagicMock:
        """Build a mock requests.Response that returns the fixture JSON."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = self._MOCK_API_RESPONSE
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    def test_get_forecast_calls_open_meteo_url(self) -> None:
        """get_forecast makes a GET request to the Open-Meteo API URL."""
        from unittest.mock import patch

        tool = WeatherTool()
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._mock_http_response()
            tool.get_forecast(days=3)
            mock_get.assert_called_once()
            call_url: str = mock_get.call_args[0][0]
            assert "api.open-meteo.com" in call_url

    def test_get_forecast_returns_weather_forecast_type(self) -> None:
        """get_forecast returns a WeatherForecast with the correct number of days."""
        from unittest.mock import patch

        tool = WeatherTool()
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._mock_http_response()
            result = tool.get_forecast(days=3)
            assert isinstance(result, WeatherForecast)
            assert len(result.days) == 3

    def test_get_forecast_sets_source_label(self) -> None:
        """get_forecast result includes Open-Meteo in the source label."""
        from unittest.mock import patch

        tool = WeatherTool()
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._mock_http_response()
            result = tool.get_forecast(days=3)
            assert "open-meteo" in result.source.lower()

    def test_get_forecast_rainy_day_indoor_recommended(self) -> None:
        """days[1] (WMO code 63, moderate rain) has indoor_recommended=True."""
        from unittest.mock import patch

        tool = WeatherTool()
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._mock_http_response()
            result = tool.get_forecast(days=3)
            assert result.days[1].is_rainy is True
            assert result.days[1].indoor_recommended is True

    def test_get_forecast_sunny_day_not_indoor(self) -> None:
        """days[0] (WMO code 0, clear sky) has indoor_recommended=False."""
        from unittest.mock import patch

        tool = WeatherTool()
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._mock_http_response()
            result = tool.get_forecast(days=3)
            assert result.days[0].is_rainy is False
            assert result.days[0].indoor_recommended is False

    def test_get_forecast_invalid_days_zero_raises_value_error(self) -> None:
        """get_forecast raises ValueError before making any HTTP call when days=0."""
        tool = WeatherTool()
        with pytest.raises(ValueError):
            tool.get_forecast(days=0)

    def test_get_forecast_invalid_days_17_raises_value_error(self) -> None:
        """get_forecast raises ValueError before making any HTTP call when days=17."""
        tool = WeatherTool()
        with pytest.raises(ValueError):
            tool.get_forecast(days=17)

    def test_get_forecast_connection_error_raises(self) -> None:
        """get_forecast re-raises as ConnectionError when the API is unreachable."""
        from unittest.mock import patch

        import requests as requests_lib

        tool = WeatherTool()
        with patch("requests.get", side_effect=requests_lib.ConnectionError("timeout")):
            with pytest.raises(ConnectionError):
                tool.get_forecast(days=3)

    def test_get_current_conditions_returns_weather_condition(self) -> None:
        """get_current_conditions returns a WeatherCondition for today."""
        from unittest.mock import patch

        tool = WeatherTool()
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._mock_http_response()
            result = tool.get_current_conditions()
            assert isinstance(result, WeatherCondition)
