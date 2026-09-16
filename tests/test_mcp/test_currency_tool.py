"""Tests for CurrencyTool — MCP Tool 2.

TDD: Tests use mocked HTTP responses — no real API calls are made.
"""

from datetime import date
from unittest.mock import MagicMock

import pytest

from features.mcp.currency_tool import ConversionResult, CurrencyTool


@pytest.mark.unit
@pytest.mark.mcp
class TestCurrencyToolInit:
    """Tests for CurrencyTool initialisation."""

    def test_init_with_default_url(self) -> None:
        """CurrencyTool initialises with Frankfurter API URL."""
        tool = CurrencyTool()
        assert "frankfurter" in tool.base_url

    def test_tool_name_constant(self) -> None:
        """CurrencyTool has the correct TOOL_NAME constant."""
        assert CurrencyTool.TOOL_NAME == "convert_currency"

    def test_source_label_contains_frankfurter(self) -> None:
        """SOURCE_LABEL references Frankfurter."""
        assert "Frankfurter" in CurrencyTool.SOURCE_LABEL


@pytest.mark.unit
@pytest.mark.mcp
class TestCurrencyConversionMocked:
    """Tests for currency conversion using mocked responses."""

    def test_convert_returns_conversion_result(
        self, mock_currency_tool: MagicMock
    ) -> None:
        """Convert returns a ConversionResult instance."""
        result = mock_currency_tool.convert(50000.0, "INR", "SGD")
        assert isinstance(result, ConversionResult)

    def test_convert_inr_to_sgd(
        self, mock_currency_tool: MagicMock, sample_conversion_result: ConversionResult
    ) -> None:
        """Converting INR to SGD returns correct amounts."""
        result = mock_currency_tool.convert(50000.0, "INR", "SGD")
        assert result.from_currency == "INR"
        assert result.to_currency == "SGD"
        assert result.converted_amount > 0

    def test_conversion_includes_exchange_rate(
        self, sample_conversion_result: ConversionResult
    ) -> None:
        """ConversionResult includes the exchange rate."""
        assert sample_conversion_result.exchange_rate > 0

    def test_conversion_includes_source_label(
        self, sample_conversion_result: ConversionResult
    ) -> None:
        """ConversionResult includes the Frankfurter source label."""
        assert "Frankfurter" in sample_conversion_result.source

    def test_conversion_includes_rate_date(
        self, sample_conversion_result: ConversionResult
    ) -> None:
        """ConversionResult includes the rate date."""
        assert isinstance(sample_conversion_result.rate_date, date)


@pytest.mark.unit
@pytest.mark.mcp
class TestCurrencyToolValidation:
    """Tests for input validation in CurrencyTool."""

    def test_convert_zero_amount_raises_value_error(self) -> None:
        """Convert raises ValueError for amount <= 0."""
        tool = CurrencyTool()
        with pytest.raises((ValueError, NotImplementedError)):
            tool.convert(0, "INR", "SGD")

    def test_convert_negative_amount_raises_value_error(self) -> None:
        """Convert raises ValueError for negative amounts."""
        tool = CurrencyTool()
        with pytest.raises((ValueError, NotImplementedError)):
            tool.convert(-100, "INR", "SGD")

    def test_as_langchain_tool_raises_not_implemented(self) -> None:
        """as_langchain_tool raises NotImplementedError until implemented."""
        tool = CurrencyTool()
        with pytest.raises(NotImplementedError):
            tool.as_langchain_tool()


@pytest.mark.unit
@pytest.mark.mcp
class TestCurrencyToolHTTP:
    """Tests for CurrencyTool HTTP data-fetching with mocked requests.get.

    All tests patch requests.get so no real network calls are made.
    These tests verify the full pipeline from raw Frankfurter JSON
    to ConversionResult objects.
    """

    # Representative Frankfurter /latest JSON response
    _MOCK_API_RESPONSE: dict = {
        "amount": 1.0,
        "base": "INR",
        "date": "2024-01-01",
        "rates": {"SGD": 0.0164},
    }

    def _mock_http_response(self) -> MagicMock:
        """Build a mock requests.Response that returns the fixture JSON."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = self._MOCK_API_RESPONSE
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    def test_get_rate_calls_correct_url(self) -> None:
        """get_rate makes a GET request to the Frankfurter API URL."""
        from unittest.mock import patch

        tool = CurrencyTool()
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._mock_http_response()
            tool.get_rate("INR", "SGD")
            mock_get.assert_called_once()
            call_url: str = mock_get.call_args[0][0]
            assert "frankfurter" in call_url

    def test_get_rate_returns_positive_float(self) -> None:
        """get_rate returns a positive float exchange rate."""
        from unittest.mock import patch

        tool = CurrencyTool()
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._mock_http_response()
            rate = tool.get_rate("INR", "SGD")
            assert isinstance(rate, float)
            assert rate > 0

    def test_get_rate_connection_error_raises(self) -> None:
        """get_rate re-raises as ConnectionError when the API is unreachable."""
        from unittest.mock import patch

        import requests as requests_lib

        tool = CurrencyTool()
        with patch("requests.get", side_effect=requests_lib.ConnectionError("timeout")):
            with pytest.raises(ConnectionError):
                tool.get_rate("INR", "SGD")

    def test_convert_returns_conversion_result_type(self) -> None:
        """Convert returns a ConversionResult instance."""
        from unittest.mock import patch

        tool = CurrencyTool()
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._mock_http_response()
            result = tool.convert(50000.0, "INR", "SGD")
            assert isinstance(result, ConversionResult)

    def test_convert_amount_is_positive(self) -> None:
        """Convert result has a positive converted_amount."""
        from unittest.mock import patch

        tool = CurrencyTool()
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._mock_http_response()
            result = tool.convert(50000.0, "INR", "SGD")
            assert result.converted_amount > 0

    def test_convert_preserves_currency_codes(self) -> None:
        """Convert result preserves from_currency and to_currency correctly."""
        from unittest.mock import patch

        tool = CurrencyTool()
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._mock_http_response()
            result = tool.convert(50000.0, "INR", "SGD")
            assert result.from_currency == "INR"
            assert result.to_currency == "SGD"

    def test_convert_zero_amount_raises_value_error(self) -> None:
        """Convert raises ValueError for amount == 0 before making HTTP call."""
        tool = CurrencyTool()
        with pytest.raises(ValueError):
            tool.convert(0.0, "INR", "SGD")

    def test_convert_negative_amount_raises_value_error(self) -> None:
        """Convert raises ValueError for negative amounts before making HTTP call."""
        tool = CurrencyTool()
        with pytest.raises(ValueError):
            tool.convert(-100.0, "INR", "SGD")

    def test_convert_invalid_currency_raises_value_error(self) -> None:
        """Convert raises ValueError for unsupported currency codes."""
        tool = CurrencyTool()
        with pytest.raises(ValueError):
            tool.convert(100.0, "ZZZ", "SGD")

    def test_convert_includes_rate_date(self) -> None:
        """Convert result includes a rate_date that is a date instance."""
        from unittest.mock import patch

        tool = CurrencyTool()
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._mock_http_response()
            result = tool.convert(50000.0, "INR", "SGD")
            assert isinstance(result.rate_date, date)

    def test_convert_source_label_contains_frankfurter(self) -> None:
        """Convert result source label references Frankfurter."""
        from unittest.mock import patch

        tool = CurrencyTool()
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._mock_http_response()
            result = tool.convert(50000.0, "INR", "SGD")
            assert "Frankfurter" in result.source
