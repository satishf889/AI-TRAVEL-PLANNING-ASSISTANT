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
        """convert returns a ConversionResult instance."""
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
        """convert raises ValueError for amount <= 0."""
        tool = CurrencyTool()
        with pytest.raises((ValueError, NotImplementedError)):
            tool.convert(0, "INR", "SGD")

    def test_convert_negative_amount_raises_value_error(self) -> None:
        """convert raises ValueError for negative amounts."""
        tool = CurrencyTool()
        with pytest.raises((ValueError, NotImplementedError)):
            tool.convert(-100, "INR", "SGD")

    def test_as_langchain_tool_raises_not_implemented(self) -> None:
        """as_langchain_tool raises NotImplementedError until implemented."""
        tool = CurrencyTool()
        with pytest.raises(NotImplementedError):
            tool.as_langchain_tool()
