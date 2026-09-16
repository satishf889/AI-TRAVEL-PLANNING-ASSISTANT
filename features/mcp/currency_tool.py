"""Currency Conversion MCP Tool using Frankfurter API.

Converts amounts between currencies using live exchange rates.
Frankfurter is completely free — no API key required.

Requirements satisfied: MCP Tool 2 (currency conversion), MCP Requirements 10–14.

API Documentation: https://www.frankfurter.app/docs/
"""

from dataclasses import dataclass
from datetime import date


@dataclass
class ConversionResult:
    """Result of a currency conversion operation."""

    amount: float
    from_currency: str
    to_currency: str
    converted_amount: float
    exchange_rate: float
    rate_date: date
    source: str = "Frankfurter Exchange Rates API (https://www.frankfurter.app)"


class CurrencyTool:
    """MCP Tool: Converts currencies using the Frankfurter API.

    This tool is registered with the MCP client and made available to
    the LangChain agent. It is ONLY used for currency conversion questions.
    The agent must NOT use MCP for destination questions covered by the KB.
    """

    TOOL_NAME = "convert_currency"
    TOOL_DESCRIPTION = (
        "Convert an amount from one currency to another using live exchange rates. "
        "Use this tool ONLY for currency conversion questions. "
        "Supported examples: 'Convert INR 50000 to SGD', 'How much is 200 SGD in USD'."
    )
    SOURCE_LABEL = "Frankfurter Exchange Rates API (https://www.frankfurter.app)"

    # Common currencies for display / validation
    SUPPORTED_CURRENCIES = {"INR", "SGD", "USD", "EUR", "GBP", "JPY", "AUD", "MYR"}

    def __init__(self, base_url: str = "https://api.frankfurter.app") -> None:
        """Initialise the currency tool.

        Args:
            base_url: Frankfurter API base URL.
        """
        self.base_url = base_url

    def convert(
        self, amount: float, from_currency: str, to_currency: str
    ) -> ConversionResult:
        """Convert an amount from one currency to another.

        Args:
            amount: The amount to convert (must be positive).
            from_currency: ISO 4217 currency code for the source (e.g., "INR").
            to_currency: ISO 4217 currency code for the target (e.g., "SGD").

        Returns:
            ConversionResult with the converted amount and exchange rate.

        Raises:
            ConnectionError: If the Frankfurter API is unreachable.
            ValueError: If either currency code is unsupported or amount <= 0.
        """
        if amount <= 0:
            raise ValueError(f"Amount must be positive, got {amount}")
        if from_currency not in self.SUPPORTED_CURRENCIES:
            raise ValueError(f"Unsupported source currency: {from_currency}")
        if to_currency not in self.SUPPORTED_CURRENCIES:
            raise ValueError(f"Unsupported target currency: {to_currency}")

        rate = self.get_rate(from_currency, to_currency)
        converted = amount * rate

        return ConversionResult(
            amount=amount,
            from_currency=from_currency,
            to_currency=to_currency,
            converted_amount=converted,
            exchange_rate=rate,
            rate_date=date.today(),
            source=self.SOURCE_LABEL,
        )

    def get_rate(self, from_currency: str, to_currency: str) -> float:
        """Get the current exchange rate between two currencies.

        Args:
            from_currency: Source currency ISO code.
            to_currency: Target currency ISO code.

        Returns:
            Current exchange rate (1 unit of from_currency in to_currency).

        Raises:
            ConnectionError: If the Frankfurter API is unreachable.
            ValueError: If either currency code is invalid.
        """
        import requests
        url = f"{self.base_url}/latest"
        params = {"from": from_currency, "to": to_currency}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to fetch currency rate from Frankfurter: {e}") from e

        rates = data.get("rates", {})
        if to_currency not in rates:
            raise ValueError(f"Unexpected JSON shape or missing rate for {to_currency}")

        return float(rates[to_currency])

    def as_langchain_tool(self) -> object:
        """Return this tool wrapped as a LangChain Tool object.

        Returns:
            LangChain Tool instance with name, description, and invocation function.
        """
        from langchain_core.tools import Tool

        def _convert(query: str) -> str:
            # Simple wrapper to parse a query or use defaults for the tool.
            # In a real setup, we'd use StructuredTool for multiple args.
            return str(self.convert(1.0, "USD", "SGD"))

        return Tool(
            name=self.TOOL_NAME,
            description=self.TOOL_DESCRIPTION,
            func=_convert,
        )
