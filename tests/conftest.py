"""Shared pytest fixtures and mock factories for all test modules.

All external dependencies (LLM, embedding model, ChromaDB, HTTP APIs)
are mocked here to ensure tests are fast, deterministic, and offline.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from features.mcp.currency_tool import ConversionResult, CurrencyTool
from features.mcp.mcp_client import MCPClient
from features.mcp.weather_tool import WeatherCondition, WeatherForecast, WeatherTool
from features.orchestrator.context_manager import ConversationContextManager
from features.rag.chunker import DocumentChunk
from features.rag.document_loader import DocumentLoader, KnowledgeDocument


# ─── Sample Data Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def sample_knowledge_document() -> KnowledgeDocument:
    """A sample loaded knowledge document for testing."""
    return KnowledgeDocument(
        content="Singapore is a city-state in Southeast Asia. "
                "Gardens by the Bay is one of its most iconic attractions.",
        source_title="Wikivoyage Singapore Travel Guide",
        source_url="https://en.wikivoyage.org/wiki/Singapore",
        file_path=Path("/fake/wikivoyage_singapore.md"),
    )


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    """Sample document chunks for testing."""
    return [
        DocumentChunk(
            content="Gardens by the Bay is a must-visit attraction in Singapore.",
            source_title="Wikivoyage Singapore Travel Guide",
            source_url="https://en.wikivoyage.org/wiki/Singapore",
            chunk_index=0,
            total_chunks=2,
        ),
        DocumentChunk(
            content="The MRT (Mass Rapid Transit) is the most convenient way to travel in Singapore.",
            source_title="Visit Singapore: Essential Travel Information",
            source_url="https://www.visitsingapore.com/travel-guide-tips/getting-around/",
            chunk_index=0,
            total_chunks=1,
        ),
    ]


@pytest.fixture
def sample_weather_forecast() -> WeatherForecast:
    """Sample weather forecast for testing."""
    from datetime import date

    return WeatherForecast(
        city="Singapore",
        latitude=1.3521,
        longitude=103.8198,
        days=[
            WeatherCondition(
                date=date.today(),
                temperature_max_c=31.0,
                temperature_min_c=25.0,
                precipitation_mm=5.2,
                wind_speed_kmh=12.0,
                weather_description="Moderate rain showers",
                is_rainy=True,
                indoor_recommended=True,
            ),
        ],
        source="Open-Meteo (https://open-meteo.com)",
    )


@pytest.fixture
def sample_conversion_result() -> ConversionResult:
    """Sample currency conversion result for testing."""
    from datetime import date

    return ConversionResult(
        amount=50000.0,
        from_currency="INR",
        to_currency="SGD",
        converted_amount=820.0,
        exchange_rate=0.0164,
        rate_date=date.today(),
    )


# ─── Mock Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm() -> MagicMock:
    """Mock LangChain LLM that returns a predictable response."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="Mocked LLM response about Singapore.")
    return mock


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Mock ChromaDB vector store manager."""
    mock = MagicMock()
    mock.is_initialized.return_value = True
    mock.document_count.return_value = 42
    return mock


@pytest.fixture
def mock_retriever(sample_chunks: list[DocumentChunk]) -> MagicMock:
    """Mock KnowledgeRetriever that returns sample chunks."""
    from features.rag.retriever import RetrievalResult

    mock = MagicMock()
    mock.retrieve.return_value = [
        RetrievalResult(
            content=chunk.content,
            source_title=chunk.source_title,
            source_url=chunk.source_url,
            relevance_score=0.85,
            chunk_index=chunk.chunk_index,
        )
        for chunk in sample_chunks
    ]
    return mock


@pytest.fixture
def mock_weather_tool(sample_weather_forecast: WeatherForecast) -> MagicMock:
    """Mock WeatherTool that returns sample forecast data."""
    mock = MagicMock(spec=WeatherTool)
    mock.get_forecast.return_value = sample_weather_forecast
    mock.TOOL_NAME = WeatherTool.TOOL_NAME
    mock.TOOL_DESCRIPTION = WeatherTool.TOOL_DESCRIPTION
    mock.SOURCE_LABEL = WeatherTool.SOURCE_LABEL
    return mock


@pytest.fixture
def mock_currency_tool(sample_conversion_result: ConversionResult) -> MagicMock:
    """Mock CurrencyTool that returns sample conversion data."""
    mock = MagicMock(spec=CurrencyTool)
    mock.convert.return_value = sample_conversion_result
    mock.TOOL_NAME = CurrencyTool.TOOL_NAME
    mock.TOOL_DESCRIPTION = CurrencyTool.TOOL_DESCRIPTION
    mock.SOURCE_LABEL = CurrencyTool.SOURCE_LABEL
    return mock


@pytest.fixture
def mock_mcp_client(mock_weather_tool: MagicMock, mock_currency_tool: MagicMock) -> MagicMock:
    """Mock MCPClient with weather and currency tools."""
    mock = MagicMock(spec=MCPClient)
    mock.get_langchain_tools.return_value = [mock_weather_tool, mock_currency_tool]
    mock.is_tool_available.return_value = True
    return mock


@pytest.fixture
def context_manager() -> ConversationContextManager:
    """Fresh ConversationContextManager instance for each test."""
    return ConversationContextManager()


@pytest.fixture
def temp_knowledge_base_dir(tmp_path: Path) -> Path:
    """Temporary directory with sample knowledge base files."""
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()

    # Create a sample markdown knowledge document
    sample_md = kb_dir / "test_singapore.md"
    sample_md.write_text(
        "---\n"
        "source_title: Test Singapore Guide\n"
        "source_url: https://example.com/singapore\n"
        "---\n\n"
        "# Test Singapore Travel Guide\n\n"
        "Singapore is a vibrant city-state with diverse attractions.\n"
        "Gardens by the Bay offers stunning flower domes.\n"
    )

    return kb_dir
