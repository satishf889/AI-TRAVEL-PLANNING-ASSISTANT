"""Unit tests for features.scraper.kb_generator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from features.scraper.kb_generator import KBGenerator
from features.scraper.web_scraper import ScrapedDocument


@pytest.fixture
def sample_doc() -> ScrapedDocument:
    """Return a sample ScrapedDocument fixture for testing."""
    return ScrapedDocument(
        url="https://en.wikivoyage.org/wiki/Singapore",
        title="Singapore Travel Guide",
        content="Singapore is known for Marina Bay Sands and Gardens by the Bay.",
        raw_html="<html>...</html>",
    )


def test_format_with_llm_success(sample_doc: ScrapedDocument) -> None:
    """Test generating structured markdown via LLM."""
    generator = KBGenerator()
    mock_llm_response = MagicMock()
    mock_llm_response.content = (
        "# Singapore Travel Guide\n\n"
        "**Source URL**: https://en.wikivoyage.org/wiki/Singapore\n\n"
        "## Overview\nSingapore is known for Marina Bay Sands."
    )

    with patch("features.scraper.kb_generator.AzureChatOpenAI") as mock_azure_cls:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_llm_response
        mock_azure_cls.return_value = mock_instance

        markdown = generator.format_with_llm(sample_doc)
        assert "# Singapore Travel Guide" in markdown
        assert "Marina Bay Sands" in markdown


def test_format_with_llm_fallback_on_error(sample_doc: ScrapedDocument) -> None:
    """Test fallback when LLM fails."""
    generator = KBGenerator()

    with patch("features.scraper.kb_generator.AzureChatOpenAI") as mock_azure_cls:
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = Exception("API rate limit")
        mock_azure_cls.return_value = mock_instance

        markdown = generator.format_with_llm(sample_doc)
        assert "Singapore Travel Guide" in markdown
        assert sample_doc.url in markdown
        assert sample_doc.content in markdown


def test_save_markdown(tmp_path: Path, sample_doc: ScrapedDocument) -> None:
    """Test saving markdown string to disk."""
    generator = KBGenerator(output_dir=tmp_path)
    md_content = "# Singapore Guide\n\nSome text content."

    saved_path = generator.save_markdown(sample_doc, md_content)

    assert saved_path.exists()
    assert saved_path.parent == tmp_path
    saved_text = saved_path.read_text(encoding="utf-8")
    assert "# Singapore Guide" in saved_text


def test_process_and_save(tmp_path: Path, sample_doc: ScrapedDocument) -> None:
    """Test process_and_save end-to-end workflow."""
    generator = KBGenerator(output_dir=tmp_path)

    with patch.object(generator, "format_with_llm", return_value="# Formatted Markdown"):
        saved_path = generator.process_and_save(sample_doc)
        assert saved_path.exists()
        assert "# Formatted Markdown" in saved_path.read_text(encoding="utf-8")
