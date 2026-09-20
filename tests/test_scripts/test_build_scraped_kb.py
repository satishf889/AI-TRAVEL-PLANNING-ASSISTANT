"""Unit tests for standalone build_scraped_kb script."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.build_scraped_kb import main, run_pipeline

from features.scraper.web_scraper import ScrapedDocument


def test_run_pipeline_success(tmp_path: Path) -> None:
    """Test running scrape and build KB pipeline with custom URLs."""
    urls = ["https://en.wikivoyage.org/wiki/Singapore"]
    sample_doc = ScrapedDocument(
        url=urls[0],
        title="Singapore Guide",
        content="Attractions and transport info",
        raw_html="<html></html>",
    )

    with (
        patch("scripts.build_scraped_kb.WebScraper") as mock_scraper_cls,
        patch("scripts.build_scraped_kb.KBGenerator") as mock_gen_cls,
        patch("scripts.build_scraped_kb.DocumentLoader") as mock_loader_cls,
        patch("scripts.build_scraped_kb.DocumentChunker") as mock_chunker_cls,
        patch("scripts.build_scraped_kb.VectorStoreManager") as mock_vs_cls,
    ):
        mock_scraper_inst = MagicMock()
        mock_scraper_inst.scrape_url.return_value = sample_doc
        mock_scraper_cls.return_value = mock_scraper_inst

        mock_gen_inst = MagicMock()
        mock_gen_inst.process_and_save.return_value = tmp_path / "singapore_guide.md"
        mock_gen_cls.return_value = mock_gen_inst

        mock_loader_inst = MagicMock()
        mock_loader_inst.load_all.return_value = []
        mock_loader_cls.return_value = mock_loader_inst

        mock_chunker_inst = MagicMock()
        mock_chunker_inst.chunk_documents.return_value = []
        mock_chunker_cls.return_value = mock_chunker_inst

        mock_vs_inst = MagicMock()
        mock_vs_cls.return_value = mock_vs_inst

        run_pipeline(urls=urls, output_dir=tmp_path)

        mock_scraper_inst.scrape_url.assert_called_once_with(urls[0])
        mock_gen_inst.process_and_save.assert_called_once_with(sample_doc)
        mock_vs_inst.create_from_chunks.assert_called_once()


def test_run_pipeline_default_urls(tmp_path: Path) -> None:
    """Test running pipeline using default seed URLs."""
    sample_doc = ScrapedDocument(
        url="https://en.wikivoyage.org/wiki/Singapore",
        title="Singapore",
        content="Details",
        raw_html="<html></html>",
    )

    with (
        patch("scripts.build_scraped_kb.WebScraper") as mock_scraper_cls,
        patch("scripts.build_scraped_kb.KBGenerator") as mock_gen_cls,
        patch("scripts.build_scraped_kb.DocumentLoader") as mock_loader_cls,
        patch("scripts.build_scraped_kb.DocumentChunker") as mock_chunker_cls,
        patch("scripts.build_scraped_kb.VectorStoreManager") as mock_vs_cls,
    ):
        mock_scraper_inst = MagicMock()
        mock_scraper_inst.scrape_url.return_value = sample_doc
        mock_scraper_cls.return_value = mock_scraper_inst

        mock_gen_inst = MagicMock()
        mock_gen_inst.process_and_save.return_value = tmp_path / "singapore.md"
        mock_gen_cls.return_value = mock_gen_inst

        mock_loader_cls.return_value.load_all.return_value = []
        mock_chunker_cls.return_value.chunk_documents.return_value = []
        mock_vs_inst = MagicMock()
        mock_vs_cls.return_value = mock_vs_inst

        run_pipeline(urls=None, output_dir=tmp_path)

        assert mock_scraper_inst.scrape_url.called
        assert mock_gen_inst.process_and_save.called


def test_main_cli_execution(tmp_path: Path) -> None:
    """Test main CLI entrypoint parsing sys.argv."""
    with (
        patch(
            "sys.argv",
            [
                "build_scraped_kb.py",
                "--urls",
                "https://example.com/travel",
                "--output-dir",
                str(tmp_path),
            ],
        ),
        patch("scripts.build_scraped_kb.run_pipeline") as mock_run,
    ):
        main()
        mock_run.assert_called_once_with(
            urls=["https://example.com/travel"],
            output_dir=tmp_path,
        )
