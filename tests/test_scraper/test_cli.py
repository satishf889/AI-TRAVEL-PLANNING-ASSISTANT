"""Unit tests for features.scraper.cli."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from features.scraper.cli import run_scraper
from features.scraper.web_scraper import ScrapedDocument


def test_run_scraper_urls(tmp_path: Path) -> None:
    """Test CLI runner with explicit URLs."""
    urls = ["https://en.wikivoyage.org/wiki/Singapore"]
    sample_doc = ScrapedDocument(
        url=urls[0],
        title="Singapore",
        content="Travel details",
        raw_html="<html></html>",
    )

    with (
        patch("features.scraper.cli.WebScraper") as mock_scraper_cls,
        patch("features.scraper.cli.KBGenerator") as mock_gen_cls,
        patch("features.scraper.cli.VectorStoreManager") as mock_vs_cls,
    ):
        mock_scraper_inst = MagicMock()
        mock_scraper_inst.scrape_url.return_value = sample_doc
        mock_scraper_cls.return_value = mock_scraper_inst

        mock_gen_inst = MagicMock()
        mock_gen_inst.process_and_save.return_value = tmp_path / "singapore.md"
        mock_gen_cls.return_value = mock_gen_inst

        mock_vs_inst = MagicMock()
        mock_vs_cls.return_value = mock_vs_inst

        results = run_scraper(urls=urls, output_dir=tmp_path, index_vector_store=True)

        assert len(results) == 1
        mock_scraper_inst.scrape_url.assert_called_once_with(urls[0])
        mock_gen_inst.process_and_save.assert_called_once_with(sample_doc)
        mock_vs_inst.create_from_chunks.assert_called_once()


def test_cli_main(tmp_path: Path) -> None:
    """Test CLI main entrypoint parsing arguments."""
    from features.scraper.cli import main

    with (
        patch(
            "sys.argv",
            ["cli.py", "--urls", "https://example.com/test", "--output-dir", str(tmp_path)],
        ),
        patch("features.scraper.cli.run_scraper", return_value=[tmp_path / "test.md"]) as mock_run,
    ):
        main()
        mock_run.assert_called_once_with(
            urls=["https://example.com/test"],
            output_dir=tmp_path,
            index_vector_store=False,
        )
