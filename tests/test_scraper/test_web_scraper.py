"""Unit tests for features.scraper.web_scraper."""

from unittest.mock import ANY, MagicMock, patch

import pytest
from features.scraper.web_scraper import ScrapedDocument, WebScraper


def test_fetch_wikivoyage_api_success() -> None:
    """Test fetching page content via official MediaWiki API endpoint."""
    scraper = WebScraper()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "parse": {
            "title": "Singapore",
            "text": {"*": "<div><h1>Singapore</h1><p>Singapore travel guide text.</p></div>"},
        }
    }

    with patch("httpx.get", return_value=mock_response) as mock_get:
        html = scraper.fetch_wikivoyage_api("Singapore")
        assert "Singapore travel guide text" in html
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args[1]
        assert "api.php" in mock_get.call_args[0][0]
        assert call_kwargs["params"]["page"] == "Singapore"


def test_fetch_page_wikivoyage_routing() -> None:
    """Test automatic MediaWiki API routing when URL is a Wikivoyage link."""
    scraper = WebScraper()
    wiki_url = "https://en.wikivoyage.org/wiki/Singapore"
    expected_html = "<html><body><p>Wikivoyage API text</p></body></html>"

    with patch.object(scraper, "fetch_wikivoyage_api", return_value=expected_html) as mock_api:
        html = scraper.fetch_page(wiki_url)
        assert html == expected_html
        mock_api.assert_called_once_with("Singapore")


def test_fetch_page_success() -> None:
    """Test fetching HTML content from standard URL."""
    scraper = WebScraper()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = (
        "<html><head><title>Test Page</title></head>"
        "<body><h1>Welcome</h1></body></html>"
    )
    mock_response.raise_for_status.return_value = None

    with patch("httpx.get", return_value=mock_response) as mock_get:
        html = scraper.fetch_page("https://example.com/travel")
        assert "Welcome" in html
        mock_get.assert_called_once_with(
            "https://example.com/travel",
            timeout=10.0,
            headers=ANY,
            follow_redirects=True,
        )


def test_fetch_page_http_error() -> None:
    """Test handling of HTTP error when fetching page."""
    scraper = WebScraper()
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("404 Not Found")

    with patch("httpx.get", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Failed to fetch page"):
            scraper.fetch_page("https://example.com/404")


def test_parse_html_clean_content() -> None:
    """Test stripping non-content tags and extracting title & article text using BeautifulSoup."""
    scraper = WebScraper()
    raw_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Singapore Travel Guide - Wikivoyage</title>
        <script>console.log("ads");</script>
        <style>body { color: red; }</style>
    </head>
    <body>
        <nav>Navigation links</nav>
        <article>
            <h1>Singapore</h1>
            <p>Singapore is a sunny island city-state in Southeast Asia.</p>
            <p>Popular attractions include Marina Bay Sands and Gardens by the Bay.</p>
        </article>
        <footer>Footer content</footer>
    </body>
    </html>
    """
    doc = scraper.parse_html(raw_html, "https://en.wikivoyage.org/wiki/Singapore")

    assert isinstance(doc, ScrapedDocument)
    assert doc.url == "https://en.wikivoyage.org/wiki/Singapore"
    assert "Singapore Travel Guide" in doc.title
    assert "Singapore is a sunny island" in doc.content
    assert "Gardens by the Bay" in doc.content
    assert "Navigation links" not in doc.content
    assert "Footer content" not in doc.content
    assert "console.log" not in doc.content


def test_parse_html_fallback_no_article_tag() -> None:
    """Test HTML parsing fallback when article/main tags are absent."""
    scraper = WebScraper()
    raw_html = """
    <html>
    <head><title>Simple Page</title></head>
    <body>
        <div>Top attraction: Merlion Park</div>
    </body>
    </html>
    """
    doc = scraper.parse_html(raw_html, "https://example.com/singapore")

    assert doc.title == "Simple Page"
    assert "Merlion Park" in doc.content


def test_scrape_url_end_to_end() -> None:
    """Test scrape_url combining fetch and parse."""
    scraper = WebScraper()
    raw_html = (
        "<html><head><title>Test</title></head>"
        "<body><main><p>Content</p></main></body></html>"
    )

    with patch.object(scraper, "fetch_page", return_value=raw_html):
        doc = scraper.scrape_url("https://example.com/test")
        assert doc.title == "Test"
        assert "Content" in doc.content
