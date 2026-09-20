"""Web scraper component for travel knowledge base acquisition.

Supports official MediaWiki REST API for Wikivoyage and BeautifulSoup4 parsing.
"""

from dataclasses import dataclass
import logging
import re
from urllib.parse import unquote

from bs4 import BeautifulSoup
import httpx

logger = logging.getLogger(__name__)


@dataclass
class ScrapedDocument:
    """Container for scraped web page content and metadata."""

    url: str
    title: str
    content: str
    raw_html: str


class WebScraper:
    """Web scraper utilizing official MediaWiki API and BeautifulSoup4."""

    MEDIAWIKI_API_URL = "https://en.wikivoyage.org/w/api.php"
    DEFAULT_USER_AGENT = (
        "TravelAssistantBot/1.0 (https://github.com/satishf889/AI-TRAVEL-PLANNING-ASSISTANT; contact@example.com)"
    )

    def __init__(self, timeout: float = 10.0, user_agent: str | None = None) -> None:
        """Initialize web scraper.

        Args:
            timeout: HTTP request timeout in seconds.
            user_agent: Custom User-Agent header string.
        """
        self.timeout = timeout
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT

    def fetch_wikivoyage_api(self, page_title: str) -> str:
        """Fetch page HTML content using official MediaWiki REST API.

        Args:
            page_title: Title of Wikivoyage travel page (e.g. 'Singapore').

        Returns:
            HTML string content.

        Raises:
            RuntimeError: If MediaWiki API request fails.
        """
        import time

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        params = {
            "action": "parse",
            "page": page_title,
            "prop": "text",
            "format": "json",
            "redirects": 1,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = httpx.get(
                    self.MEDIAWIKI_API_URL,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                    follow_redirects=True,
                )
                if response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3.0
                    logger.warning(f"429 Rate limit hit for '{page_title}'. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                data = response.json()

                if "error" in data:
                    error_info = data["error"].get("info", "MediaWiki API Error")
                    raise RuntimeError(f"MediaWiki API error for {page_title}: {error_info}")

                html_text: str = data["parse"]["text"]["*"]
                return html_text
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429 and attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 3.0)
                    continue
                logger.error(f"Failed to fetch Wikivoyage API page '{page_title}': {exc}")
                raise RuntimeError(f"Failed to fetch Wikivoyage API page '{page_title}': {exc}") from exc
            except Exception as exc:
                logger.error(f"Failed to fetch Wikivoyage API page '{page_title}': {exc}")
                raise RuntimeError(f"Failed to fetch Wikivoyage API page '{page_title}': {exc}") from exc

        raise RuntimeError(f"Failed to fetch Wikivoyage API page '{page_title}' after retries.")

    def _extract_wikivoyage_title(self, url: str) -> str | None:
        """Extract page title from Wikivoyage URL structure."""
        match = re.search(r"wikivoyage\.org/wiki/([^#?]+)", url)
        if match:
            return unquote(match.group(1)).replace("_", " ")
        return None

    def fetch_page(self, url: str) -> str:
        """Fetch raw HTML content from a URL.

        Automatically uses official MediaWiki API for Wikivoyage URLs.

        Args:
            url: Target URL to scrape.

        Returns:
            HTML string content.

        Raises:
            RuntimeError: If HTTP request fails.
        """
        wiki_title = self._extract_wikivoyage_title(url)
        if wiki_title:
            logger.info(f"Detected Wikivoyage URL. Routing via MediaWiki API for page: '{wiki_title}'")
            return self.fetch_wikivoyage_api(wiki_title)

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            response = httpx.get(
                url,
                timeout=self.timeout,
                headers=headers,
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.text
        except Exception as exc:
            logger.error(f"Failed to fetch web page {url}: {exc}")
            raise RuntimeError(f"Failed to fetch page {url}: {exc}") from exc

    def parse_html(self, html_content: str, url: str) -> ScrapedDocument:
        """Parse raw HTML and extract title, clean text content, and metadata.

        Args:
            html_content: Raw HTML text.
            url: Origin URL of the document.

        Returns:
            ScrapedDocument containing clean structured content.
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract title
        title = "Untitled Document"
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            h1 = soup.find("h1")
            if h1 and h1.text:
                title = h1.text.strip()

        if title == "Untitled Document":
            wiki_title = self._extract_wikivoyage_title(url)
            if wiki_title:
                title = wiki_title

        # Remove non-content tags
        non_content = ["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]
        for element in soup(non_content):
            element.decompose()

        # Target main content container if available
        main_container = (
            soup.find("article")
            or soup.find("main")
            or soup.find(id="content")
            or soup.body
        )

        if main_container:
            lines: list[str] = []
            elements = main_container.find_all(["h1", "h2", "h3", "h4", "p", "li", "div"])
            for elem in elements:
                if elem.find(["h1", "h2", "h3", "h4", "p", "li", "div"]):
                    continue
                text = elem.get_text(strip=True)
                if text:
                    lines.append(text)
            if lines:
                content = "\n\n".join(lines)
            else:
                content = main_container.get_text(separator="\n", strip=True)
        else:
            content = soup.get_text(separator="\n", strip=True)

        return ScrapedDocument(
            url=url,
            title=title,
            content=content,
            raw_html=html_content,
        )

    def scrape_url(self, url: str) -> ScrapedDocument:
        """Fetch and parse a web page URL.

        Args:
            url: Target URL to scrape.

        Returns:
            ScrapedDocument instance.
        """
        logger.info(f"Scraping travel facts from: {url}")
        html = self.fetch_page(url)
        return self.parse_html(html, url)
