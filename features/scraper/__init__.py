"""Web scraper and knowledge base generator module."""

from features.scraper.kb_generator import KBGenerator
from features.scraper.web_scraper import ScrapedDocument, WebScraper

__all__ = ["ScrapedDocument", "WebScraper", "KBGenerator"]
