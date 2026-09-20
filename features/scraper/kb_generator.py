"""Knowledge base generator module.

Uses Azure OpenAI (gpt-5-mini) to structure scraped travel facts into
clean Markdown documents with source metadata headers.
"""

import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

from features.config.settings import settings
from features.scraper.web_scraper import ScrapedDocument

logger = logging.getLogger(__name__)


class KBGenerator:
    """Formats scraped travel web pages into knowledge base Markdown files."""

    SYSTEM_PROMPT = (
        "You are an expert travel knowledge base editor. Your task is to process raw scraped "
        "travel content about a destination and reformat it into a clean, well-structured "
        "Markdown travel document.\n\n"
        "Rules:\n"
        "1. MUST start the document with a YAML frontmatter block:\n"
        "---\n"
        "source_title: <Title>\n"
        "source_url: <URL>\n"
        "---\n\n"
        "2. Include a top title heading `# <Title>` under the frontmatter.\n"
        "3. Organize details into logical sections using `##` headings (e.g., Overview, "
        "Attractions, Transportation, Local Experiences, Cultural Tips, Sample Itineraries).\n"
        "4. Do NOT invent facts. Only use facts present in the provided text.\n"
        "5. Output clean Markdown only without any conversational wrapper."
    )

    def __init__(self, output_dir: Path | None = None) -> None:
        """Initialize KB Generator.

        Args:
            output_dir: Directory where formatted markdown files will be saved.
                        Defaults to settings.knowledge_base_directory.
        """
        self.output_dir = output_dir or settings.knowledge_base_directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_llm(self) -> AzureChatOpenAI:
        """Instantiate AzureChatOpenAI LLM using application settings."""
        return AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=SecretStr(settings.azure_openai_api_key),
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_deployment_name,
        )

    def format_with_llm(self, doc: ScrapedDocument) -> str:
        """Format scraped content into structured Markdown using Azure OpenAI.

        Args:
            doc: ScrapedDocument containing raw text and metadata.

        Returns:
            Structured Markdown string with YAML frontmatter.
        """
        try:
            llm = self._get_llm()
            user_prompt = (
                f"Document Title: {doc.title}\n"
                f"Source URL: {doc.url}\n\n"
                f"Raw Scraped Content:\n{doc.content[:8000]}"
            )

            response = llm.invoke(
                [
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
            content = str(response.content).strip()
            if content:
                # Ensure frontmatter is present
                if not content.startswith("---"):
                    frontmatter = f"---\nsource_title: {doc.title}\nsource_url: {doc.url}\n---\n\n"
                    content = frontmatter + content
                return content
        except Exception as exc:
            logger.warning(
                f"LLM formatting failed for {doc.url}, "
                f"falling back to raw markdown structure: {exc}"
            )

        # Fallback formatting with mandatory frontmatter if LLM fails or is unavailable
        return (
            f"---\n"
            f"source_title: {doc.title}\n"
            f"source_url: {doc.url}\n"
            f"---\n\n"
            f"# {doc.title}\n\n"
            f"## Scraped Travel Details\n\n"
            f"{doc.content}\n"
        )

    def _sanitize_filename(self, title: str) -> str:
        """Generate a filesystem-safe filename from a document title.

        Args:
            title: Document title string.

        Returns:
            Sanitized snake_case filename string.
        """
        clean = re.sub(r"[^\w\s-]", "", title).strip().lower()
        clean = re.sub(r"[-\s]+", "_", clean)
        return clean or "scraped_travel_doc"

    def save_markdown(self, doc: ScrapedDocument, markdown_content: str) -> Path:
        """Save Markdown string to output directory.

        Args:
            doc: ScrapedDocument instance for title/url.
            markdown_content: Formatted markdown string.

        Returns:
            Path object pointing to the saved markdown file.
        """
        filename = f"{self._sanitize_filename(doc.title)}.md"
        filepath = self.output_dir / filename
        filepath.write_text(markdown_content, encoding="utf-8")
        logger.info(f"Saved KB document to: {filepath}")
        return filepath

    def process_and_save(self, doc: ScrapedDocument) -> Path:
        """Process a scraped document with LLM formatting and persist to disk.

        Args:
            doc: ScrapedDocument instance.

        Returns:
            Path to saved markdown file.
        """
        markdown_text = self.format_with_llm(doc)
        return self.save_markdown(doc, markdown_text)
