"""Document loader for the Singapore travel knowledge base.

Loads Markdown, PDF, and HTML files from the knowledge_base directory.
Each document is tagged with source metadata (title + URL) for citations.

Requirements satisfied: RAG Requirement 1 (load travel content).
"""

from dataclasses import dataclass
from pathlib import Path

import fitz
from bs4 import BeautifulSoup


@dataclass
class KnowledgeDocument:
    """Represents a loaded knowledge base document with source metadata."""

    content: str
    source_title: str
    source_url: str
    file_path: Path


class DocumentLoader:
    """Loads travel knowledge base documents from the filesystem.

    Supports Markdown (.md), PDF (.pdf), and HTML (.html/.htm) files.
    Each file must contain a metadata header specifying source_title and source_url.
    """

    def __init__(self, knowledge_base_dir: Path) -> None:
        """Initialise the loader with the knowledge base directory path.

        Args:
            knowledge_base_dir: Path to the directory containing KB documents.
        """
        self.knowledge_base_dir = knowledge_base_dir

    def load_all(self) -> list[KnowledgeDocument]:
        """Load all supported documents from the knowledge base directory.

        Returns:
            List of KnowledgeDocument instances with content and metadata.

        Raises:
            FileNotFoundError: If the knowledge_base_dir does not exist.
        """
        if not self.knowledge_base_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.knowledge_base_dir}")
        docs = []
        for file_path in self.knowledge_base_dir.iterdir():
            if file_path.is_file():
                if file_path.suffix == '.md':
                    docs.append(self.load_markdown(file_path))
                elif file_path.suffix == '.pdf':
                    docs.append(self.load_pdf(file_path))
                elif file_path.suffix in ['.html', '.htm']:
                    docs.append(self.load_html(file_path))
        return docs

    def load_markdown(self, file_path: Path) -> KnowledgeDocument:
        """Load a Markdown document and extract its frontmatter metadata.

        Args:
            file_path: Absolute path to the .md file.

        Returns:
            KnowledgeDocument with content and source metadata.

        Raises:
            ValueError: If the file is missing required metadata fields.
        """
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        source_title, source_url = None, None

        if lines and lines[0].strip() == '---':
            for i, line in enumerate(lines[1:]):
                if line.strip() == '---':
                    content = '\n'.join(lines[i+2:])
                    break
                if line.startswith('source_title:'):
                    source_title = line.split(':', 1)[1].strip()
                elif line.startswith('source_url:'):
                    source_url = line.split(':', 1)[1].strip()

        if not source_title:
            raise ValueError(f"Missing source_title in {file_path}")
        if not source_url:
            raise ValueError(f"Missing source_url in {file_path}")

        return KnowledgeDocument(
            content=content.strip(),
            source_title=source_title,
            source_url=source_url,
            file_path=file_path
        )

    def load_pdf(self, file_path: Path) -> KnowledgeDocument:
        """Load a PDF document using PyMuPDF.

        Args:
            file_path: Absolute path to the .pdf file.

        Returns:
            KnowledgeDocument with text content and source metadata.
        """
        content = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                content += page.get_text()

        lines = content.split('\n')
        source_title, source_url = "Unknown PDF", "unknown://pdf"
        if lines and lines[0].strip() == '---':
            for _i, line in enumerate(lines[1:]):
                if line.strip() == '---':
                    break
                if line.startswith('source_title:'):
                    source_title = line.split(':', 1)[1].strip()
                elif line.startswith('source_url:'):
                    source_url = line.split(':', 1)[1].strip()

        return KnowledgeDocument(
            content=content,
            source_title=source_title,
            source_url=source_url,
            file_path=file_path
        )

    def load_html(self, file_path: Path) -> KnowledgeDocument:
        """Load an HTML document and strip tags using BeautifulSoup.

        Args:
            file_path: Absolute path to the .html or .htm file.

        Returns:
            KnowledgeDocument with plain text content and source metadata.
        """
        html_content = file_path.read_text(encoding='utf-8')
        soup = BeautifulSoup(html_content, 'html.parser')

        title_tag = soup.find('title')
        source_title = title_tag.text if title_tag else "Unknown HTML"

        meta_url = soup.find('meta', {'name': 'source_url'})
        source_url = meta_url['content'] if meta_url else "unknown://html"

        content_text = soup.get_text(separator='\n', strip=True)
        return KnowledgeDocument(
            content=content_text,
            source_title=source_title,
            source_url=source_url,
            file_path=file_path
        )
