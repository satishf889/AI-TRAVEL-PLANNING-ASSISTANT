"""Document loader for the Singapore travel knowledge base.

Loads Markdown, PDF, and HTML files from the knowledge_base directory.
Each document is tagged with source metadata (title + URL) for citations.

Requirements satisfied: RAG Requirement 1 (load travel content).
"""

from dataclasses import dataclass
from pathlib import Path


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
            raise FileNotFoundError(f"Knowledge base directory not found: {self.knowledge_base_dir}")

        docs = []
        for file_path in self.knowledge_base_dir.glob("*.md"):
            if file_path.name.lower() == "readme.md":
                continue
            docs.append(self.load_markdown(file_path))
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
        content = file_path.read_text(encoding="utf-8")

        source_title = None
        source_url = None
        body_content = content

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1]
                body_content = parts[2].strip()

                for line in frontmatter.splitlines():
                    line = line.strip()
                    if line.startswith("source_title:"):
                        source_title = line.split(":", 1)[1].strip()
                    elif line.startswith("source_url:"):
                        source_url = line.split(":", 1)[1].strip()

        if not source_title:
            raise ValueError(f"Missing source_title in {file_path}")
        if not source_url:
            raise ValueError(f"Missing source_url in {file_path}")

        return KnowledgeDocument(
            content=body_content,
            source_title=source_title,
            source_url=source_url,
            file_path=file_path,
        )

    def load_pdf(self, file_path: Path) -> KnowledgeDocument:
        """Load a PDF document using PyMuPDF.

        Args:
            file_path: Absolute path to the .pdf file.

        Returns:
            KnowledgeDocument with text content and source metadata.
        """
        raise NotImplementedError("Implement in TDD cycle")

    def load_html(self, file_path: Path) -> KnowledgeDocument:
        """Load an HTML document and strip tags using BeautifulSoup.

        Args:
            file_path: Absolute path to the .html or .htm file.

        Returns:
            KnowledgeDocument with plain text content and source metadata.
        """
        raise NotImplementedError("Implement in TDD cycle")
