"""Tests for DocumentLoader — RAG Requirement 1.

TDD cycle: These tests define the expected behaviour BEFORE implementation.
All tests should initially FAIL (Red phase). Implement document_loader.py to make them pass.
"""

from pathlib import Path

import pytest

from features.rag.document_loader import DocumentLoader, KnowledgeDocument


@pytest.mark.unit
@pytest.mark.rag
class TestDocumentLoaderInit:
    """Tests for DocumentLoader initialisation."""

    def test_init_with_valid_directory(self, temp_knowledge_base_dir: Path) -> None:
        """DocumentLoader initialises successfully with a valid directory."""
        loader = DocumentLoader(knowledge_base_dir=temp_knowledge_base_dir)
        assert loader.knowledge_base_dir == temp_knowledge_base_dir

    def test_load_all_raises_on_missing_directory(self, tmp_path: Path) -> None:
        """load_all raises FileNotFoundError when directory does not exist."""
        missing_dir = tmp_path / "nonexistent"
        loader = DocumentLoader(knowledge_base_dir=missing_dir)
        with pytest.raises(FileNotFoundError):
            loader.load_all()


@pytest.mark.unit
@pytest.mark.rag
class TestMarkdownLoader:
    """Tests for Markdown document loading."""

    def test_load_markdown_returns_knowledge_document(
        self, temp_knowledge_base_dir: Path
    ) -> None:
        """load_markdown returns a KnowledgeDocument with correct content."""
        loader = DocumentLoader(knowledge_base_dir=temp_knowledge_base_dir)
        md_file = temp_knowledge_base_dir / "test_singapore.md"
        doc = loader.load_markdown(md_file)
        assert isinstance(doc, KnowledgeDocument)
        assert "Singapore" in doc.content

    def test_load_markdown_extracts_source_title(
        self, temp_knowledge_base_dir: Path
    ) -> None:
        """load_markdown extracts source_title from frontmatter."""
        loader = DocumentLoader(knowledge_base_dir=temp_knowledge_base_dir)
        md_file = temp_knowledge_base_dir / "test_singapore.md"
        doc = loader.load_markdown(md_file)
        assert doc.source_title == "Test Singapore Guide"

    def test_load_markdown_extracts_source_url(
        self, temp_knowledge_base_dir: Path
    ) -> None:
        """load_markdown extracts source_url from frontmatter."""
        loader = DocumentLoader(knowledge_base_dir=temp_knowledge_base_dir)
        md_file = temp_knowledge_base_dir / "test_singapore.md"
        doc = loader.load_markdown(md_file)
        assert doc.source_url == "https://example.com/singapore"

    def test_load_markdown_raises_on_missing_source_title(
        self, tmp_path: Path
    ) -> None:
        """load_markdown raises ValueError when source_title is missing from frontmatter."""
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        bad_md = kb_dir / "bad.md"
        bad_md.write_text("---\nsource_url: https://example.com\n---\n# Content")
        loader = DocumentLoader(knowledge_base_dir=kb_dir)
        with pytest.raises(ValueError, match="source_title"):
            loader.load_markdown(bad_md)

    def test_load_markdown_raises_on_missing_source_url(
        self, tmp_path: Path
    ) -> None:
        """load_markdown raises ValueError when source_url is missing from frontmatter."""
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        bad_md = kb_dir / "bad.md"
        bad_md.write_text("---\nsource_title: Test\n---\n# Content")
        loader = DocumentLoader(knowledge_base_dir=kb_dir)
        with pytest.raises(ValueError, match="source_url"):
            loader.load_markdown(bad_md)


@pytest.mark.unit
@pytest.mark.rag
class TestLoadAll:
    """Tests for loading all documents from the KB directory."""

    def test_load_all_returns_list(self, temp_knowledge_base_dir: Path) -> None:
        """load_all returns a list of KnowledgeDocument instances."""
        loader = DocumentLoader(knowledge_base_dir=temp_knowledge_base_dir)
        docs = loader.load_all()
        assert isinstance(docs, list)
        assert len(docs) >= 1

    def test_load_all_documents_have_content(
        self, temp_knowledge_base_dir: Path
    ) -> None:
        """All loaded documents have non-empty content."""
        loader = DocumentLoader(knowledge_base_dir=temp_knowledge_base_dir)
        docs = loader.load_all()
        assert all(doc.content.strip() for doc in docs)

    def test_load_all_documents_have_source_metadata(
        self, temp_knowledge_base_dir: Path
    ) -> None:
        """All loaded documents have source_title and source_url metadata."""
        loader = DocumentLoader(knowledge_base_dir=temp_knowledge_base_dir)
        docs = loader.load_all()
        for doc in docs:
            assert doc.source_title, f"Missing source_title in {doc.file_path}"
            assert doc.source_url, f"Missing source_url in {doc.file_path}"
