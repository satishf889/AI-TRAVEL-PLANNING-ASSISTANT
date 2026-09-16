"""Tests for DocumentChunker — RAG Requirement 2.

TDD: Write tests first. Implement chunker.py to make them pass.
"""

import pytest

from features.rag.chunker import DocumentChunk, DocumentChunker
from features.rag.document_loader import KnowledgeDocument


@pytest.fixture
def long_document() -> KnowledgeDocument:
    """A document long enough to produce multiple chunks."""
    return KnowledgeDocument(
        content=" ".join(["Singapore is a fascinating city-state."] * 100),
        source_title="Test Source",
        source_url="https://example.com",
        file_path=None,  # type: ignore[arg-type]
    )


@pytest.mark.unit
@pytest.mark.rag
class TestDocumentChunker:
    """Tests for DocumentChunker."""

    def test_init_with_default_settings(self) -> None:
        """DocumentChunker initialises with default chunk_size=1000, overlap=200."""
        chunker = DocumentChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200

    def test_chunk_document_returns_list(
        self, sample_knowledge_document: KnowledgeDocument
    ) -> None:
        """chunk_document returns a list of DocumentChunk instances."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_document(sample_knowledge_document)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_chunks_inherit_source_metadata(
        self, sample_knowledge_document: KnowledgeDocument
    ) -> None:
        """Each chunk preserves source_title and source_url from the parent document."""
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(sample_knowledge_document)
        for chunk in chunks:
            assert chunk.source_title == sample_knowledge_document.source_title
            assert chunk.source_url == sample_knowledge_document.source_url

    def test_long_document_produces_multiple_chunks(
        self, long_document: KnowledgeDocument
    ) -> None:
        """A long document produces more than one chunk."""
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk_document(long_document)
        assert len(chunks) > 1

    def test_chunks_have_correct_index(self, long_document: KnowledgeDocument) -> None:
        """Chunks are indexed correctly from 0 to n-1."""
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk_document(long_document)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)

    def test_chunk_content_is_not_empty(
        self, sample_knowledge_document: KnowledgeDocument
    ) -> None:
        """No chunk should have empty content."""
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(sample_knowledge_document)
        assert all(chunk.content.strip() for chunk in chunks)

    def test_chunk_documents_processes_multiple_docs(
        self, sample_knowledge_document: KnowledgeDocument
    ) -> None:
        """chunk_documents returns a flat list from multiple documents."""
        chunker = DocumentChunker()
        docs = [sample_knowledge_document, sample_knowledge_document]
        chunks = chunker.chunk_documents(docs)
        assert isinstance(chunks, list)
        assert len(chunks) >= 2

    def test_empty_documents_list_returns_empty_chunks(self) -> None:
        """chunk_documents with empty list returns empty list."""
        chunker = DocumentChunker()
        assert chunker.chunk_documents([]) == []
