"""Tests for KnowledgeRetriever — RAG Requirements 5, 6, 7."""

from unittest.mock import MagicMock

import pytest

from features.rag.retriever import KnowledgeRetriever, RetrievalResult
from features.orchestrator.prompt_templates import FALLBACK_NO_KB_CONTENT


@pytest.fixture
def mock_vector_store_manager():
    """Fixture for a mocked VectorStoreManager."""
    manager = MagicMock()
    manager.is_initialized.return_value = True
    
    mock_store = MagicMock()
    manager._vector_store = mock_store
    
    return manager


@pytest.mark.unit
@pytest.mark.rag
class TestKnowledgeRetriever:

    def test_init_sets_attributes(self, mock_vector_store_manager) -> None:
        retriever = KnowledgeRetriever(vector_store_manager=mock_vector_store_manager, top_k=3)
        assert retriever.vector_store_manager == mock_vector_store_manager
        assert retriever.top_k == 3

    def test_retrieve_raises_if_not_initialized(self, mock_vector_store_manager) -> None:
        mock_vector_store_manager.is_initialized.return_value = False
        retriever = KnowledgeRetriever(vector_store_manager=mock_vector_store_manager)
        with pytest.raises(RuntimeError, match="initialized"):
            retriever.retrieve("Singapore food")

    def test_retrieve_raises_on_empty_query(self, mock_vector_store_manager) -> None:
        retriever = KnowledgeRetriever(vector_store_manager=mock_vector_store_manager)
        with pytest.raises(ValueError, match="empty"):
            retriever.retrieve("")

    def test_retrieve_returns_results(self, mock_vector_store_manager) -> None:
        from langchain_core.documents import Document
        
        doc = Document(
            page_content="Marina Bay Sands is iconic.",
            metadata={
                "source_title": "Things to Do",
                "source_url": "https://example.com/things-to-do",
                "chunk_index": 0
            }
        )
        
        mock_vector_store_manager._vector_store.similarity_search_with_score.return_value = [(doc, 0.85)]
        
        retriever = KnowledgeRetriever(vector_store_manager=mock_vector_store_manager, top_k=5)
        results = retriever.retrieve("What is iconic?")
        
        assert len(results) == 1
        assert isinstance(results[0], RetrievalResult)
        assert results[0].content == "Marina Bay Sands is iconic."
        assert results[0].source_title == "Things to Do"
        assert results[0].source_url == "https://example.com/things-to-do"
        assert results[0].chunk_index == 0
        assert results[0].relevance_score == 0.85
        
        mock_vector_store_manager._vector_store.similarity_search_with_score.assert_called_once_with("What is iconic?", k=5)

    def test_retrieve_with_fallback_returns_results(self, mock_vector_store_manager) -> None:
        retriever = KnowledgeRetriever(vector_store_manager=mock_vector_store_manager)
        
        mock_result = RetrievalResult(
            content="test", source_title="test", source_url="test", relevance_score=0.9, chunk_index=1
        )
        retriever.retrieve = MagicMock(return_value=[mock_result])
        
        results, fallback = retriever.retrieve_with_fallback_message("query")
        assert results == [mock_result]
        assert fallback is None

    def test_retrieve_with_fallback_returns_fallback_if_empty(self, mock_vector_store_manager) -> None:
        retriever = KnowledgeRetriever(vector_store_manager=mock_vector_store_manager)
        retriever.retrieve = MagicMock(return_value=[])
        
        results, fallback = retriever.retrieve_with_fallback_message("query")
        assert results == []
        assert fallback == FALLBACK_NO_KB_CONTENT

    def test_format_context(self, mock_vector_store_manager) -> None:
        retriever = KnowledgeRetriever(vector_store_manager=mock_vector_store_manager)
        results = [
            RetrievalResult(
                content="First chunk.", source_title="Title A", source_url="http://a", relevance_score=1.0, chunk_index=0
            ),
            RetrievalResult(
                content="Second chunk.", source_title="Title B", source_url="http://b", relevance_score=0.9, chunk_index=1
            )
        ]
        
        context = retriever.format_context(results)
        assert "Source: Title A" in context
        assert "URL: http://a" in context
        assert "First chunk." in context
        assert "Source: Title B" in context
        assert "Second chunk." in context

    def test_invoke_returns_documents(self, mock_vector_store_manager) -> None:
        retriever = KnowledgeRetriever(vector_store_manager=mock_vector_store_manager)
        mock_result = RetrievalResult(
            content="Sample text",
            source_title="Guide",
            source_url="https://example.com/guide",
            relevance_score=0.95,
            chunk_index=2,
        )
        retriever.retrieve = MagicMock(return_value=[mock_result])

        docs = retriever.invoke("Tell me about Singapore")
        assert len(docs) == 1
        assert docs[0].page_content == "Sample text"
        assert docs[0].metadata["source_title"] == "Guide"
        assert docs[0].metadata["source_url"] == "https://example.com/guide"
        assert docs[0].metadata["title"] == "Guide"
        assert docs[0].metadata["source"] == "https://example.com/guide"
        assert docs[0].metadata["chunk_index"] == 2

