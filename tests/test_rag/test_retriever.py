"""Tests for Retriever — RAG Requirement 5, 6, 7."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from features.rag.retriever import KnowledgeRetriever, RetrievalResult


@pytest.fixture
def mock_vector_store_manager() -> MagicMock:
    manager = MagicMock()
    retriever_mock = MagicMock()
    manager.get_retriever.return_value = retriever_mock
    manager._vector_store = MagicMock()
    return manager


@pytest.mark.unit
@pytest.mark.rag
class TestKnowledgeRetriever:
    """Tests for KnowledgeRetriever."""

    def test_retrieve_success(self, mock_vector_store_manager: MagicMock) -> None:
        retriever = KnowledgeRetriever(mock_vector_store_manager, top_k=2)

        # Mock the underlying chroma vector store similarity search
        doc1 = Document(page_content="c1", metadata={"source_title": "t1", "source_url": "u1", "chunk_index": 0})
        doc2 = Document(page_content="c2", metadata={"source_title": "t2", "source_url": "u2", "chunk_index": 1})
        mock_vector_store_manager._vector_store.similarity_search_with_relevance_scores.return_value = [
            (doc1, 0.9), (doc2, 0.8)
        ]

        results = retriever.retrieve("query")

        mock_vector_store_manager._vector_store.similarity_search_with_relevance_scores.assert_called_once_with("query", k=2)
        assert len(results) == 2
        assert isinstance(results[0], RetrievalResult)
        assert results[0].content == "c1"
        assert results[0].source_title == "t1"
        assert results[0].source_url == "u1"
        assert results[0].relevance_score == 0.9
        assert results[0].chunk_index == 0

    def test_retrieve_empty_query(self, mock_vector_store_manager: MagicMock) -> None:
        retriever = KnowledgeRetriever(mock_vector_store_manager)
        with pytest.raises(ValueError):
            retriever.retrieve("")

    def test_retrieve_with_fallback_message_success(self, mock_vector_store_manager: MagicMock) -> None:
        retriever = KnowledgeRetriever(mock_vector_store_manager)
        doc1 = Document(page_content="c1", metadata={"source_title": "t1", "source_url": "u1", "chunk_index": 0})
        mock_vector_store_manager._vector_store.similarity_search_with_relevance_scores.return_value = [(doc1, 0.9)]

        results, msg = retriever.retrieve_with_fallback_message("query")
        assert len(results) == 1
        assert msg is None

    def test_retrieve_with_fallback_message_empty(self, mock_vector_store_manager: MagicMock) -> None:
        retriever = KnowledgeRetriever(mock_vector_store_manager)
        mock_vector_store_manager._vector_store.similarity_search_with_relevance_scores.return_value = []

        results, msg = retriever.retrieve_with_fallback_message("query")
        assert len(results) == 0
        assert msg == "No relevant information found in the destination knowledge base. Do not fabricate an answer. State clearly that the knowledge base does not contain this information."

    def test_format_context(self, mock_vector_store_manager: MagicMock) -> None:
        retriever = KnowledgeRetriever(mock_vector_store_manager)
        results = [
            RetrievalResult(content="content1", source_title="Title1", source_url="Url1", relevance_score=0.9, chunk_index=0),
            RetrievalResult(content="content2", source_title="Title2", source_url="Url2", relevance_score=0.8, chunk_index=1),
        ]
        context = retriever.format_context(results)

        assert "Source: Title1 (Url1)" in context
        assert "content1" in context
        assert "Source: Title2 (Url2)" in context
        assert "content2" in context
