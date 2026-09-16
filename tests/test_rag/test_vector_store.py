"""Tests for VectorStoreManager — RAG Requirement 4."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from features.rag.chunker import DocumentChunk
from features.rag.vector_store import VectorStoreManager


@pytest.fixture
def mock_chroma() -> MagicMock:
    with patch("features.rag.vector_store.Chroma") as mock_chroma_class:
        mock_instance = MagicMock()
        mock_instance.as_retriever.return_value = MagicMock()
        mock_instance._collection.count.return_value = 42
        mock_chroma_class.return_value = mock_instance
        mock_chroma_class.from_documents.return_value = mock_instance
        yield mock_chroma_class


@pytest.fixture
def mock_embedder() -> MagicMock:
    with patch("features.rag.vector_store.Embedder") as mock_embedder_class:
        mock_instance = MagicMock()
        mock_instance.get_langchain_embeddings.return_value = MagicMock()
        mock_embedder_class.return_value = mock_instance
        yield mock_embedder_class


@pytest.mark.unit
@pytest.mark.rag
class TestVectorStoreManager:
    """Tests for VectorStoreManager."""

    def test_init(self, tmp_path: Path) -> None:
        vsm = VectorStoreManager(tmp_path, "test_collection", "test-model")
        assert vsm.persist_directory == tmp_path
        assert vsm.collection_name == "test_collection"
        assert vsm.embedding_model_name == "test-model"
        assert vsm.embedding_device == "cpu"
        assert vsm._vector_store is None
        assert not vsm.is_initialized()

    def test_create_from_chunks(self, tmp_path: Path, mock_chroma: MagicMock, mock_embedder: MagicMock) -> None:
        vsm = VectorStoreManager(tmp_path, "test_collection", "test-model")
        chunks = [
            DocumentChunk(content="text", source_title="T", source_url="U", chunk_index=0, total_chunks=1)
        ]

        vsm.create_from_chunks(chunks)

        assert vsm.is_initialized()
        mock_chroma.from_documents.assert_called_once()
        args, kwargs = mock_chroma.from_documents.call_args
        docs = kwargs["documents"]
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].page_content == "text"
        assert docs[0].metadata == {"source_title": "T", "source_url": "U", "chunk_index": 0}
        assert kwargs["embedding"] == mock_embedder.return_value.get_langchain_embeddings.return_value
        assert kwargs["persist_directory"] == str(tmp_path)
        assert kwargs["collection_name"] == "test_collection"

    def test_create_from_chunks_empty(self, tmp_path: Path) -> None:
        vsm = VectorStoreManager(tmp_path, "test", "test")
        with pytest.raises(ValueError, match="empty"):
            vsm.create_from_chunks([])

    def test_load_success(self, tmp_path: Path, mock_chroma: MagicMock, mock_embedder: MagicMock) -> None:
        vsm = VectorStoreManager(tmp_path, "test_collection", "test-model")

        # Simulate an existing database
        (tmp_path / "chroma.sqlite3").touch()

        vsm.load()

        assert vsm.is_initialized()
        mock_chroma.assert_called_once_with(
            collection_name="test_collection",
            embedding_function=mock_embedder.return_value.get_langchain_embeddings.return_value,
            persist_directory=str(tmp_path)
        )

    def test_load_fails_when_missing(self, tmp_path: Path) -> None:
        vsm = VectorStoreManager(tmp_path, "test_collection", "test-model")

        with pytest.raises(FileNotFoundError):
            vsm.load()

    def test_get_retriever(self, tmp_path: Path, mock_chroma: MagicMock, mock_embedder: MagicMock) -> None:
        vsm = VectorStoreManager(tmp_path, "test", "test")
        (tmp_path / "chroma.sqlite3").touch()
        vsm.load()

        retriever = vsm.get_retriever(top_k=3)
        mock_chroma.return_value.as_retriever.assert_called_once_with(search_kwargs={"k": 3})
        assert retriever == mock_chroma.return_value.as_retriever.return_value

    def test_get_retriever_uninitialized(self, tmp_path: Path) -> None:
        vsm = VectorStoreManager(tmp_path, "test", "test")
        with pytest.raises(RuntimeError, match="not been created or loaded"):
            vsm.get_retriever()

    def test_document_count(self, tmp_path: Path, mock_chroma: MagicMock, mock_embedder: MagicMock) -> None:
        vsm = VectorStoreManager(tmp_path, "test", "test")
        (tmp_path / "chroma.sqlite3").touch()
        vsm.load()

        count = vsm.document_count()
        assert count == 42
