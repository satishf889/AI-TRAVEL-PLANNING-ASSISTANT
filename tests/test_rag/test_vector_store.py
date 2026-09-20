"""Tests for VectorStoreManager — RAG Requirement 4.

TDD cycle: These tests define the expected behaviour BEFORE implementation.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from features.rag.chunker import DocumentChunk
from features.rag.vector_store import VectorStoreManager


@pytest.fixture
def vector_store_manager(tmp_path: Path) -> VectorStoreManager:
    """Fixture for a VectorStoreManager instance."""
    return VectorStoreManager(
        persist_directory=tmp_path / "chroma_db",
        collection_name="test_collection",
    )


@pytest.mark.unit
@pytest.mark.rag
class TestVectorStoreManager:
    """Tests for VectorStoreManager."""

    def test_init_with_valid_parameters(self, tmp_path: Path) -> None:
        """VectorStoreManager initialises successfully."""
        persist_dir = tmp_path / "chroma_db"
        manager = VectorStoreManager(
            persist_directory=persist_dir,
            collection_name="test_collection",
        )
        assert manager.persist_directory == persist_dir
        assert manager.collection_name == "test_collection"
        assert not manager.is_initialized()

    def test_create_from_chunks_raises_on_empty(self, vector_store_manager: VectorStoreManager) -> None:
        """create_from_chunks raises ValueError if chunks list is empty."""
        with pytest.raises(ValueError, match="empty"):
            vector_store_manager.create_from_chunks([])

    @patch("features.rag.vector_store.Chroma")
    @patch("features.rag.vector_store.Embedder")
    def test_create_from_chunks_initialises_store(
        self,
        mock_embedder: MagicMock,
        mock_chroma: MagicMock,
        vector_store_manager: VectorStoreManager,
        sample_chunks: list[DocumentChunk],
    ) -> None:
        """create_from_chunks embeds and stores the provided chunks using Embedder."""
        mock_instance = MagicMock()
        mock_chroma.from_texts.return_value = mock_instance

        vector_store_manager.create_from_chunks(sample_chunks)

        assert mock_embedder.call_count == 1
        assert mock_chroma.from_texts.call_count == 1
        assert vector_store_manager.is_initialized()
        assert vector_store_manager._vector_store == mock_instance

    def test_load_raises_on_missing_directory(self, vector_store_manager: VectorStoreManager) -> None:
        """load raises FileNotFoundError if persist directory does not exist."""
        # The persist_directory doesn't exist yet
        with pytest.raises(FileNotFoundError):
            vector_store_manager.load()

    @patch("features.rag.vector_store.Chroma")
    @patch("features.rag.vector_store.Embedder")
    def test_load_initialises_store(
        self,
        mock_embedder: MagicMock,
        mock_chroma: MagicMock,
        vector_store_manager: VectorStoreManager,
    ) -> None:
        """load successfully initializes the store from disk."""
        # Create the directory so load() doesn't fail
        vector_store_manager.persist_directory.mkdir()

        mock_instance = MagicMock()
        mock_chroma.return_value = mock_instance

        vector_store_manager.load()

        assert mock_embedder.call_count == 1
        assert mock_chroma.call_count == 1
        assert vector_store_manager.is_initialized()
        assert vector_store_manager._vector_store == mock_instance

    def test_get_retriever_raises_if_not_initialized(self, vector_store_manager: VectorStoreManager) -> None:
        """get_retriever raises RuntimeError if called before initialization."""
        with pytest.raises(RuntimeError):
            vector_store_manager.get_retriever()

    def test_document_count_raises_if_not_initialized(self, vector_store_manager: VectorStoreManager) -> None:
        """document_count raises RuntimeError if called before initialization."""
        with pytest.raises(RuntimeError):
            vector_store_manager.document_count()

    def test_methods_delegate_to_chroma(self, vector_store_manager: VectorStoreManager) -> None:
        """Test that get_retriever and document_count delegate to the underlying Chroma instance."""
        mock_chroma = MagicMock()
        mock_chroma.as_retriever.return_value = "mock_retriever"
        # Mock get() returning a dict with "ids"
        mock_chroma.get.return_value = {"ids": ["id1", "id2", "id3"]}
        
        vector_store_manager._vector_store = mock_chroma
        
        assert vector_store_manager.get_retriever(top_k=2) == "mock_retriever"
        mock_chroma.as_retriever.assert_called_once_with(search_kwargs={"k": 2})
        
        assert vector_store_manager.document_count() == 3
        mock_chroma.get.assert_called_once()
