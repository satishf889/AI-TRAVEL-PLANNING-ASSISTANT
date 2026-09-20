"""Unit tests for the Embedder module."""

from unittest.mock import MagicMock, patch

from features.rag.chunker import DocumentChunk
from features.rag.embedder import Embedder


def test_embedder_init() -> None:
    """Test initializing the Embedder."""
    embedder = Embedder()
    assert embedder is not None


@patch("features.rag.embedder.HuggingFaceEmbeddings")
def test_get_langchain_embeddings(mock_hf_embeddings: MagicMock) -> None:
    """Test get_langchain_embeddings creates a HuggingFaceEmbeddings instance."""
    embedder = Embedder()
    embeddings = embedder.get_langchain_embeddings()
    assert embeddings is not None
    mock_hf_embeddings.assert_called_once()


@patch("features.rag.embedder.Embedder.get_langchain_embeddings")
def test_embed_texts(mock_get_embeddings: MagicMock) -> None:
    """Test embedding a list of text strings."""
    mock_embeddings_obj = MagicMock()
    mock_embeddings_obj.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock_get_embeddings.return_value = mock_embeddings_obj

    embedder = Embedder()
    result = embedder.embed_texts(["test string"])

    assert result == [[0.1, 0.2, 0.3]]
    mock_embeddings_obj.embed_documents.assert_called_once_with(["test string"])


@patch("features.rag.embedder.Embedder.embed_texts")
def test_embed_chunks(mock_embed_texts: MagicMock) -> None:
    """Test embedding DocumentChunk instances."""
    mock_embed_texts.return_value = [[0.1, 0.2, 0.3]]

    chunk = DocumentChunk(
        content="test content",
        source_title="Title",
        source_url="http://example.com",
        chunk_index=0,
        total_chunks=1,
    )

    embedder = Embedder()
    result = embedder.embed_chunks([chunk])

    assert result == [[0.1, 0.2, 0.3]]
    mock_embed_texts.assert_called_once_with(["test content"])
