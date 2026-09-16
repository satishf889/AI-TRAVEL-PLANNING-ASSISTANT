"""Tests for Embedder — RAG Requirement 3.

TDD cycle: These tests define the expected behaviour BEFORE implementation.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.embeddings import Embeddings

from features.rag.chunker import DocumentChunk
from features.rag.embedder import Embedder


@pytest.fixture
def mock_huggingface_embeddings() -> MagicMock:
    with patch("features.rag.embedder.HuggingFaceEmbeddings") as mock_hf:
        mock_instance = MagicMock(spec=Embeddings)
        mock_instance.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_hf.return_value = mock_instance
        yield mock_hf


@pytest.mark.unit
@pytest.mark.rag
class TestEmbedder:
    """Tests for Embedder."""

    def test_init_with_defaults(self) -> None:
        embedder = Embedder()
        assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert embedder.device == "cpu"

    def test_get_langchain_embeddings_lazy_loads(self, mock_huggingface_embeddings: MagicMock) -> None:
        embedder = Embedder(model_name="test-model", device="cuda")
        lc_embedder = embedder.get_langchain_embeddings()

        mock_huggingface_embeddings.assert_called_once_with(
            model_name="test-model", model_kwargs={"device": "cuda"}
        )
        assert embedder._model == mock_huggingface_embeddings.return_value
        assert lc_embedder == mock_huggingface_embeddings.return_value

    def test_embed_texts(self, mock_huggingface_embeddings: MagicMock) -> None:
        embedder = Embedder()
        texts = ["text1", "text2"]
        embeddings = embedder.embed_texts(texts)

        mock_instance = mock_huggingface_embeddings.return_value
        mock_instance.embed_documents.assert_called_once_with(texts)
        assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    def test_embed_chunks(self, mock_huggingface_embeddings: MagicMock) -> None:
        embedder = Embedder()
        chunks = [
            DocumentChunk(content="text1", source_title="", source_url="", chunk_index=0, total_chunks=1),
            DocumentChunk(content="text2", source_title="", source_url="", chunk_index=1, total_chunks=1),
        ]
        embeddings = embedder.embed_chunks(chunks)

        mock_instance = mock_huggingface_embeddings.return_value
        mock_instance.embed_documents.assert_called_once_with(["text1", "text2"])
        assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
