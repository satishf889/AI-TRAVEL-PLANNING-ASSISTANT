"""Embedding generator for the RAG pipeline.

Generates vector embeddings for document chunks using HuggingFace sentence-transformers.
Uses the all-MiniLM-L6-v2 model which runs locally — no API key required.

Requirements satisfied: RAG Requirement 3 (generate embeddings for chunks).
"""

from langchain_huggingface import HuggingFaceEmbeddings

from features.config.settings import settings
from features.rag.chunker import DocumentChunk


class Embedder:
    """Generates embeddings for document chunks using a local HuggingFace model.

    The model runs entirely on the local machine (CPU by default).
    No external API calls are made for embedding generation.
    """

    def __init__(self, model_name: str = settings.embedding_model_name,
                 device: str = settings.embedding_device) -> None:
        """Initialise the embedder and load the model settings.

        Args:
            model_name: HuggingFace model identifier for sentence-transformers.
            device: Device to run the model on ("cpu" or "cuda").
        """
        self.model_name = model_name
        self.device = device

    def get_langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """Return a LangChain-compatible HuggingFaceEmbeddings object.

        Returns:
            HuggingFaceEmbeddings instance configured for HuggingFace.
        """
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of text strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each is a list of floats).
        """
        embeddings_obj: str | object = self.get_langchain_embeddings()
        return embeddings_obj.embed_documents(texts)  # type: ignore[union-attr]

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[list[float]]:
        """Generate embeddings for a list of DocumentChunks.

        Args:
            chunks: List of DocumentChunk instances to embed.

        Returns:
            List of embedding vectors corresponding to each chunk's content.
        """
        texts = [chunk.content for chunk in chunks]
        return self.embed_texts(texts)
