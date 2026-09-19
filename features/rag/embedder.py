"""Embedding generator for the RAG pipeline.

Generates vector embeddings for document chunks using HuggingFace sentence-transformers.
Uses the all-MiniLM-L6-v2 model which runs locally — no API key required.

Requirements satisfied: RAG Requirement 3 (generate embeddings for chunks).
"""

from features.rag.chunker import DocumentChunk


class Embedder:
    """Generates embeddings for document chunks using a local HuggingFace model.

    The model runs entirely on the local machine (CPU by default).
    No external API calls are made for embedding generation.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu") -> None:
        """Initialise the embedder and load the model.

        Args:
            model_name: HuggingFace model identifier for sentence-transformers.
            device: Device to run the model on ("cpu" or "cuda").
        """
        self.model_name = model_name
        self.device = device
        self._model = None  # Lazy-loaded on first use

    def get_langchain_embeddings(self) -> object:
        """Return a LangChain-compatible HuggingFaceEmbeddings object.

        This is the primary interface used by the vector store and retriever.

        Returns:
            HuggingFaceEmbeddings instance configured with the specified model.
        """
        raise NotImplementedError("Implement in TDD cycle")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of text strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each is a list of floats).
        """
        raise NotImplementedError("Implement in TDD cycle")

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[list[float]]:
        """Generate embeddings for a list of DocumentChunks.

        Args:
            chunks: List of DocumentChunk instances to embed.

        Returns:
            List of embedding vectors corresponding to each chunk's content.
        """
        raise NotImplementedError("Implement in TDD cycle")
