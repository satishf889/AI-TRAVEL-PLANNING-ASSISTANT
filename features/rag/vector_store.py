"""ChromaDB vector store manager for the RAG pipeline.

Handles creating, persisting, and querying the ChromaDB collection
that stores embedded document chunks.

Requirements satisfied: RAG Requirement 4 (store embeddings in a vector store).
"""

from pathlib import Path

from features.rag.chunker import DocumentChunk


class VectorStoreManager:
    """Manages the ChromaDB vector store for the Singapore knowledge base.

    Supports both creation (ingestion time) and loading (query time) of
    the vector store. The store is persisted to disk for reuse across sessions.
    """

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
        embedding_model_name: str,
        embedding_device: str = "cpu",
    ) -> None:
        """Initialise the vector store manager.

        Args:
            persist_directory: Directory where ChromaDB persists its data.
            collection_name: Name of the ChromaDB collection.
            embedding_model_name: HuggingFace model for embeddings.
            embedding_device: Device for the embedding model ("cpu" or "cuda").
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.embedding_device = embedding_device
        self._vector_store = None

    def create_from_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Create and persist a vector store from document chunks.

        Embeds all chunks and stores them with their source metadata.
        This is called during the ingestion phase (scripts/ingest_knowledge_base.py).

        Args:
            chunks: List of DocumentChunk instances to embed and store.

        Raises:
            ValueError: If chunks list is empty.
        """
        raise NotImplementedError("Implement in TDD cycle")

    def load(self) -> None:
        """Load an existing persisted vector store from disk.

        Raises:
            FileNotFoundError: If the persist directory does not contain a valid store.
        """
        raise NotImplementedError("Implement in TDD cycle")

    def get_retriever(self, top_k: int = 5) -> object:
        """Return a LangChain-compatible retriever for the vector store.

        Args:
            top_k: Number of top similar chunks to retrieve per query.

        Returns:
            LangChain VectorStoreRetriever instance.

        Raises:
            RuntimeError: If the vector store has not been created or loaded.
        """
        raise NotImplementedError("Implement in TDD cycle")

    def is_initialized(self) -> bool:
        """Check whether the vector store has been loaded or created.

        Returns:
            True if the vector store is ready for queries, False otherwise.
        """
        raise NotImplementedError("Implement in TDD cycle")

    def document_count(self) -> int:
        """Return the number of documents stored in the collection.

        Returns:
            Count of documents in the ChromaDB collection.
        """
        raise NotImplementedError("Implement in TDD cycle")
