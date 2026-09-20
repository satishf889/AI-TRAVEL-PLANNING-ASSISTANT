"""ChromaDB vector store manager for the RAG pipeline.

Handles creating, persisting, and querying the ChromaDB collection
that stores embedded document chunks.

Requirements satisfied: RAG Requirement 4 (store embeddings in a vector store).
"""

from pathlib import Path

from langchain_chroma import Chroma

from features.rag.chunker import DocumentChunk
from features.rag.embedder import Embedder


class VectorStoreManager:
    """Manages the ChromaDB vector store for the Singapore knowledge base.

    Supports both creation (ingestion time) and loading (query time) of
    the vector store using Embedder. The store is persisted to disk.
    """

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
    ) -> None:
        """Initialise the vector store manager.

        Args:
            persist_directory: Directory where ChromaDB persists its data.
            collection_name: Name of the ChromaDB collection.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._vector_store = None

    def _get_embeddings(self) -> object:
        """Instantiate embeddings using Embedder."""
        return Embedder().get_langchain_embeddings()

    def create_from_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Create and persist a vector store from document chunks.

        Embeds all chunks and stores them with their source metadata.
        This is called during the ingestion phase (scripts/ingest_knowledge_base.py).

        Args:
            chunks: List of DocumentChunk instances to embed and store.

        Raises:
            ValueError: If chunks list is empty.
        """
        if not chunks:
            raise ValueError("chunks list cannot be empty")

        embeddings = self._get_embeddings()

        texts = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "source_title": chunk.source_title,
                "source_url": chunk.source_url,
                "chunk_index": chunk.chunk_index
            }
            for chunk in chunks
        ]

        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._vector_store = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            collection_name=self.collection_name,
            persist_directory=str(self.persist_directory)
        )

    def load(self) -> None:
        """Load an existing persisted vector store from disk.

        Raises:
            FileNotFoundError: If the persist directory does not contain a valid store.
        """
        if not self.persist_directory.exists():
            raise FileNotFoundError(f"Persist directory not found: {self.persist_directory}")

        embeddings = self._get_embeddings()

        self._vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=str(self.persist_directory)
        )

    def get_retriever(self, top_k: int = 5) -> object:
        """Return a LangChain-compatible retriever for the vector store.

        Args:
            top_k: Number of top similar chunks to retrieve per query.

        Returns:
            LangChain VectorStoreRetriever instance.

        Raises:
            RuntimeError: If the vector store has not been created or loaded.
        """
        if not self.is_initialized():
            raise RuntimeError("Vector store not initialized")
        return self._vector_store.as_retriever(search_kwargs={"k": top_k})

    def is_initialized(self) -> bool:
        """Check whether the vector store has been loaded or created.

        Returns:
            True if the vector store is ready for queries, False otherwise.
        """
        return self._vector_store is not None

    def document_count(self) -> int:
        """Return the number of documents stored in the collection.

        Returns:
            Count of documents in the ChromaDB collection.
        """
        if not self.is_initialized():
            raise RuntimeError("Vector store not initialized")
        return len(self._vector_store.get()["ids"])
