"""ChromaDB vector store manager for the RAG pipeline.

Handles creating, persisting, and querying the ChromaDB collection
that stores embedded document chunks.

Requirements satisfied: RAG Requirement 4 (store embeddings in a vector store).
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from features.rag.chunker import DocumentChunk
from features.rag.embedder import Embedder


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
        if not chunks:
            raise ValueError("Cannot create vector store from an empty list of chunks.")

        docs = []
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk.content,
                metadata={
                    "source_title": chunk.source_title,
                    "source_url": chunk.source_url,
                    "chunk_index": chunk.chunk_index
                }
            ))

        embedder = Embedder(
            model_name=self.embedding_model_name,
            device=self.embedding_device
        ).get_langchain_embeddings()

        self._vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embedder,
            persist_directory=str(self.persist_directory),
            collection_name=self.collection_name
        )

    def load(self) -> None:
        """Load an existing persisted vector store from disk.

        Raises:
            FileNotFoundError: If the persist directory does not contain a valid store.
        """
        if not self.persist_directory.exists() or not any(self.persist_directory.iterdir()):
            raise FileNotFoundError("Persist directory does not exist or is empty.")

        embedder = Embedder(
            model_name=self.embedding_model_name,
            device=self.embedding_device
        ).get_langchain_embeddings()

        self._vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embedder,
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
            raise RuntimeError("Vector store has not been created or loaded.")
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
            return 0
        return self._vector_store._collection.count()
