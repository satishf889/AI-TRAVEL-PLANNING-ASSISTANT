"""Semantic retriever for the RAG pipeline.

Retrieves relevant knowledge base chunks for a given user query,
returning results with their source metadata for citation.

Requirements satisfied: RAG Requirements 5, 6, 7 (retrieve, generate, cite).
"""

from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """A single retrieval result with content and source attribution."""

    content: str
    source_title: str
    source_url: str
    relevance_score: float
    chunk_index: int


class KnowledgeRetriever:
    """Performs semantic retrieval over the Singapore travel knowledge base.

    Wraps the ChromaDB vector store to provide a clean retrieval interface
    that returns results with source metadata for citation generation.
    """

    def __init__(self, vector_store_manager: object, top_k: int = 5) -> None:
        """Initialise the retriever with a loaded vector store manager.

        Args:
            vector_store_manager: A loaded VectorStoreManager instance.
            top_k: Number of top results to return per query.
        """
        self.vector_store_manager = vector_store_manager
        self.top_k = top_k

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Retrieve the most relevant knowledge base chunks for a query.

        Args:
            query: The user's natural language question.

        Returns:
            List of RetrievalResult instances ordered by relevance (highest first).

        Raises:
            RuntimeError: If the vector store has not been initialised.
            ValueError: If query is empty.
        """
        raise NotImplementedError("Implement in TDD cycle")

    def retrieve_with_fallback_message(self, query: str) -> tuple[list[RetrievalResult], str | None]:
        """Retrieve relevant chunks and provide a fallback message if none found.

        If no relevant chunks are found (or all scores are below threshold),
        returns an empty list and a fallback message instructing the LLM
        not to fabricate information.

        Args:
            query: The user's natural language question.

        Returns:
            Tuple of (results, fallback_message). fallback_message is None
            when relevant results are found, and a string when no results found.
        """
        raise NotImplementedError("Implement in TDD cycle")

    def format_context(self, results: list[RetrievalResult]) -> str:
        """Format retrieval results into a context string for the LLM prompt.

        Args:
            results: List of RetrievalResult instances.

        Returns:
            Formatted string with content and source citations.
        """
        raise NotImplementedError("Implement in TDD cycle")
