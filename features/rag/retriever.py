from __future__ import annotations

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
        if not self.vector_store_manager.is_initialized():
            raise RuntimeError("Vector store is not initialized")
        if not query:
            raise ValueError("Query cannot be empty")

        docs_and_scores = self.vector_store_manager._vector_store.similarity_search_with_score(query, k=self.top_k)

        results = []
        for doc, score in docs_and_scores:
            results.append(
                RetrievalResult(
                    content=doc.page_content,
                    source_title=doc.metadata.get("source_title", "Unknown"),
                    source_url=doc.metadata.get("source_url", ""),
                    relevance_score=float(score),
                    chunk_index=doc.metadata.get("chunk_index", 0),
                )
            )
        return results

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
        results = self.retrieve(query)
        if not results:
            from features.orchestrator.prompt_templates import FALLBACK_NO_KB_CONTENT
            return [], FALLBACK_NO_KB_CONTENT
        return results, None

    def format_context(self, results: list[RetrievalResult]) -> str:
        """Format retrieval results into a context string for the LLM prompt.

        Args:
            results: List of RetrievalResult instances.

        Returns:
            Formatted string with content and source citations.
        """
        parts = []
        for res in results:
            parts.append(f"Source: {res.source_title}\nURL: {res.source_url}\n{res.content}")
        return "\n\n".join(parts)

    def invoke(self, query: str) -> list[object]:
        """LangChain compatible invoke method returning Document objects.

        Args:
            query: The user's natural language question.

        Returns:
            List of LangChain Document instances with full source metadata.
        """
        from langchain_core.documents import Document

        results = self.retrieve(query)
        return [
            Document(
                page_content=res.content,
                metadata={
                    "title": res.source_title,
                    "source": res.source_url,
                    "source_title": res.source_title,
                    "source_url": res.source_url,
                    "relevance_score": res.relevance_score,
                    "chunk_index": res.chunk_index,
                },
            )
            for res in results
        ]

