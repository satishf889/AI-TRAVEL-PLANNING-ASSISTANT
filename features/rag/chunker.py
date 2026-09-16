"""Document chunker for the RAG pipeline.

Splits loaded documents into overlapping chunks suitable for embedding and retrieval.
Preserves source metadata on each chunk for citation purposes.

Requirements satisfied: RAG Requirement 2 (divide content into meaningful chunks).
"""

from dataclasses import dataclass

from features.rag.document_loader import KnowledgeDocument


@dataclass
class DocumentChunk:
    """A single chunk of a knowledge document, ready for embedding."""

    content: str
    source_title: str
    source_url: str
    chunk_index: int
    total_chunks: int


class DocumentChunker:
    """Splits KnowledgeDocuments into overlapping text chunks.

    Uses LangChain's RecursiveCharacterTextSplitter strategy to create
    semantically meaningful chunks that respect sentence and paragraph boundaries.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """Initialise the chunker with size and overlap settings.

        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of characters to overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: KnowledgeDocument) -> list[DocumentChunk]:
        """Split a single document into overlapping chunks.

        Each chunk inherits the source metadata (title and URL) from the document.

        Args:
            document: The loaded knowledge document to split.

        Returns:
            List of DocumentChunk instances with source metadata preserved.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        
        text_chunks = splitter.split_text(document.content)
        
        doc_chunks = []
        for i, text in enumerate(text_chunks):
            doc_chunks.append(
                DocumentChunk(
                    content=text,
                    source_title=document.source_title,
                    source_url=document.source_url,
                    chunk_index=i,
                    total_chunks=len(text_chunks)
                )
            )
        return doc_chunks

    def chunk_documents(self, documents: list[KnowledgeDocument]) -> list[DocumentChunk]:
        """Split multiple documents into chunks.

        Args:
            documents: List of KnowledgeDocument instances to process.

        Returns:
            Flat list of all DocumentChunk instances from all documents.
        """
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
