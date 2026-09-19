"""Knowledge Base Ingestion Script.

Run this once (or after updating KB documents) to:
1. Load all documents from knowledge_base/
2. Split into chunks
3. Generate embeddings
4. Store in ChromaDB

Usage:
    python scripts/ingest_knowledge_base.py

Or via Docker:
    docker-compose run --rm ingest
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path so features/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.config.settings import settings
from features.rag.chunker import DocumentChunker
from features.rag.document_loader import DocumentLoader
from features.rag.vector_store import VectorStoreManager

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the knowledge base ingestion pipeline."""
    logger.info("Starting knowledge base ingestion...")
    logger.info(f"KB directory: {settings.knowledge_base_directory}")
    logger.info(f"ChromaDB directory: {settings.chroma_persist_directory}")

    # Step 1: Load documents
    loader = DocumentLoader(knowledge_base_dir=settings.knowledge_base_directory)
    documents = loader.load_all()
    logger.info(f"Loaded {len(documents)} documents")
    for doc in documents:
        logger.info(f"  - {doc.source_title} ({doc.file_path.name})")

    # Step 2: Chunk documents
    chunker = DocumentChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = chunker.chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks (size={settings.chunk_size}, overlap={settings.chunk_overlap})")

    # Step 3 & 4: Embed and store in ChromaDB
    vector_store = VectorStoreManager(
        persist_directory=settings.chroma_persist_directory,
        collection_name=settings.chroma_collection_name,
        embedding_model_name=settings.embedding_model_name,
        embedding_device=settings.embedding_device,
    )
    vector_store.create_from_chunks(chunks)
    logger.info(f"Stored {vector_store.document_count()} vectors in ChromaDB")
    logger.info("Ingestion complete! ✅")


if __name__ == "__main__":
    main()
