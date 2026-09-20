"""CLI tool and entrypoint for the travel web scraper and KB generator module."""

import argparse
import logging
from pathlib import Path

from features.config.settings import settings
from features.rag.chunker import DocumentChunker
from features.rag.document_loader import DocumentLoader
from features.rag.vector_store import VectorStoreManager
from features.scraper.kb_generator import KBGenerator
from features.scraper.web_scraper import WebScraper

logger = logging.getLogger(__name__)

DEFAULT_SEED_URLS = [
    "https://en.wikivoyage.org/wiki/Singapore",
    "https://en.wikivoyage.org/wiki/Singapore/Sentosa",
    "https://en.wikivoyage.org/wiki/Singapore/Marina_Bay",
]


def run_scraper(
    urls: list[str] | None = None,
    output_dir: Path | None = None,
    index_vector_store: bool = False,
) -> list[Path]:
    """Run the scraping, LLM formatting, and optional vector indexing workflow.

    Args:
        urls: List of URLs to scrape. Defaults to DEFAULT_SEED_URLS.
        output_dir: Output directory for Markdown KB files.
        index_vector_store: Whether to trigger ChromaDB vector store re-indexing.

    Returns:
        List of paths to generated Markdown files.
    """
    target_urls = urls or DEFAULT_SEED_URLS
    target_output = output_dir or settings.knowledge_base_directory

    scraper = WebScraper()
    generator = KBGenerator(output_dir=target_output)

    saved_files: list[Path] = []
    for url in target_urls:
        logger.info(f"Processing URL: {url}")
        doc = scraper.scrape_url(url)
        filepath = generator.process_and_save(doc)
        saved_files.append(filepath)

    if index_vector_store and saved_files:
        logger.info("Re-indexing knowledge base documents into ChromaDB vector store...")
        loader = DocumentLoader(knowledge_base_dir=target_output)
        documents = loader.load_all()
        chunker = DocumentChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = chunker.chunk_documents(documents)
        vector_store = VectorStoreManager(
            persist_directory=settings.chroma_persist_directory,
            collection_name=settings.chroma_collection_name,
            embedding_model_name=settings.embedding_model_name,
            embedding_device=settings.embedding_device,
        )
        vector_store.create_from_chunks(chunks)
        logger.info(f"Successfully indexed {len(chunks)} chunks into vector store.")

    return saved_files


def main() -> None:
    """CLI parser entrypoint."""
    parser = argparse.ArgumentParser(
        description="Scrape travel data from web sources and build Markdown KB documents."
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        help="Target URLs to scrape. If not provided, default seed URLs are used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save generated markdown files.",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Automatically trigger ChromaDB vector store ingestion after scraping.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    saved = run_scraper(
        urls=args.urls,
        output_dir=args.output_dir,
        index_vector_store=args.index,
    )
    print(f"\n✅ Scraped {len(saved)} documents successfully:")
    for path in saved:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
