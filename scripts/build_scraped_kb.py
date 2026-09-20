"""Standalone pipeline script to scrape travel web data and build the knowledge base.

Execution steps:
1. Scrapes web page content using BeautifulSoup4 (WebScraper).
2. Cleans and formats content into Markdown using Azure OpenAI gpt-5-mini (KBGenerator).
3. Saves formatted Markdown files to knowledge_base/.
4. Loads, chunks, embeds, and persists documents into ChromaDB (VectorStoreManager).

Usage:
    python scripts/build_scraped_kb.py

    python scripts/build_scraped_kb.py --urls https://en.wikivoyage.org/wiki/Singapore
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure repository root is on sys.path for features module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.config.settings import settings
from features.rag.chunker import DocumentChunker
from features.rag.document_loader import DocumentLoader
from features.rag.vector_store import VectorStoreManager
from features.scraper.kb_generator import KBGenerator
from features.scraper.web_scraper import WebScraper

logger = logging.getLogger(__name__)

DEFAULT_SEED_URLS: list[str] = [
    "https://en.wikivoyage.org/wiki/Singapore",
    "https://en.wikivoyage.org/wiki/Singapore/Sentosa",
    "https://en.wikivoyage.org/wiki/Singapore/Marina_Bay",
]


def run_pipeline(
    urls: list[str] | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Execute end-to-end web scraping, LLM formatting, and ChromaDB vector store ingestion.

    Args:
        urls: List of target URLs to scrape. Defaults to DEFAULT_SEED_URLS.
        output_dir: Directory where formatted markdown files are stored.
                    Defaults to settings.knowledge_base_directory.

    Returns:
        List of saved document Path objects.
    """
    target_urls = urls or DEFAULT_SEED_URLS
    target_output = output_dir or settings.knowledge_base_directory

    logger.info("Starting standalone Scrape & Build KB Pipeline...")
    logger.info(f"Target output directory: {target_output}")
    logger.info(f"ChromaDB persist directory: {settings.chroma_persist_directory}")

    scraper = WebScraper()
    generator = KBGenerator(output_dir=target_output)

    # Step 1 & 2 & 3: Scrape, format via gpt-5-mini, and save to knowledge_base/
    import time

    saved_files: list[Path] = []
    for idx, url in enumerate(target_urls):
        if idx > 0:
            time.sleep(2.0)
        logger.info(f"Fetching and parsing: {url}")
        doc = scraper.scrape_url(url)
        filepath = generator.process_and_save(doc)
        saved_files.append(filepath)

    logger.info(f"Successfully created {len(saved_files)} Markdown document(s).")

    # Step 4: Chunk, embed, and ingest into ChromaDB vector store
    logger.info("Ingesting knowledge base documents into ChromaDB vector store...")
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

    logger.info(f"Stored {len(chunks)} embedded chunks in ChromaDB vector store.")
    logger.info("Pipeline execution completed successfully! ✅")

    return saved_files


def main() -> None:
    """CLI entrypoint for standalone pipeline."""
    parser = argparse.ArgumentParser(
        description="Standalone pipeline to scrape travel data and build ChromaDB knowledge base."
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        help="Custom list of URLs to scrape.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save generated Markdown files.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    saved = run_pipeline(urls=args.urls, output_dir=args.output_dir)
    print("\n✅ Standalone Scrape & Ingest Pipeline completed!")
    print(f"Generated {len(saved)} Markdown file(s):")
    for file_path in saved:
        print(f"  - {file_path}")


if __name__ == "__main__":
    main()
