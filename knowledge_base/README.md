# Knowledge Base Directory

This directory contains the Markdown knowledge documents used by the RAG (Retrieval-Augmented Generation) pipeline for the AI Travel Planning Assistant.

## 📄 Existing Knowledge Sources

The knowledge base currently contains **3 primary scraped travel sources** from Wikivoyage:

1. [`singapore.md`](file:///Users/satishfulwani/Documents/Github/AI-TRAVEL-PLANNING-ASSISTANT/knowledge_base/singapore.md) — Main Singapore travel guide (arrival, districts, culture, dining, safety, transit).
2. [`singaporesentosa.md`](file:///Users/satishfulwani/Documents/Github/AI-TRAVEL-PLANNING-ASSISTANT/knowledge_base/singaporesentosa.md) — Sentosa Island guide (resorts, attractions, beaches, entertainment).
3. [`singaporemarina_bay.md`](file:///Users/satishfulwani/Documents/Github/AI-TRAVEL-PLANNING-ASSISTANT/knowledge_base/singaporemarina_bay.md) — Marina Bay district guide (Marina Bay Sands, Gardens by the Bay, Singapore Flyer).

Additionally, supplementary travel guides are included:
- `visit_singapore_essentials.md`
- `visit_singapore_itineraries.md`
- `visit_singapore_things_to_do.md`
- `wikivoyage_singapore.md`

## 🔄 Generating More Knowledge Sources

For details on scraping new web pages or re-indexing vector embeddings into ChromaDB, refer to the [Scraper Documentation](file:///Users/satishfulwani/Documents/Github/AI-TRAVEL-PLANNING-ASSISTANT/features/scraper/README.md).

Quick command to scrape custom URLs and re-index vector store:
```bash
python -m features.scraper.cli --urls "https://en.wikivoyage.org/wiki/Tokyo" --index
```
