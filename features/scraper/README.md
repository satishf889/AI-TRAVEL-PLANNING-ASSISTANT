# Scraper & Knowledge Base Generator Module

This module (`features/scraper/`) provides automated tools to scrape travel web sources (e.g., Wikivoyage, official tourism portals) and format raw scraped web data into structured Markdown knowledge base documents with YAML frontmatter source attribution.

---

## 📌 Scraped Knowledge Sources

Currently, **3 seed knowledge sources** have been scraped and processed into structured Markdown files in the `knowledge_base/` directory:

1. **Singapore Overview** (`knowledge_base/singapore.md`)
   - **Source URL:** `https://en.wikivoyage.org/wiki/Singapore`
   - Covers comprehensive arrival, district guides, cultural norms, dining, safety, and transit details.
2. **Singapore / Sentosa** (`knowledge_base/singaporesentosa.md`)
   - **Source URL:** `https://en.wikivoyage.org/wiki/Singapore/Sentosa`
   - Covers resort island attractions, beaches, Universal Studios, cable car access, and entertainment options.
3. **Singapore / Marina Bay** (`knowledge_base/singaporemarina_bay.md`)
   - **Source URL:** `https://en.wikivoyage.org/wiki/Singapore/Marina_Bay`
   - Covers iconic landmarks like Marina Bay Sands, Gardens by the Bay, Singapore Flyer, dining, and promenade walks.

---

## 🛠 How It Works (Architecture & Pipeline)

The scraper module consists of three main sub-components:

```
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│      WebScraper        │ ───► │      KBGenerator       │ ───► │      ChromaDB RAG      │
│  (MediaWiki API/BS4)   │      │ (LLM Markdown Struct)  │      │ (VectorStore Indexing) │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
```

1. **`web_scraper.py` (`WebScraper`)**:
   - Detects Wikivoyage URLs and routes requests via the official **MediaWiki REST API** (`https://en.wikivoyage.org/w/api.php`) to avoid aggressive HTML parsing issues.
   - Includes automatic HTTP 429 rate limit retry handling and customized `User-Agent` headers.
   - Uses **BeautifulSoup4** as fallback for non-Wiki web content extraction.

2. **`kb_generator.py` (`KBGenerator`)**:
   - Uses Azure OpenAI (`gpt-5-mini` / `AzureChatOpenAI`) to reformat raw text into clean, structured Markdown.
   - Mandates YAML frontmatter headers at the top of every generated file for source attribution:
     ```yaml
     ---
     source_title: Singapore/Sentosa
     source_url: https://en.wikivoyage.org/wiki/Singapore/Sentosa
     ---
     ```
   - Includes fallback logic to raw Markdown structure if the LLM API is unreachable.

3. **CLI Orchestrator (`cli.py`)**:
   - Offers CLI entry points to scrape default seed URLs or custom target URLs.
   - Optionally triggers automatic ChromaDB vector store chunking and re-indexing via the RAG pipeline when the `--index` flag is provided.

---

## 🚀 Running Scraper Commands

If additional travel destinations or knowledge sources are required, or if existing sources need to be re-scraped, you can run the scraper CLI module.

### 1. Run Default Scraping (Scrapes 3 Default Seed URLs)
To re-scrape the default 3 Singapore Wikivoyage pages into `knowledge_base/`:

```bash
python -m features.scraper.cli
```

### 2. Scrape Custom URLs
To scrape any specific travel pages (Wikivoyage or web URLs), pass `--urls` followed by one or more web addresses:

```bash
python -m features.scraper.cli --urls \
  "https://en.wikivoyage.org/wiki/Tokyo" \
  "https://en.wikivoyage.org/wiki/Kyoto"
```

### 3. Scrape and Automatically Re-Index ChromaDB Vector Store
To scrape new destinations AND immediately update the ChromaDB RAG vector embeddings store in one step, add the `--index` flag:

```bash
python -m features.scraper.cli --urls "https://en.wikivoyage.org/wiki/Tokyo" --index
```

### 4. Custom Output Directory
To output generated Markdown files to a custom folder instead of default `knowledge_base/`:

```bash
python -m features.scraper.cli --output-dir ./custom_kb --urls "https://en.wikivoyage.org/wiki/Bali"
```

### 5. Running via Docker Container
If running the project inside Docker containers:

```bash
docker-compose exec app python -m features.scraper.cli --index
```

---

## ⚙️ Configuration & Requirements

- **Environment Variables**: Make sure your `.env` file contains valid Azure OpenAI configuration (for LLM formatting) and ChromaDB settings:
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_DEPLOYMENT_NAME`
  - `AZURE_OPENAI_API_VERSION`
- **Rate Limits & Etiquette**: Scrapes respect Wikivoyage guidelines by routing via API with exponential backoff on HTTP 429 status codes.
