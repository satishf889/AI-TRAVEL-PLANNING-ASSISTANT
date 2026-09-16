# ✈️ AI Travel Planning Assistant

A context-aware AI assistant for planning trips to **Singapore**, combining a document-based knowledge base with real-time data via MCP tools.

> **Assignment:** Developer Assignment — AI Travel Planning Assistant
> **Destination:** Singapore
> **Stack:** Python 3.11 · LangChain · Google Gemini Pro · ChromaDB · Streamlit · Docker

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                  Streamlit UI (features/ui/)             │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│          TravelAgent Orchestrator (features/orchestrator/)│
│  ┌─────────────────┐     ┌───────────────────────────┐  │
│  │  Intent         │     │  Context Manager          │  │
│  │  Classifier     │     │  (Multi-turn history)     │  │
│  └────────┬────────┘     └───────────────────────────┘  │
│           │                                               │
│    ┌──────┴──────┐                                        │
│    ▼             ▼                                        │
│  RAG Path    MCP Path                                     │
│    │             │                                        │
│    ▼             ▼                                        │
│ ┌──────┐   ┌──────────────┐                              │
│ │Chroma│   │ Weather Tool │  → Open-Meteo API (free)    │
│ │  DB  │   │ Currency Tool│  → Frankfurter API (free)   │
│ └──────┘   └──────────────┘                              │
│                                                           │
│           Google Gemini Pro LLM                          │
└─────────────────────────────────────────────────────────┘
```

### Information Flow
| User Question Type | Sources Used | Attribution |
|-------------------|-------------|-------------|
| Destination facts (attractions, transport) | ChromaDB (RAG) | "📚 Source: [title](url)" |
| Weather forecast | Open-Meteo via MCP | "🌤️ Live data from Open-Meteo" |
| Currency conversion | Frankfurter via MCP | "💱 Live rate from Frankfurter" |
| Combined (itinerary + weather) | RAG + MCP + LLM | All three, clearly labelled |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| UI | Streamlit |
| AI Orchestration | LangChain |
| LLM | Google Gemini Pro (`gemini-1.5-pro`) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector Store | ChromaDB (persisted to disk) |
| MCP Weather | Open-Meteo API (free, no key) |
| MCP Currency | Frankfurter Exchange Rates API (free, no key) |
| Testing | pytest + pytest-cov + pytest-mock |
| Containerization | Docker + Docker Compose |

---

## Knowledge Base Sources (Singapore)

The KB is built from these public resources (see `knowledge_base/`):

| File | Source | Coverage |
|------|--------|----------|
| `wikivoyage_singapore.md` | [Wikivoyage Singapore](https://en.wikivoyage.org/wiki/Singapore) | Districts, attractions, transport, food, itineraries |
| `visit_singapore_essentials.md` | [Visit Singapore: Essentials](https://www.visitsingapore.com/travel-guide-tips/practical-info/) | Climate, language, currency, connectivity |
| `visit_singapore_itineraries.md` | [Visit Singapore: Itineraries](https://www.visitsingapore.com/see-do-singapore/itineraries/) | 1/3/5-day sample itineraries |
| `visit_singapore_things_to_do.md` | [Visit Singapore: Things To Do](https://www.visitsingapore.com/see-do-singapore/) | Attractions, activities, family, culture |

> **License:** Wikivoyage content is CC BY-SA 4.0. Visit Singapore content — review their reuse terms at visitsingapore.com.

---

## RAG Workflow

```
knowledge_base/*.md
       │
       ▼ DocumentLoader (features/rag/document_loader.py)
Load + extract source metadata (title, URL)
       │
       ▼ DocumentChunker (features/rag/chunker.py)
Split into 1000-char chunks with 200-char overlap
       │
       ▼ Embedder (features/rag/embedder.py)
Generate embeddings with all-MiniLM-L6-v2 (local)
       │
       ▼ VectorStoreManager (features/rag/vector_store.py)
Persist to ChromaDB
       │
       ▼ KnowledgeRetriever (features/rag/retriever.py)
Retrieve top-5 chunks per query → formatted context + citations
```

---

## MCP Tools

### Tool 1: Weather (Open-Meteo)
- **No API key required**
- Endpoint: `api.open-meteo.com/v1/forecast`
- Returns: Daily max/min temp, precipitation, wind speed, WMO weather codes
- Rainy-day detection → `indoor_recommended` flag for itinerary planning

### Tool 2: Currency (Frankfurter)
- **No API key required**
- Endpoint: `api.frankfurter.app/latest`
- Returns: Live exchange rate + converted amount

---

## Prompt Strategy

> 📄 Full prompt templates in `features/orchestrator/prompt_templates.py`

The prompts enforce these rules:
1. **KB content only for destination facts** — never hallucinated
2. **MCP data only for real-time information** — clearly labeled
3. **LLM generates suggestions** — labeled as "💡 AI Suggestion"
4. **Fallback** — if KB is empty, states clearly and suggests official sources
5. **Tool failure** — if MCP tool is down, states clearly instead of fabricating
6. **User preferences preserved** — budget, trip style, duration tracked across turns

---

## Project Structure

```
AI-TRAVEL-PLANNING-ASSISTANT/
├── .agents/AGENTS.md              # Agent rules (TDD, confirmation, module boundaries)
├── requirement-docs/
│   ├── AI_Travel_Planning_Assistant_Assignment.pdf
│   └── requirements.md            # Extracted requirements (agent-readable)
├── knowledge_base/                # Singapore KB documents (Markdown)
├── features/
│   ├── config/settings.py         # All settings via pydantic-settings
│   ├── rag/                       # RAG pipeline (loader → chunker → embedder → store → retriever)
│   ├── mcp/                       # MCP tools (weather, currency)
│   ├── orchestrator/              # LangChain agent + prompts + context manager
│   └── ui/                        # Streamlit app + components + session state
├── tests/                         # TDD test suite (mirrors features/ structure)
├── scripts/ingest_knowledge_base.py
├── .env.example                   # Environment variable template
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml                 # pytest + ruff + mypy config
```

---

## Setup Instructions

### Prerequisites
- Docker + Docker Compose
- Google API key (for Gemini Pro)

### 1. Clone and configure
```bash
git clone <repo-url>
cd AI-TRAVEL-PLANNING-ASSISTANT
cp .env.example .env
# Edit .env and set your GOOGLE_API_KEY
```

### 2. Populate knowledge base
```bash
# Add actual content to knowledge_base/*.md files
# (replacing the placeholder content)
```

### 3. Ingest knowledge base
```bash
docker-compose run --rm ingest
```

### 4. Run the app
```bash
docker-compose up app
# Open http://localhost:8501
```

### 5. Run tests
```bash
# Inside Docker
docker-compose run --rm app pytest tests/ -v --cov=features

# Or locally (with .venv activated)
.venv/bin/pytest tests/ -v --cov=features
```

---

## Minimum Acceptance Criteria Status

- [ ] Knowledge base from ≥ 3 travel resources
- [ ] Embedding-based semantic retrieval
- [ ] Grounded answers with source references
- [ ] Weather information via MCP tool
- [ ] Currency conversion via MCP tool
- [ ] Combined RAG + MCP response (weather-aware itinerary)
- [ ] Multi-turn conversation with context retention
- [ ] Appropriate tool selection based on user intent
- [ ] Clear handling of missing KB / tool failures
- [ ] Simple, usable Streamlit interface

---

## Sample Questions

### RAG (Knowledge Base)
- "What are the must-visit attractions in Singapore?"
- "Which neighbourhoods are best for cultural experiences?"
- "How do I get around Singapore using public transport?"
- "Suggest indoor activities for rainy days."
- "Create a three-day sightseeing itinerary."

### MCP Tools
- "What is the weather forecast for Singapore this week?"
- "Is it going to rain tomorrow in Singapore?"
- "Convert INR 50,000 to SGD."
- "How much is 200 SGD in USD?"

### Combined (RAG + MCP)
- "Create a 3-day Singapore itinerary for next week, adjusted for the weather."
- "I have a budget of INR 60,000. Convert it to SGD and suggest a 3-day trip."
- "Plan a family-friendly trip and check if indoor alternatives are needed based on the forecast."