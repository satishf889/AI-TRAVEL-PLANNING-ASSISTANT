# AI Travel Planning Assistant — Implementation Plan

## Overview

Build a context-aware AI travel assistant for Singapore that combines:
- **RAG** (Retrieval-Augmented Generation) using a pre-built knowledge base
- **MCP tools** for real-time weather and currency data
- **Multi-turn conversation** with retained context
- **Streamlit UI** with Docker for local deployment

---

## Open Questions

> [!IMPORTANT]
> **LLM Choice:** The spec allows any LLM. Recommended: **OpenAI GPT-4o** (best LangChain integration) or **Anthropic Claude 3.5 Sonnet**. Which do you prefer, or should we make it configurable via `.env`?

> [!IMPORTANT]
> **Embedding Model:** Recommended: `text-embedding-3-small` (OpenAI) or `sentence-transformers/all-MiniLM-L6-v2` (local/free). Use OpenAI embeddings or local HuggingFace embeddings?

> [!IMPORTANT]
> **MCP Weather API:** Which weather API to use for the MCP Weather Tool? Options: OpenWeatherMap (free tier), WeatherAPI.com, or Open-Meteo (completely free, no key needed). Recommend **Open-Meteo** to avoid API key friction.

> [!IMPORTANT]
> **MCP Currency API:** Which currency API? Options: ExchangeRate-API (free tier), Open Exchange Rates, or Frankfurter (free, no key). Recommend **Frankfurter** (no API key needed).

---

## Proposed Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Core Language | **Python 3.11+** | Specified |
| UI Framework | **Streamlit** | Specified |
| Containerization | **Docker + Docker Compose** | Specified |
| AI Orchestration | **LangChain** | Required by spec |
| LLM | **Google Gemini Pro** (via `langchain-google-genai`) | User's choice |
| Embeddings | **HuggingFace `all-MiniLM-L6-v2`** (local, free) | No API key needed |
| Vector Store | **ChromaDB** | Persistent, easy Docker setup |
| MCP Client | **langchain-mcp-adapters** | Official LangChain MCP integration |
| Testing | **pytest + pytest-cov** | TDD framework |
| Mocking | **unittest.mock + pytest-mock** | For MCP/LLM mocking in tests |
| Linting | **ruff + mypy** | Code quality |
| Config | **python-dotenv** | Env var management |
| HTTP | **httpx** | Async HTTP for MCP tools |

---

## File Structure

```
AI-TRAVEL-PLANNING-ASSISTANT/
├── .agents/                          # Agent rules & workspace config
│   └── AGENTS.md                     # Project-specific agent rules
├── requirement-docs/
│   ├── AI_Travel_Planning_Assistant_Assignment.pdf
│   └── requirements.md               # Human-readable extracted requirements
├── knowledge_base/                   # Raw KB source documents
│   ├── wikivoyage_singapore.md
│   ├── visit_singapore_essentials.md
│   ├── visit_singapore_itineraries.md
│   └── visit_singapore_things_to_do.md
├── features/                         # Feature modules (one per capability)
│   ├── __init__.py
│   ├── rag/                          # RAG pipeline module
│   │   ├── __init__.py
│   │   ├── document_loader.py        # Load & parse KB documents
│   │   ├── chunker.py                # Split docs into chunks
│   │   ├── embedder.py               # Generate embeddings
│   │   ├── vector_store.py           # ChromaDB store management
│   │   └── retriever.py              # Semantic retrieval logic
│   ├── mcp/                          # MCP tools module
│   │   ├── __init__.py
│   │   ├── mcp_client.py             # MCP client setup & tool registry
│   │   ├── weather_tool.py           # Weather MCP tool (Open-Meteo)
│   │   └── currency_tool.py          # Currency MCP tool (Frankfurter)
│   ├── orchestrator/                 # LangChain agent orchestration
│   │   ├── __init__.py
│   │   ├── agent.py                  # Main LangChain agent with RAG+MCP
│   │   ├── prompt_templates.py       # All prompt templates
│   │   └── context_manager.py        # Multi-turn conversation context
│   ├── ui/                           # Streamlit UI module
│   │   ├── __init__.py
│   │   ├── app.py                    # Main Streamlit app entry point
│   │   ├── components.py             # Reusable UI components
│   │   └── session_state.py          # Streamlit session state management
│   └── config/                       # Configuration module
│       ├── __init__.py
│       └── settings.py               # App settings via pydantic-settings
├── tests/                            # TDD test suite
│   ├── __init__.py
│   ├── conftest.py                   # Shared fixtures & mocks
│   ├── test_rag/
│   │   ├── __init__.py
│   │   ├── test_document_loader.py
│   │   ├── test_chunker.py
│   │   ├── test_embedder.py
│   │   ├── test_vector_store.py
│   │   └── test_retriever.py
│   ├── test_mcp/
│   │   ├── __init__.py
│   │   ├── test_mcp_client.py
│   │   ├── test_weather_tool.py
│   │   └── test_currency_tool.py
│   ├── test_orchestrator/
│   │   ├── __init__.py
│   │   ├── test_agent.py
│   │   ├── test_prompt_templates.py
│   │   └── test_context_manager.py
│   └── test_ui/
│       ├── __init__.py
│       └── test_session_state.py
├── scripts/                          # Utility scripts
│   ├── ingest_knowledge_base.py      # One-time KB ingestion script
│   └── test_mcp_tools.py             # Manual MCP tool verification
├── chroma_db/                        # Persisted vector store (gitignored)
├── .env.example                      # Example environment variables
├── .gitignore
├── docker-compose.yml                # Docker Compose for local dev
├── Dockerfile                        # Streamlit app container
├── pyproject.toml                    # Python project config + dependencies
├── pytest.ini                        # Pytest configuration
├── ruff.toml                         # Ruff linter config
└── README.md                         # Project documentation
```

---

## Module Breakdown (Feature-by-Feature)

### Module 1: `features/config/` — Configuration
**What it does:** Centralizes all settings via environment variables.
**Key settings:** API keys, model names, vector store path, chunk size, retrieval k.

---

### Module 2: `features/rag/` — RAG Pipeline
**What it does:** Ingests knowledge base documents and enables semantic retrieval.

| File | Responsibility |
|------|---------------|
| `document_loader.py` | Load MD/PDF/HTML docs, attach source metadata (title + URL) |
| `chunker.py` | Split documents into overlapping chunks (RecursiveCharacterTextSplitter) |
| `embedder.py` | Generate embeddings using configured embedding model |
| `vector_store.py` | Create/load ChromaDB collection, upsert/query embeddings |
| `retriever.py` | Retrieve top-k relevant chunks for a query, return with source metadata |

---

### Module 3: `features/mcp/` — MCP Tools
**What it does:** Connects to external services via MCP protocol.

| File | Responsibility |
|------|---------------|
| `mcp_client.py` | Initialize MCP client, register all tools, expose tool list to agent |
| `weather_tool.py` | Call Open-Meteo API, return structured forecast data |
| `currency_tool.py` | Call Frankfurter API, return converted amount with rate info |

---

### Module 4: `features/orchestrator/` — LangChain Agent
**What it does:** Routes questions to RAG/MCP/both and synthesizes responses.

| File | Responsibility |
|------|---------------|
| `agent.py` | LangChain ReAct agent with RAG retriever + MCP tools bound |
| `prompt_templates.py` | System prompt + user prompt templates with source-attribution rules |
| `context_manager.py` | Maintain conversation history, inject relevant context per turn |

---

### Module 5: `features/ui/` — Streamlit UI
**What it does:** Provides the user interface.

| File | Responsibility |
|------|---------------|
| `app.py` | Main Streamlit app: layout, routing, orchestrator calls |
| `components.py` | Chat message component, source citation display, weather widget |
| `session_state.py` | Initialize and manage Streamlit session state (history, settings) |

---

## Agent Rules (AGENTS.md)

The `.agents/AGENTS.md` file will enforce:
1. **Always confirm** before modifying existing files
2. **TDD first** — write tests before implementation
3. **Module boundaries** — each feature module is self-contained
4. **No hallucination** — KB constraints must be respected
5. **Follow requirements.md** — all features traced to spec

---

## TDD Approach

Every module follows **Red → Green → Refactor**:

1. **Write test first** (in `tests/test_<module>/`)
2. **Run test** — confirm it fails
3. **Implement minimum code** to pass the test
4. **Refactor** without breaking tests

**Test categories per module:**
- **Unit tests:** All pure functions (chunking, prompt building, tool parsing)
- **Integration tests:** RAG retrieval pipeline, MCP tool calls (mocked)
- **Contract tests:** MCP tool response schemas
- **E2E tests:** Full agent response for key scenarios (mocked LLM + MCP)

---

## Verification Plan

### Automated Tests
```bash
pytest tests/ -v --cov=features --cov-report=term-missing
```

### Manual Verification
1. Run `docker-compose up` locally
2. Open `http://localhost:8501`
3. Test the 10 minimum acceptance criteria queries
4. Verify source citations appear for RAG responses
5. Verify "MCP tool used" label appears for weather/currency responses
6. Verify combined RAG+MCP response for the required itinerary scenario

---

## Implementation Order (Phases)

| Phase | Modules | Description |
|-------|---------|-------------|
| 1 | `config` | Settings & env setup |
| 2 | `rag` | Full RAG pipeline + tests |
| 3 | `mcp` | Weather + currency tools + tests |
| 4 | `orchestrator` | Agent, prompts, context + tests |
| 5 | `ui` | Streamlit app + components |
| 6 | Docker | Dockerfile + docker-compose |
| 7 | Docs | README, sample Q&A |
