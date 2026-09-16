# AI Travel Planning Assistant — Full Requirements

> **Source:** `AI_Travel_Planning_Assistant_Assignment.pdf`
> **Last extracted:** 2026-09-16

---

## Overview

Build a **context-aware travel assistant** that combines:
1. A **document-based knowledge base** (destination facts via RAG)
2. **Current information** retrieved through **MCP tools** (weather, currency)

---

## 1. Background

Travel planning requires two distinct types of information:

| Type | Examples | Retrieval Method |
|------|----------|-----------------|
| Destination knowledge | Attractions, neighbourhoods, transport, cultural guidance, itineraries | RAG (pre-collected documents) |
| Current information | Weather forecasts, currency exchange rates | MCP tools (live external services) |

---

## 2. Problem Statement

Build an AI Travel Planning Assistant that:
- Helps users plan a trip to a **selected destination** (Singapore recommended)
- Uses a **travel knowledge base** for destination recommendations
- Uses **MCP tools** to retrieve current information
- **Combines both sources** when a question requires both destination knowledge and live data

**Primary Example Query:**
> "Plan a three-day trip to Singapore and adjust the activities based on the weather forecast."

---

## 3. Application Scope

### In scope
- One destination (Singapore recommended)
- Destination knowledge Q&A
- Current weather retrieval
- Currency conversion
- Combined RAG + MCP responses
- Multi-turn conversation with context retention

### Out of scope
- Flight or hotel booking, payment processing, route navigation, travel reservations

---

## 4. Core Features

### 4.1 Destination Knowledge Assistant (RAG)

**Knowledge base must cover:**
- Major attractions and neighbourhoods
- Local transportation guidance
- Cultural and practical travel tips
- Food and local experiences
- Sample itineraries
- Indoor and outdoor activity suggestions

**RAG Requirements:**
1. Load travel content from public documents or web pages
2. Divide content into meaningful chunks
3. Generate embeddings for the chunks
4. Store embeddings in a vector store
5. Retrieve relevant chunks for each user question
6. Generate answers grounded in the retrieved content
7. Display the source title or source link used for the answer

> **Constraint:** Must NOT invent destination facts if KB is insufficient — state clearly.

---

### 4.2 Current Travel Information (MCP)

#### MCP Tool 1 — Weather Information
Retrieve current conditions or forecast for the destination.

#### MCP Tool 2 — Currency Conversion
Convert amounts between currencies using live service data.

**MCP Requirements:**
8. Connect to at least two MCP tools
9. Make the tools available to the AI application
10. Select the appropriate tool based on the user request
11. Pass the required input to the selected tool
12. Use the returned information in the final response
13. Clearly indicate when current information came from an MCP tool
14. Handle unavailable tools or failed results without fabricating an answer

> **Constraint:** MCP must NOT answer questions covered by the knowledge base.

---

### 4.3 Combined RAG + MCP Response

**Required scenario (mandatory):**
> "Create a three-day Singapore itinerary for next week and adjust it according to the weather forecast."

The response must distinguish:
- Knowledge-base facts
- MCP-provided current information
- LLM-generated recommendations

---

## 5. Prompt Engineering Requirements

Prompts must instruct the model to:
- Use KB content for destination facts
- Use MCP responses for current information
- Avoid unsupported claims
- State when information is unavailable
- Produce structured travel recommendations
- Include source references
- Distinguish facts from AI suggestions
- Preserve user preferences across turns

---

## 6. Suggested Knowledge Base Sources (Singapore)

Use at least **3 public resources**:
1. Wikivoyage Singapore Travel Guide
2. Visit Singapore: Essential Travel Information
3. Visit Singapore: Sample Itineraries
4. Visit Singapore: Things to Do

Retain original source title and URL as metadata for citations.

---

## 7. Technology Requirements

| Component | Requirement |
|-----------|-------------|
| Orchestration | LangChain |
| Embedding model | Candidate's choice |
| Vector store | FAISS or Chroma |
| LLM | Candidate's choice |
| MCP | MCP-compatible client + MCP tools |
| UI | Any framework |

---

## 8. Minimum Acceptance Criteria

- [ ] Knowledge base from at least 3 travel resources
- [ ] Embedding-based semantic retrieval
- [ ] Grounded answers with source references
- [ ] Weather information through MCP tool
- [ ] Currency conversion through MCP tool
- [ ] At least one combined RAG + MCP response
- [ ] Multi-turn conversation with retained context
- [ ] Appropriate tool selection based on user intent
- [ ] Clear handling of missing knowledge and tool failures
- [ ] Simple, usable interface

---

## 9. Deliverables

1. Source code in a Git repository
2. A working application
3. Knowledge-base documents or clear instructions for obtaining them
4. README covering: architecture, KB sources, RAG workflow, MCP tools, prompt & context strategy, setup
5. Sample questions and application responses
6. Short demo showing RAG, MCP, combined response, and conversational context
