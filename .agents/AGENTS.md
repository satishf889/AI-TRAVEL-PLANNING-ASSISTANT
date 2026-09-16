# AI Travel Planning Assistant — Agent Rules

These rules apply to **all agents** working on this project. They are binding and must be followed strictly.

---

## 1. Requirements Traceability (MANDATORY)

- Every feature, function, and module **must trace back** to a requirement in [`requirement-docs/requirements.md`](../requirement-docs/requirements.md).
- Before implementing anything, confirm which requirement(s) it satisfies.
- Do **not** implement features not listed in `requirements.md` without explicit user approval.

---

## 2. Confirm Before Modifying (MANDATORY)

- **Always ask for confirmation** before modifying any existing file.
- Exception: creating **new** files does not require confirmation.
- When modifying, clearly state:
  - What file is being changed
  - What lines/sections are affected
  - Why the change is needed
  - What requirement it satisfies

---

## 3. Test-Driven Development (TDD) — MANDATORY

All development follows **Red → Green → Refactor**:

1. **Write the test first** — create the test in `tests/test_<module>/` before any implementation
2. **Run the test** — confirm it fails (Red)
3. **Write minimum implementation** to make it pass (Green)
4. **Refactor** — clean up without breaking tests

### TDD Rules
- Never merge untested code
- Every public function/method must have at least one unit test
- Mocking is required for all external calls: LLM, MCP tools, embedding APIs
- Run `pytest tests/ -v --cov=features` before any PR/commit

---

## 4. Module Boundaries (MANDATORY)

Each feature module under `features/` is **self-contained**:

| Module | Owns | Must NOT |
|--------|------|----------|
| `features/rag/` | Document loading, chunking, embedding, vector store, retrieval | Call MCP tools or LLM directly |
| `features/mcp/` | MCP client, weather tool, currency tool | Access vector store or KB directly |
| `features/orchestrator/` | Agent logic, prompts, context | Directly call external APIs |
| `features/ui/` | Streamlit UI, session state | Contain business logic |
| `features/config/` | Settings, env vars | Contain any logic |

Cross-module communication happens **only through the orchestrator**.

---

## 5. No Hallucination / Knowledge Constraints

- The agent (application) must **never fabricate** destination facts
- If the KB does not contain information, the response must say so explicitly
- MCP tools must **never** be used to answer questions the KB already covers
- This rule applies to both the AI application AND the coding agent

---

## 6. Source Attribution (MANDATORY)

- Every RAG response must include the **source title and URL** from document metadata
- Every MCP response must be clearly labeled as coming from an MCP tool
- Combined responses must clearly distinguish KB facts, MCP data, and LLM suggestions

---

## 7. Configuration via Environment Variables

- All secrets (API keys), model names, and tunable parameters go in `.env` (gitignored)
- `.env.example` must always be updated when adding a new variable
- Never hardcode API keys, model names, or endpoints in source code
- Use `features/config/settings.py` (pydantic-settings) for all settings access

---

## 8. Docker-First Local Development

- The app runs locally via `docker-compose up`
- All dependencies must be installable inside the container
- `chroma_db/` (vector store) is mounted as a volume — not copied into image
- The Dockerfile must be minimal and reproducible

---

## 9. Python Code Standards

- Python **3.11+** minimum
- Type hints are **required** on all function signatures
- Docstrings are **required** on all public functions and classes
- Linting: `ruff check .` must pass with zero errors
- Type checking: `mypy features/` must pass
- Line length: 100 characters max

---

## 10. File Naming Conventions

| Item | Convention |
|------|-----------|
| Python files | `snake_case.py` |
| Test files | `test_<module_name>.py` |
| Classes | `PascalCase` |
| Functions/variables | `snake_case` |
| Constants | `UPPER_SNAKE_CASE` |
| Environment variables | `UPPER_SNAKE_CASE` |

---

## 11. Git Commit Message Format

```
<type>(<scope>): <short description>

<body if needed>
```

Types: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`
Scopes: `rag`, `mcp`, `orchestrator`, `ui`, `config`, `docker`, `tests`

Examples:
- `feat(rag): add document chunker with overlap`
- `test(mcp): add weather tool unit tests with mock`
- `fix(orchestrator): handle missing KB content gracefully`

---

## 12. Acceptance Criteria Checklist

Before marking any module as complete, verify:

- [ ] All requirements for the module are implemented
- [ ] All tests pass (`pytest tests/test_<module>/ -v`)
- [ ] Code coverage ≥ 80% for the module
- [ ] `ruff check` passes
- [ ] `mypy` passes
- [ ] `.env.example` updated if new env vars added
- [ ] Docstrings present on all public APIs
- [ ] Module does not cross module boundaries
