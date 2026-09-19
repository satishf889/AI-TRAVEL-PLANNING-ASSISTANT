---
name: build-feature
description: >-
  Use this skill whenever the user asks to implement, add, or build a new
  feature in the AI Travel Planning Assistant. It enforces a strict TDD
  workflow: clarify requirements → plan → confirm → write tests → confirm →
  write implementation. The agent must NOT write any source code until the user
  has explicitly confirmed the plan AND the proposed tests.
---

# Build-Feature Skill

This skill governs how new features are planned, tested, and implemented in
this project. Follow **every phase in order** — skipping or merging phases is
not allowed.

---

## Phase 0 — Understand Before Anything Else

1. Read the feature request carefully.
2. Cross-reference the request against
   [`requirement-docs/requirements.md`](../../../requirement-docs/requirements.md)
   to confirm the feature is listed.
   - If it is **not** listed → tell the user, and **stop**. Do not proceed
     without explicit approval to add the requirement first.
3. Identify which module(s) in `features/` own this feature (see the module
   boundary table in [AGENTS.md](../../AGENTS.md)).

---

## Phase 1 — Ask Clarifying Questions (No Code, No Plan Yet)

Before writing anything, surface every ambiguity. Ask the user:

- Which requirement ID / section does this feature map to?
- Which module boundary does it live in (`rag`, `mcp`, `orchestrator`, `ui`,
  `config`)?
- Are there external APIs or MCP tools involved?
- What are the expected inputs and outputs?
- Are there edge cases or failure modes to handle?
- Does this touch existing files? (If yes, identify them — changes require
  confirmation per **Rule 2** in AGENTS.md.)

> **Rule**: Do NOT move to Phase 2 until all questions are answered and you
> have a clear understanding of the feature scope.

---

## Phase 2 — Create the Implementation Plan

After questions are answered, produce a written plan using the artifact system
(`implementation_plan.md`). The plan must include:

1. **Summary** — one-paragraph description of what is being built and why.
2. **Requirement Traceability** — which requirement(s) this satisfies.
3. **Module Ownership** — which `features/` module owns this work.
4. **Files to Create** (`[NEW]`) and **Files to Modify** (`[MODIFY]`).
   - For every modified file, state exactly what sections/lines change and why.
5. **API / Function Signatures** — the public interfaces you plan to implement.
6. **Error Handling Strategy** — how failures are surfaced.
7. **Open Questions** — anything still unresolved.

> **Rule**: Do NOT write any implementation code. The plan is prose + file
> lists + signatures only.

### Get Plan Confirmation

Present the plan to the user and wait for **explicit approval** before
continuing. If the user requests changes, update the plan and re-request
approval.

---

## Phase 3 — Design the Tests First

Once the plan is approved, design the test suite **before any implementation**
(Red → Green → Refactor per **Rule 3** in AGENTS.md).

For each public function / class in the plan, specify:

| Test Name | Module Under Test | Scenario | Expected Result | Mocks Required |
|-----------|-------------------|----------|-----------------|----------------|
| `test_<name>_happy_path` | `features/<module>/...` | Normal input | Expected output | LLM / MCP / embedding mocks |
| `test_<name>_edge_case` | ... | Edge/boundary input | Expected result | ... |
| `test_<name>_failure` | ... | Error condition | Exception / fallback | ... |

Also specify:

- **Test file path**: `tests/test_<module>/test_<filename>.py`
- **Fixtures / conftest changes** needed
- **Coverage target**: ≥ 80% for the module

Present this test plan to the user and wait for **explicit approval** before
writing any code.

---

## Phase 4 — Write the Tests (Red Phase)

After test plan approval:

1. Create the test files in `tests/test_<module>/`.
2. Write the test functions exactly as designed in Phase 3.
3. Run the tests to confirm they **fail** (Red):
   ```bash
   pytest tests/test_<module>/ -v
   ```
4. Show the user the failing test output and confirm they are failing for the
   right reason (missing implementation, not a test error).

> **Rule**: Do not proceed to Phase 5 until tests are confirmed failing.

---

## Phase 5 — Write the Implementation (Green Phase)

Now write the minimum implementation to make the tests pass:

1. Create or modify files as listed in the approved plan.
2. For **every modified file**, state:
   - File being changed
   - Lines/sections affected
   - Why the change is needed
   - Which requirement it satisfies
   (per **Rule 2** in AGENTS.md)
3. Run the tests after implementation:
   ```bash
   pytest tests/test_<module>/ -v --cov=features/<module>
   ```
4. All tests must pass (Green). Fix failures before continuing.

---

## Phase 6 — Refactor

1. Clean up the implementation without breaking tests.
2. Ensure code meets all standards:
   ```bash
   ruff check .
   mypy features/
   ```
3. Fix any linting or type errors.
4. Re-run the full test suite:
   ```bash
   pytest tests/ -v --cov=features
   ```

---

## Phase 7 — Acceptance Checklist

Before marking the feature complete, verify every item:

- [ ] All requirements for the module are implemented
- [ ] All tests pass (`pytest tests/test_<module>/ -v`)
- [ ] Code coverage >= 80% for the module
- [ ] `ruff check .` passes with zero errors
- [ ] `mypy features/` passes
- [ ] `.env.example` updated if new env vars were added
- [ ] Docstrings present on all public APIs
- [ ] Module does not cross module boundaries
- [ ] Source attribution added to all RAG/MCP responses
- [ ] `walkthrough.md` artifact updated with a summary of changes
- [ ] `requirement-docs/roadmap.md` updated with the new feature status

---

## Quick-Reference Phase Summary

```
Phase 0  ->  Understand & map to requirements
Phase 1  ->  Ask clarifying questions         [STOP -- wait for answers]
Phase 2  ->  Write implementation plan        [STOP -- wait for approval]
Phase 3  ->  Design tests                     [STOP -- wait for approval]
Phase 4  ->  Write tests (confirm Red)        [STOP -- confirm failing]
Phase 5  ->  Write implementation (Green)
Phase 6  ->  Refactor & lint
Phase 7  ->  Acceptance checklist
```

Never collapse phases. Each phase gate (STOP) requires explicit user
confirmation before proceeding.
