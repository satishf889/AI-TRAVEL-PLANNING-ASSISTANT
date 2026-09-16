"""Tests for prompt templates — Orchestrator module.

TDD: Prompt templates are pure string/constant values — testable without implementation.
"""

import pytest

from features.orchestrator.prompt_templates import (
    COMBINED_RAG_MCP_PROMPT_TEMPLATE,
    FALLBACK_MCP_TOOL_FAILURE,
    FALLBACK_NO_KB_CONTENT,
    RAG_QA_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)


@pytest.mark.unit
@pytest.mark.orchestrator
class TestSystemPrompt:
    """Tests that the system prompt contains all required instructions."""

    def test_system_prompt_instructs_kb_use(self) -> None:
        """System prompt instructs model to use KB for destination facts."""
        assert "Knowledge Base" in SYSTEM_PROMPT or "knowledge base" in SYSTEM_PROMPT.lower()

    def test_system_prompt_instructs_mcp_use(self) -> None:
        """System prompt instructs model to use MCP tools for real-time data."""
        assert "MCP" in SYSTEM_PROMPT or "mcp" in SYSTEM_PROMPT.lower()

    def test_system_prompt_prohibits_fabrication(self) -> None:
        """System prompt instructs model not to fabricate information."""
        prohibit_keywords = ["not", "cannot", "do not", "never", "without"]
        fabricate_keywords = ["fabricate", "invent", "made up", "hallucinate"]
        prompt_lower = SYSTEM_PROMPT.lower()
        has_prohibition = any(kw in prompt_lower for kw in fabricate_keywords)
        assert has_prohibition, "System prompt must prohibit fabrication"

    def test_system_prompt_requires_source_citations(self) -> None:
        """System prompt requires citing sources."""
        assert "source" in SYSTEM_PROMPT.lower() or "cite" in SYSTEM_PROMPT.lower()

    def test_system_prompt_distinguishes_information_types(self) -> None:
        """System prompt distinguishes KB facts, MCP data, and AI suggestions."""
        prompt_lower = SYSTEM_PROMPT.lower()
        assert "knowledge base" in prompt_lower
        assert "mcp" in prompt_lower or "tool" in prompt_lower
        assert "ai" in prompt_lower or "suggestion" in prompt_lower


@pytest.mark.unit
@pytest.mark.orchestrator
class TestRAGPromptTemplate:
    """Tests for the RAG Q&A prompt template."""

    def test_rag_template_has_context_placeholder(self) -> None:
        """RAG prompt template contains {context} placeholder."""
        assert "{context}" in RAG_QA_PROMPT_TEMPLATE

    def test_rag_template_has_question_placeholder(self) -> None:
        """RAG prompt template contains {question} placeholder."""
        assert "{question}" in RAG_QA_PROMPT_TEMPLATE

    def test_rag_template_includes_fallback_instruction(self) -> None:
        """RAG prompt template instructs model what to do if KB is insufficient."""
        template_lower = RAG_QA_PROMPT_TEMPLATE.lower()
        assert "not" in template_lower or "don't" in template_lower
        assert "information" in template_lower


@pytest.mark.unit
@pytest.mark.orchestrator
class TestCombinedPromptTemplate:
    """Tests for the combined RAG + MCP prompt template."""

    def test_combined_template_has_all_placeholders(self) -> None:
        """Combined template has {kb_context}, {mcp_data}, and {user_request}."""
        assert "{kb_context}" in COMBINED_RAG_MCP_PROMPT_TEMPLATE
        assert "{mcp_data}" in COMBINED_RAG_MCP_PROMPT_TEMPLATE
        assert "{user_request}" in COMBINED_RAG_MCP_PROMPT_TEMPLATE


@pytest.mark.unit
@pytest.mark.orchestrator
class TestFallbackMessages:
    """Tests for fallback message templates."""

    def test_kb_fallback_message_is_not_empty(self) -> None:
        """KB fallback message is a non-empty string."""
        assert FALLBACK_NO_KB_CONTENT.strip()

    def test_kb_fallback_suggests_external_source(self) -> None:
        """KB fallback message directs user to external sources."""
        assert "visitsingapore" in FALLBACK_NO_KB_CONTENT.lower() or "wikivoyage" in FALLBACK_NO_KB_CONTENT.lower()

    def test_mcp_fallback_has_tool_type_placeholder(self) -> None:
        """MCP tool failure fallback has {tool_type} placeholder."""
        assert "{tool_type}" in FALLBACK_MCP_TOOL_FAILURE
