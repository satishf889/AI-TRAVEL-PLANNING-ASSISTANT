"""Unit tests for PII Sanitizer."""
from __future__ import annotations

import pytest

from features.orchestrator.pii_sanitizer import PIISanitizer


@pytest.mark.unit
@pytest.mark.orchestrator
class TestPIISanitizer:
    """Tests for scrubbing Personally Identifiable Information (PII)."""

    def test_sanitize_emails(self) -> None:
        """Test sanitizing email addresses."""
        text = "Contact me at alice.smith@example.com or support@travel.org"
        sanitized = PIISanitizer.sanitize(text)
        assert "alice.smith@example.com" not in sanitized
        assert "support@travel.org" not in sanitized
        assert "[EMAIL_REDACTED]" in sanitized

    def test_sanitize_phone_numbers(self) -> None:
        """Test sanitizing phone numbers."""
        text = "Call me on +65 9123 4567 or +1-800-555-0199 or 9876543210"
        sanitized = PIISanitizer.sanitize(text)
        assert "9123 4567" not in sanitized
        assert "555-0199" not in sanitized
        assert "[PHONE_REDACTED]" in sanitized

    def test_sanitize_credit_card_numbers(self) -> None:
        """Test sanitizing credit card numbers."""
        text = "My card number is 4532-1234-5678-9012 for booking"
        sanitized = PIISanitizer.sanitize(text)
        assert "4532-1234-5678-9012" not in sanitized
        assert "[CARD_REDACTED]" in sanitized

    def test_sanitize_singapore_nric_fin(self) -> None:
        """Test sanitizing Singapore NRIC/FIN numbers."""
        text = "My IC is S1234567A and passport is T9876543Z"
        sanitized = PIISanitizer.sanitize(text)
        assert "S1234567A" not in sanitized
        assert "T9876543Z" not in sanitized
        assert "[ID_REDACTED]" in sanitized

    def test_clean_text_unchanged(self) -> None:
        """Test that regular travel queries are unchanged."""
        query = "What are the best attractions in Marina Bay Singapore?"
        sanitized = PIISanitizer.sanitize(query)
        assert sanitized == query
