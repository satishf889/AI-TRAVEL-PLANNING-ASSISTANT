from __future__ import annotations

"""PII Sanitization module for privacy preservation.

Scrubs personally identifiable information (emails, phone numbers,
credit card numbers, national identification numbers) before queries
and context are processed or persisted.
"""

import re


class PIISanitizer:
    """Utility class to sanitize PII from user inputs and prompt context."""

    EMAIL_REGEX = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
    # Matches international and local phone numbers (e.g. +65 9123 4567, 1-800-555-0199, 9876543210)
    PHONE_REGEX = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}')
    # Matches standard 13-19 digit credit card numbers with spaces or dashes
    CREDIT_CARD_REGEX = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{15,16}\b')
    # Matches Singapore NRIC / FIN format (e.g., S1234567A, T1234567Z, G1234567M)
    SG_ID_REGEX = re.compile(r'\b[STFGstfg]\d{7}[A-Za-z]\b')

    @classmethod
    def sanitize(cls, text: str) -> str:
        """Sanitize PII elements from the given text string.

        Args:
            text: Input string potentially containing PII.

        Returns:
            Redacted string with PII replaced by tokens.
        """
        if not text:
            return ""

        sanitized = cls.EMAIL_REGEX.sub("[EMAIL_REDACTED]", text)
        sanitized = cls.CREDIT_CARD_REGEX.sub("[CARD_REDACTED]", sanitized)
        sanitized = cls.SG_ID_REGEX.sub("[ID_REDACTED]", sanitized)
        sanitized = cls.PHONE_REGEX.sub("[PHONE_REDACTED]", sanitized)

        return sanitized
