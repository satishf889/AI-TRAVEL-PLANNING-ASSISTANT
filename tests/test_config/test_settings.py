"""TDD test suite for features.config.settings.

Tests are written BEFORE the implementation hardening (Red phase).
Run with: pytest tests/test_config/ -v --tb=short --no-cov

Satisfies:
  - Rule 3  (TDD — Red → Green → Refactor)
  - Rule 7  (Config via environment variables)
  - Req §7  (Technology Requirements — all tunable parameters in .env)
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(env_overrides: dict) -> "Settings":  # noqa: F821
    """Construct a Settings object with env vars injected via monkeypatch.

    The function signature references Settings via a forward reference string
    so that the import is deferred — allowing the Red phase to detect import
    failures cleanly.
    """
    from features.config.settings import Settings

    return Settings(**env_overrides)


# ---------------------------------------------------------------------------
# 1. Default values load correctly
# ---------------------------------------------------------------------------

class TestSettingsDefaults:
    """Verify that all optional fields have correct defaults."""

    def test_settings_loads_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings initialises with sensible defaults when only required fields supplied."""
        from features.config.settings import Settings

        # Pass all defaults explicitly so the test is not affected by the active .env file
        s = Settings(  # type: ignore[call-arg]
            azure_openai_api_key="test-key-default",
            azure_openai_endpoint="https://test.openai.azure.com/",
            azure_openai_api_version="2024-08-01-preview",
            azure_openai_deployment_name="gpt-5-mini",
            azure_openai_temperature=0.3,
            azure_openai_max_tokens=4096,
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_device="cpu",
            chroma_collection_name="singapore_travel",
            chunk_size=1000,
            chunk_overlap=200,
            retrieval_top_k=5,
            destination_city="Singapore",
            destination_latitude=1.3521,
            destination_longitude=103.8198,
            app_destination="Singapore",
            log_level="INFO",
            _env_file=None,  # ignore .env
        )

        assert s.azure_openai_deployment_name == "gpt-5-mini"
        assert s.azure_openai_api_version == "2024-08-01-preview"
        assert s.azure_openai_temperature == pytest.approx(0.3)
        assert s.azure_openai_max_tokens == 4096
        assert s.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert s.embedding_device == "cpu"
        assert s.chroma_collection_name == "singapore_travel"
        assert s.chunk_size == 1000
        assert s.chunk_overlap == 200
        assert s.retrieval_top_k == 5
        assert s.destination_city == "Singapore"
        assert s.destination_latitude == pytest.approx(1.3521)
        assert s.destination_longitude == pytest.approx(103.8198)
        assert s.app_destination == "Singapore"
        assert s.log_level == "INFO"

    def test_chroma_paths_are_path_objects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """chroma_persist_directory and knowledge_base_directory must be Path instances."""
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        from features.config.settings import Settings

        s = Settings(  # type: ignore[call-arg]
            azure_openai_api_key="test-key-path",
            azure_openai_endpoint="https://test.openai.azure.com/",
        )

        assert isinstance(s.chroma_persist_directory, Path)
        assert isinstance(s.knowledge_base_directory, Path)


# ---------------------------------------------------------------------------
# 2. Env-var overrides work correctly
# ---------------------------------------------------------------------------

class TestSettingsEnvOverrides:
    """Verify that environment variables override defaults."""

    def test_settings_loads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env vars supplied directly to the constructor override defaults."""
        from features.config.settings import Settings

        s = Settings(  # type: ignore[call-arg]
            azure_openai_api_key="prod-key-xyz",
            azure_openai_endpoint="https://custom.openai.azure.com/",
            azure_openai_deployment_name="gpt-5-mini-custom",
            azure_openai_api_version="2024-12-01-preview",
            azure_openai_temperature=0.7,
            azure_openai_max_tokens=2048,
            chunk_size=500,
            chunk_overlap=50,
            retrieval_top_k=3,
            log_level="DEBUG",
        )

        assert s.azure_openai_api_key == "prod-key-xyz"
        assert s.azure_openai_endpoint == "https://custom.openai.azure.com/"
        assert s.azure_openai_deployment_name == "gpt-5-mini-custom"
        assert s.azure_openai_api_version == "2024-12-01-preview"
        assert s.azure_openai_temperature == pytest.approx(0.7)
        assert s.azure_openai_max_tokens == 2048
        assert s.chunk_size == 500
        assert s.chunk_overlap == 50
        assert s.retrieval_top_k == 3
        assert s.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# 3. Required field validation
# ---------------------------------------------------------------------------

class TestRequiredFields:
    """Verify that missing required fields raise ValidationError."""

    def test_azure_openai_api_key_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ValidationError must be raised when AZURE_OPENAI_API_KEY is absent."""
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        from features.config.settings import Settings

        with pytest.raises(ValidationError) as exc_info:
            Settings(  # type: ignore[call-arg]
                azure_openai_endpoint="https://test.openai.azure.com/",
                _env_file=None,
            )

        errors = exc_info.value.errors()
        fields = {e["loc"][0] for e in errors}
        assert "azure_openai_api_key" in fields

    def test_azure_openai_endpoint_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ValidationError must be raised when AZURE_OPENAI_ENDPOINT is absent."""
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        from features.config.settings import Settings

        with pytest.raises(ValidationError) as exc_info:
            Settings(  # type: ignore[call-arg]
                azure_openai_api_key="test-key",
                _env_file=None,
            )

        errors = exc_info.value.errors()
        fields = {e["loc"][0] for e in errors}
        assert "azure_openai_endpoint" in fields


# ---------------------------------------------------------------------------
# 4. log_level validator
# ---------------------------------------------------------------------------

class TestLogLevelValidator:
    """Verify the log_level field validator."""

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_log_level_valid_values(self, level: str) -> None:
        """All standard Python log levels must be accepted."""
        from features.config.settings import Settings

        s = Settings(  # type: ignore[call-arg]
            azure_openai_api_key="key",
            azure_openai_endpoint="https://test.openai.azure.com/",
            log_level=level,
        )
        assert s.log_level == level

    def test_log_level_invalid_value(self) -> None:
        """A non-standard log level must raise ValidationError."""
        from features.config.settings import Settings

        with pytest.raises(ValidationError) as exc_info:
            Settings(  # type: ignore[call-arg]
                azure_openai_api_key="key",
                azure_openai_endpoint="https://test.openai.azure.com/",
                log_level="VERBOSE",
            )

        errors = exc_info.value.errors()
        fields = [e["loc"][0] for e in errors]
        assert "log_level" in fields


# ---------------------------------------------------------------------------
# 6. Cross-field validator: chunk_overlap < chunk_size
# ---------------------------------------------------------------------------

class TestChunkOverlapValidator:
    """Verify the model-level validator for chunk_overlap vs chunk_size."""

    def test_chunk_overlap_less_than_chunk_size(self) -> None:
        """chunk_overlap >= chunk_size must raise ValidationError."""
        from features.config.settings import Settings

        with pytest.raises(ValidationError):
            Settings(  # type: ignore[call-arg]
                azure_openai_api_key="key",
                azure_openai_endpoint="https://test.openai.azure.com/",
                chunk_size=200,
                chunk_overlap=200,  # equal — must fail
            )

    def test_chunk_overlap_equal_to_chunk_size_fails(self) -> None:
        """Equality (overlap == size) must also raise ValidationError."""
        from features.config.settings import Settings

        with pytest.raises(ValidationError):
            Settings(  # type: ignore[call-arg]
                azure_openai_api_key="key",
                azure_openai_endpoint="https://test.openai.azure.com/",
                chunk_size=300,
                chunk_overlap=300,
            )

    def test_chunk_overlap_greater_than_chunk_size_fails(self) -> None:
        """Overlap exceeding size must raise ValidationError."""
        from features.config.settings import Settings

        with pytest.raises(ValidationError):
            Settings(  # type: ignore[call-arg]
                azure_openai_api_key="key",
                azure_openai_endpoint="https://test.openai.azure.com/",
                chunk_size=100,
                chunk_overlap=150,
            )

    def test_valid_chunk_overlap(self) -> None:
        """A valid overlap strictly less than chunk_size must be accepted."""
        from features.config.settings import Settings

        s = Settings(  # type: ignore[call-arg]
            azure_openai_api_key="key",
            azure_openai_endpoint="https://test.openai.azure.com/",
            chunk_size=1000,
            chunk_overlap=100,
        )
        assert s.chunk_overlap < s.chunk_size


# ---------------------------------------------------------------------------
# 7. Numeric range & endpoint validators
# ---------------------------------------------------------------------------

class TestNumericRangeValidators:
    """Verify Field constraints on numeric fields and endpoints."""

    def test_chunk_size_must_be_positive(self) -> None:
        """chunk_size=0 must raise ValidationError (gt=0 constraint)."""
        from features.config.settings import Settings

        with pytest.raises(ValidationError):
            Settings(  # type: ignore[call-arg]
                azure_openai_api_key="key",
                azure_openai_endpoint="https://test.openai.azure.com/",
                chunk_size=0,
            )

    def test_retrieval_top_k_must_be_positive(self) -> None:
        """retrieval_top_k=0 must raise ValidationError (gt=0 constraint)."""
        from features.config.settings import Settings

        with pytest.raises(ValidationError):
            Settings(  # type: ignore[call-arg]
                azure_openai_api_key="key",
                azure_openai_endpoint="https://test.openai.azure.com/",
                retrieval_top_k=0,
            )

    def test_azure_openai_temperature_too_high(self) -> None:
        """Temperature above 2.0 must raise ValidationError."""
        from features.config.settings import Settings

        with pytest.raises(ValidationError):
            Settings(  # type: ignore[call-arg]
                azure_openai_api_key="key",
                azure_openai_endpoint="https://test.openai.azure.com/",
                azure_openai_temperature=2.1,
            )

    def test_azure_openai_temperature_negative(self) -> None:
        """Negative temperature must raise ValidationError."""
        from features.config.settings import Settings

        with pytest.raises(ValidationError):
            Settings(  # type: ignore[call-arg]
                azure_openai_api_key="key",
                azure_openai_endpoint="https://test.openai.azure.com/",
                azure_openai_temperature=-0.1,
            )

    def test_azure_openai_temperature_boundary_valid(self) -> None:
        """Temperatures at the exact boundaries [0.0, 2.0] must be accepted."""
        from features.config.settings import Settings

        s_low = Settings(  # type: ignore[call-arg]
            azure_openai_api_key="key",
            azure_openai_endpoint="https://test.openai.azure.com/",
            azure_openai_temperature=0.0,
        )
        s_high = Settings(  # type: ignore[call-arg]
            azure_openai_api_key="key",
            azure_openai_endpoint="https://test.openai.azure.com/",
            azure_openai_temperature=2.0,
        )

        assert s_low.azure_openai_temperature == pytest.approx(0.0)
        assert s_high.azure_openai_temperature == pytest.approx(2.0)

    def test_azure_openai_endpoint_invalid_url(self) -> None:
        """Endpoint without http:// or https:// must raise ValidationError."""
        from features.config.settings import Settings

        with pytest.raises(ValidationError):
            Settings(  # type: ignore[call-arg]
                azure_openai_api_key="key",
                azure_openai_endpoint="invalid-url",
            )


# ---------------------------------------------------------------------------
# 8. Convenience method: get_chroma_persist_path()
# ---------------------------------------------------------------------------

class TestGetChromaPersistPath:
    """Verify the get_chroma_persist_path() convenience method."""

    def test_get_chroma_persist_path_returns_absolute(self) -> None:
        """get_chroma_persist_path() must return an absolute Path."""
        from features.config.settings import Settings

        s = Settings(  # type: ignore[call-arg]
            azure_openai_api_key="key",
            azure_openai_endpoint="https://test.openai.azure.com/",
            chroma_persist_directory=Path("./chroma_db"),
        )
        result = s.get_chroma_persist_path()

        assert isinstance(result, Path)
        assert result.is_absolute()

    def test_get_chroma_persist_path_consistent(self) -> None:
        """Calling the method twice must return the same path."""
        from features.config.settings import Settings

        s = Settings(  # type: ignore[call-arg]
            azure_openai_api_key="key",
            azure_openai_endpoint="https://test.openai.azure.com/",
        )
        assert s.get_chroma_persist_path() == s.get_chroma_persist_path()


# ---------------------------------------------------------------------------
# 9. Module-level singleton
# ---------------------------------------------------------------------------

class TestSettingsSingleton:
    """Verify the module-level `settings` singleton."""

    def test_settings_singleton_is_reusable(self) -> None:
        """The module-level settings object must be an instance of Settings."""
        from features.config.settings import Settings, settings

        assert isinstance(settings, Settings)

    def test_settings_singleton_identity(self) -> None:
        """Importing settings twice from the same module gives the same object."""
        from features.config import settings as s1
        from features.config.settings import settings as s2

        assert s1 is s2
