"""Application settings loaded from environment variables.

All configuration is centralised here via pydantic-settings.
Import the singleton `settings` object in other modules:

    from features.config.settings import settings
"""

from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings sourced from the .env file and environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── Azure OpenAI LLM ─────────────────────────────────────────────────────
    azure_openai_api_key: str = Field(..., description="Azure OpenAI API key")
    azure_openai_endpoint: str = Field(..., description="Azure OpenAI Endpoint URL")
    azure_openai_api_version: str = Field(
        default="2024-08-01-preview",
        description="Azure OpenAI API version",
    )
    azure_openai_deployment_name: str = Field(
        default="gpt-5-mini",
        description="Azure OpenAI deployment / model name",
    )
    azure_openai_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    azure_openai_max_tokens: int = Field(default=4096, gt=0)

    # ─── Embedding Model ──────────────────────────────────────────────────────
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model name for embeddings",
    )
    embedding_device: str = Field(default="cpu")

    # ─── ChromaDB Vector Store ────────────────────────────────────────────────
    chroma_persist_directory: Path = Field(default=Path("./chroma_db"))
    chroma_collection_name: str = Field(default="singapore_travel")

    # ─── RAG Settings ─────────────────────────────────────────────────────────
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    retrieval_top_k: int = Field(default=5, gt=0)

    # ─── MCP: Weather (Open-Meteo) ────────────────────────────────────────────
    weather_api_base_url: str = Field(default="https://api.open-meteo.com/v1")
    weather_geocoding_url: str = Field(
        default="https://geocoding-api.open-meteo.com/v1"
    )
    destination_latitude: float = Field(default=1.3521)
    destination_longitude: float = Field(default=103.8198)
    destination_city: str = Field(default="Singapore")

    # ─── MCP: Currency (Frankfurter) ─────────────────────────────────────────
    currency_api_base_url: str = Field(default="https://api.frankfurter.app")

    # ─── Knowledge Base ───────────────────────────────────────────────────────
    knowledge_base_directory: Path = Field(default=Path("./knowledge_base"))

    # ─── Cache Settings (Redis & In-Memory) ───────────────────────────────────
    redis_enabled: bool = Field(default=True, description="Enable Redis query cache")
    redis_host: str = Field(default="redis", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    cache_ttl_seconds: int = Field(default=3600, gt=0, description="Cache TTL in seconds")


    # ─── Application ──────────────────────────────────────────────────────────
    app_title: str = Field(default="AI Travel Planning Assistant")
    app_destination: str = Field(default="Singapore")
    log_level: str = Field(default="INFO")


    # ─── Field Validators ─────────────────────────────────────────────────────

    @field_validator("azure_openai_endpoint")
    @classmethod
    def validate_azure_openai_endpoint(cls, v: str) -> str:
        """Ensure azure_openai_endpoint is a valid HTTP/HTTPS URL."""
        trimmed = v.strip()
        if not (trimmed.startswith("http://") or trimmed.startswith("https://")):
            raise ValueError(
                f"azure_openai_endpoint must start with 'http://' or 'https://', got {v!r}"
            )
        return trimmed

    @field_validator("embedding_device")
    @classmethod
    def validate_embedding_device(cls, v: str) -> str:
        """Ensure embedding_device is either 'cpu' or 'cuda'."""
        allowed: set[str] = {"cpu", "cuda"}
        if v.lower() not in allowed:
            raise ValueError(
                f"embedding_device must be one of {sorted(allowed)}, got {v!r}"
            )
        return v.lower()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log_level is a standard Python logging level."""
        valid: set[str] = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(
                f"log_level must be one of {sorted(valid)}, got {v!r}"
            )
        return upper

    @model_validator(mode="after")
    def validate_chunk_overlap_less_than_chunk_size(self) -> "Settings":
        """Ensure chunk_overlap is strictly less than chunk_size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than "
                f"chunk_size ({self.chunk_size})"
            )
        return self

    # ─── Convenience Methods ──────────────────────────────────────────────────

    def get_chroma_persist_path(self) -> Path:
        """Return chroma_persist_directory resolved to an absolute path."""
        return self.chroma_persist_directory.resolve()


# Module-level singleton — import this in other modules
settings = Settings()  # type: ignore[call-arg]  # required fields are loaded from .env at runtime
