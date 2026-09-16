"""Application settings loaded from environment variables.

All configuration is centralised here via pydantic-settings.
Import the singleton `settings` object in other modules:

    from features.config.settings import settings
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings sourced from the .env file and environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── Google Gemini LLM ────────────────────────────────────────────────────
    google_api_key: str = Field(..., description="Google API key for Gemini Pro")
    gemini_model_name: str = Field(default="gemini-1.5-pro")
    gemini_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    gemini_max_output_tokens: int = Field(default=4096, gt=0)

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

    # ─── Application ──────────────────────────────────────────────────────────
    app_title: str = Field(default="AI Travel Planning Assistant")
    app_destination: str = Field(default="Singapore")
    log_level: str = Field(default="INFO")


# Module-level singleton — import this in other modules
settings = Settings()
