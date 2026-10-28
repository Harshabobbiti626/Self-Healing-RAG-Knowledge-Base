import os
from pathlib import Path
from typing import Any, Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Inject native OS certificate truststore (supports enterprise SSL proxy certificates)
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass


class Settings(BaseSettings):
    """Production Settings for Self-Healing RAG Knowledge Base."""

    # Project Directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    TELEMETRY_LOG_PATH: Path = BASE_DIR / "data" / "telemetry.jsonl"

    # LLM & Model Settings
    LLM_PROVIDER: Literal["mock", "openai", "gemini"] = Field(
        default="mock",
        description="LLM provider: 'mock' for standalone testing/dev, 'openai' or 'gemini' for production.",
    )
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    SYNTHESIS_MODEL: str = "gpt-4o-mini"
    GRADER_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    VECTOR_SIZE: int = 1536

    # Qdrant Vector DB Settings
    QDRANT_LOCATION: str = ":memory:"  # Used if QDRANT_URL is not set
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "self_healing_knowledge_base"

    # Fast-Loop RAG Thresholds (Runtime Query Healing)
    RELEVANCE_THRESHOLD: float = Field(
        default=0.65,
        description="Minimum score for a retrieved chunk to be graded as RELEVANT.",
    )
    MAX_RETRIES: int = Field(
        default=2,
        description="Maximum query rewrite / regeneration iterations before graceful fallback.",
    )
    TOP_K_RETRIEVAL: int = Field(
        default=10,
        description="Initial hybrid retrieval top_k.",
    )
    TOP_K_RERANK: int = Field(
        default=5,
        description="Number of chunks retained after cross-encoder / precision reranking.",
    )

    # Slow-Loop Continuous Healing Thresholds (Knowledge Base Maintenance)
    LOW_CONFIDENCE_LOG_THRESHOLD: float = Field(
        default=0.50,
        description="Retrieval scores below this trigger an automated failure telemetry event.",
    )
    FAILURE_BUFFER_MAX_SIZE: int = 200
    CONTRADICTION_SIMILARITY_THRESHOLD: float = 0.80

    # Observability
    ENABLE_TELEMETRY: bool = True

    def model_post_init(self, __context: Any) -> None:
        if self.LLM_PROVIDER == "gemini":
            if self.SYNTHESIS_MODEL in ("gpt-4o-mini", "gemini-1.5-flash", "gemini-3.8-flash"):
                self.SYNTHESIS_MODEL = "gemini-3.1-flash-lite"
            if self.GRADER_MODEL in ("gpt-4o-mini", "gemini-1.5-flash", "gemini-3.8-flash"):
                self.GRADER_MODEL = "gemini-3.1-flash-lite"
            if self.EMBEDDING_MODEL in ("text-embedding-3-small", "text-embedding-004"):
                self.EMBEDDING_MODEL = "gemini-embedding-001"
                self.VECTOR_SIZE = 3072

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# Ensure required directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
(settings.DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
(settings.DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
(settings.DATA_DIR / "golden_eval").mkdir(parents=True, exist_ok=True)
