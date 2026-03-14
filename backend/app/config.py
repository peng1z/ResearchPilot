from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "ResearchPilot API"
    cors_origins_raw: str = Field(default="http://localhost:3000", alias="CORS_ORIGINS")
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4.1-mini", alias="LLM_MODEL")
    llm_api_key: Optional[str] = Field(default=None, alias="LLM_API_KEY")
    llm_base_url: Optional[str] = Field(default=None, alias="LLM_BASE_URL")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    embedding_backend: str = Field(default="auto", alias="EMBEDDING_BACKEND")
    embedding_model: Optional[str] = Field(default=None, alias="EMBEDDING_MODEL")
    local_embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="LOCAL_EMBEDDING_MODEL")
    semantic_scholar_api_key: Optional[str] = Field(default=None, alias="SEMANTIC_SCHOLAR_API_KEY")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    report_db_path: str = Field(default="data/researchpilot.db", alias="REPORT_DB_PATH")

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins_raw.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def apply_runtime_overrides(settings: Settings, overrides: dict[str, Any]) -> Settings:
    cleaned = {
        key: value
        for key, value in overrides.items()
        if value is not None and not (isinstance(value, str) and value.strip() == "")
    }
    return settings.model_copy(update=cleaned)
