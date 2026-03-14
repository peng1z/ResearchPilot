from __future__ import annotations

from unittest.mock import patch

from app.config import Settings
from app.embeddings import EmbeddingService


def test_auto_backend_prefers_remote_when_model_is_available() -> None:
    settings = Settings(
        LLM_PROVIDER="openai",
        LLM_MODEL="gpt-4.1-mini",
        LLM_API_KEY="test",
        EMBEDDING_BACKEND="auto",
    )

    service = EmbeddingService(settings)

    assert service.uses_remote is True


def test_auto_backend_falls_back_to_local_when_remote_model_is_missing() -> None:
    settings = Settings(
        LLM_PROVIDER="anthropic",
        LLM_MODEL="claude-3-7-sonnet-latest",
        LLM_API_KEY="test",
        EMBEDDING_BACKEND="auto",
    )

    service = EmbeddingService(settings)

    assert service.uses_remote is False


def test_local_backend_uses_sentence_transformers() -> None:
    settings = Settings(
        LLM_PROVIDER="openai",
        LLM_MODEL="gpt-4.1-mini",
        LLM_API_KEY="test",
        EMBEDDING_BACKEND="local",
        LOCAL_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5",
    )
    service = EmbeddingService(settings)

    with patch("app.embeddings.load_local_embedding_model") as loader:
        loader.return_value.encode.return_value.tolist.return_value = [[0.1, 0.2]]
        embeddings = service._embed_local(["hello world"])

    assert embeddings == [[0.1, 0.2]]

