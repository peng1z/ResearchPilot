from __future__ import annotations

import asyncio
from functools import lru_cache

import litellm

from app.config import Settings
from app.llm import _normalized_provider


DEFAULT_EMBEDDING_MODELS = {
    "openai": "text-embedding-3-small",
    "openrouter": "openai/text-embedding-3-small",
}


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self.provider = _normalized_provider(settings.llm_provider)
        self.api_key = settings.llm_api_key
        self.api_base = settings.llm_base_url
        self.backend = settings.embedding_backend.strip().lower()
        self.remote_model = settings.embedding_model or DEFAULT_EMBEDDING_MODELS.get(self.provider)
        self.local_model = settings.local_embedding_model

        if self.backend not in {"auto", "remote", "local"}:
            raise RuntimeError("EMBEDDING_BACKEND must be one of: auto, remote, local.")

    @property
    def enabled(self) -> bool:
        return self.backend != "remote" or bool(self.remote_model)

    @property
    def uses_remote(self) -> bool:
        if self.backend == "remote":
            return True
        if self.backend == "local":
            return False
        return bool(self.remote_model)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.uses_remote:
            if not self.remote_model:
                raise RuntimeError(
                    f"No remote embedding model configured for provider '{self.provider}'. Set EMBEDDING_MODEL or use EMBEDDING_BACKEND=local."
                )
            response = await asyncio.to_thread(
                litellm.embedding,
                model=self.remote_model,
                input=texts,
                api_key=self.api_key,
                api_base=self.api_base,
                caching=False,
            )
            return [item["embedding"] for item in response.data]
        return await asyncio.to_thread(self._embed_local, texts)

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        model = load_local_embedding_model(self.local_model)
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


@lru_cache(maxsize=2)
def load_local_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)
