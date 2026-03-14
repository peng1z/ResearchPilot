from __future__ import annotations

import json
import os
from contextlib import AbstractContextManager
from typing import Any

import dspy
from pydantic import BaseModel, ValidationError

from app.config import Settings


PROVIDER_ENV_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

PROVIDER_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def _normalized_provider(provider: str) -> str:
    provider = provider.strip().lower()
    return "anthropic" if provider == "claude" else provider


def resolve_model_string(settings: Settings) -> str:
    model = settings.llm_model.strip()
    provider = _normalized_provider(settings.llm_provider)
    if "/" in model:
        return model
    return f"{provider}/{model}"


def configure_dspy(settings: Settings) -> None:
    _build_lm(settings)
    dspy.settings.configure(lm=_build_lm(settings))


def _build_lm(settings: Settings) -> dspy.LM:
    provider = _normalized_provider(settings.llm_provider)
    api_key = settings.llm_api_key
    env_var = PROVIDER_ENV_MAP.get(provider)
    if env_var and api_key and not os.getenv(env_var):
        os.environ[env_var] = api_key

    if not api_key and env_var and not os.getenv(env_var):
        raise RuntimeError(
            f"Missing API key for provider '{provider}'. Set {env_var} or LLM_API_KEY in backend/.env."
        )

    kwargs: dict[str, Any] = {
        "model": resolve_model_string(settings),
        "temperature": settings.llm_temperature,
        "cache": False,
    }
    if api_key:
        kwargs["api_key"] = api_key

    base_url = settings.llm_base_url or PROVIDER_BASE_URLS.get(provider)
    if base_url:
        kwargs["api_base"] = base_url

    return dspy.LM(**kwargs)


def dspy_context(settings: Settings) -> AbstractContextManager:
    return dspy.context(lm=_build_lm(settings))


def parse_json_payload(raw_text: str, schema: type[BaseModel]) -> BaseModel:
    text = raw_text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Model output did not contain JSON: {raw_text}")
    payload = json.loads(text[start : end + 1])
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Model output failed schema validation: {exc}") from exc
