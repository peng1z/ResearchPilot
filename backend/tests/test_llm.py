from __future__ import annotations

import os
from unittest import mock

import pytest
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from litellm import ContextWindowExceededError
from pydantic import BaseModel

from app.config import Settings
from app.llm import (
    PROVIDER_BASE_URLS,
    AsyncFallbackChatAdapter,
    _build_lm,
    _normalized_provider,
    parse_json_payload,
    resolve_model_string,
)


def _settings(**overrides) -> Settings:
    base = {"llm_provider": "openai", "llm_model": "gpt-4.1-mini", "llm_api_key": "sk-test"}
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    """_build_lm writes into os.environ; keep that out of other tests."""
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)


# --- provider normalisation ----------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [("claude", "anthropic"), ("Claude", "anthropic"), ("  CLAUDE  ", "anthropic"),
     ("anthropic", "anthropic"), ("OpenAI", "openai"), (" groq ", "groq")],
)
def test_provider_names_are_normalised(given: str, expected: str) -> None:
    assert _normalized_provider(given) == expected


# --- model string ---------------------------------------------------------


def test_model_without_a_slash_is_prefixed_with_the_provider() -> None:
    assert resolve_model_string(_settings(llm_provider="groq", llm_model="llama-3.3-70b")) == (
        "groq/llama-3.3-70b"
    )


def test_a_model_that_already_names_its_provider_is_left_alone() -> None:
    settings = _settings(llm_provider="openrouter", llm_model="openai/gpt-4.1-mini")
    assert resolve_model_string(settings) == "openai/gpt-4.1-mini"


def test_claude_is_normalised_in_the_model_string_too() -> None:
    assert resolve_model_string(_settings(llm_provider="claude", llm_model="sonnet")).startswith(
        "anthropic/"
    )


# --- building the LM ------------------------------------------------------


def test_a_missing_key_fails_with_the_variable_to_set() -> None:
    with pytest.raises(RuntimeError) as excinfo:
        _build_lm(_settings(llm_provider="groq", llm_api_key=None))
    message = str(excinfo.value)
    assert "GROQ_API_KEY" in message
    assert "groq" in message


def test_the_configured_key_is_exported_for_libraries_that_read_the_environment() -> None:
    """A deliberate side effect: litellm and dspy read os.environ, not Settings."""
    assert "OPENAI_API_KEY" not in os.environ
    _build_lm(_settings(llm_api_key="sk-from-settings"))
    assert os.environ["OPENAI_API_KEY"] == "sk-from-settings"


def test_an_existing_environment_key_is_not_overwritten(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-already-set")
    _build_lm(_settings(llm_api_key="sk-from-settings"))
    assert os.environ["OPENAI_API_KEY"] == "sk-already-set"


def test_an_environment_key_alone_is_enough(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env-only")
    _build_lm(_settings(llm_provider="openrouter", llm_api_key=None))  # must not raise


def test_known_providers_get_their_base_url() -> None:
    lm = _build_lm(_settings(llm_provider="openrouter", llm_api_key="sk-test"))
    # dspy.LM keeps provider options in .kwargs rather than as attributes.
    assert lm.kwargs["api_base"] == PROVIDER_BASE_URLS["openrouter"]


def test_an_explicit_base_url_wins_over_the_provider_default() -> None:
    lm = _build_lm(
        _settings(llm_provider="openrouter", llm_api_key="sk-test", llm_base_url="http://local:8080/v1")
    )
    assert lm.kwargs["api_base"] == "http://local:8080/v1"


# --- JSON extraction ------------------------------------------------------


class _Shape(BaseModel):
    name: str
    count: int


def test_json_is_extracted_from_surrounding_prose() -> None:
    raw = 'Sure! Here is the result:\n{"name": "a", "count": 2}\nHope that helps.'
    assert parse_json_payload(raw, _Shape) == _Shape(name="a", count=2)


def test_json_inside_a_fenced_block_is_extracted() -> None:
    raw = '```json\n{"name": "a", "count": 2}\n```'
    assert parse_json_payload(raw, _Shape).name == "a"


@pytest.mark.parametrize("raw", ["", "no braces here", "}{ backwards"])
def test_output_without_json_is_rejected(raw: str) -> None:
    with pytest.raises(ValueError, match="did not contain JSON"):
        parse_json_payload(raw, _Shape)


def test_json_that_does_not_match_the_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="failed schema validation"):
        parse_json_payload('{"name": "a"}', _Shape)


def test_the_raw_text_is_kept_in_the_error_for_debugging() -> None:
    with pytest.raises(ValueError, match="totally unparseable"):
        parse_json_payload("totally unparseable", _Shape)


class _Claims(BaseModel):
    claims: list[str]


def test_parse_json_payload_accepts_a_python_dict_literal() -> None:
    # Models frequently answer with single quotes, which is a valid Python
    # literal but not JSON.
    parsed = parse_json_payload(
        "{'claims': ['a'], 'methods': [], 'datasets': [], 'results': [], 'limitations': []}",
        _Claims,
    )

    assert parsed.claims == ["a"]


def test_parse_json_payload_reports_unparseable_output_as_a_value_error() -> None:
    # json.loads used to run outside the try, so a malformed payload escaped as
    # a bare JSONDecodeError naming neither the model nor its output.
    with pytest.raises(ValueError, match="not parseable JSON"):
        parse_json_payload("{not: valid, 'either'}", _Claims)


def test_parse_json_payload_rejects_a_non_object_payload() -> None:
    # literal_eval reads this as a set, not a mapping.
    with pytest.raises(ValueError, match="not a JSON object"):
        parse_json_payload("{'a', 'b'}", _Claims)


@pytest.mark.anyio
async def test_async_fallback_adapter_retries_through_json_adapter() -> None:
    # dspy's ChatAdapter retries via JSONAdapter in __call__ but not in acall,
    # so an awaited call raised where the same call succeeded synchronously.
    adapter = AsyncFallbackChatAdapter()
    calls: list[str] = []

    async def failing_super(*args, **kwargs):
        calls.append("chat")
        raise ValueError("could not parse")

    async def json_acall(self, *args, **kwargs):
        calls.append("json")
        return [{"extraction_json": "{}"}]

    with mock.patch.object(ChatAdapter, "acall", failing_super), mock.patch.object(
        JSONAdapter, "acall", json_acall
    ):
        result = await adapter.acall(None, {}, None, [], {})

    assert calls == ["chat", "json"]
    assert result == [{"extraction_json": "{}"}]


@pytest.mark.anyio
async def test_async_fallback_adapter_does_not_retry_a_context_window_error() -> None:
    adapter = AsyncFallbackChatAdapter()

    async def failing_super(*args, **kwargs):
        raise ContextWindowExceededError("too long", "openrouter", "m")

    with mock.patch.object(ChatAdapter, "acall", failing_super):
        with pytest.raises(ContextWindowExceededError):
            await adapter.acall(None, {}, None, [], {})
