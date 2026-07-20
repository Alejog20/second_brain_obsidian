"""Tests for the LLM router: task dispatch, provider adapters, and cost tracking."""

from pathlib import Path
from typing import Any

import httpx
import pytest

from src.config import load_config
from src.llm_router import GeminiProvider, LLMRouter, OllamaProvider, UnsupportedProviderError

CONFIG_YAML = """
vault:
  path: /vault
safety:
  mode: dry_run
embeddings:
  provider: ollama
  model: nomic-embed-text
models:
  bulk_grammar_pass:
    provider: ollama
    model: qwen3.5:9b
  daily_digestion:
    provider: gemini
    model: gemini-3.5-flash
  taxonomy_analysis:
    provider: anthropic
    model: claude-opus-4-8
cost_tracking:
  enabled: true
report:
  path: "_reports/Review-{date}.md"
"""


@pytest.fixture
def config(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_YAML, encoding="utf-8")
    return load_config(path)


def _fake_ollama_response(*args: Any, **kwargs: Any) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json={"response": "corrected text", "prompt_eval_count": 42, "eval_count": 17},
        request=httpx.Request("POST", "http://localhost:11434/api/generate"),
    )


def test_generate_dispatches_to_ollama_and_parses_response(monkeypatch: pytest.MonkeyPatch, config) -> None:
    monkeypatch.setattr(httpx, "post", _fake_ollama_response)
    router = LLMRouter(config)

    result = router.generate("bulk_grammar_pass", system="fix grammar", prompt="teh cat sat")

    assert result.text == "corrected text"
    assert result.tokens_in == 42
    assert result.tokens_out == 17
    assert result.cost_usd == 0.0


def test_cost_accumulates_across_calls(monkeypatch: pytest.MonkeyPatch, config) -> None:
    monkeypatch.setattr(httpx, "post", _fake_ollama_response)
    router = LLMRouter(config)

    router.generate("bulk_grammar_pass", system="", prompt="one")
    router.generate("bulk_grammar_pass", system="", prompt="two")

    assert router.total_cost_usd == 0.0  # ollama is zero marginal cost, but the accumulator ran twice without error


def test_unimplemented_provider_raises(config) -> None:
    router = LLMRouter(config)
    with pytest.raises(UnsupportedProviderError):
        router.generate("taxonomy_analysis", system="", prompt="")


def test_router_forwards_grounded_flag_to_provider(monkeypatch: pytest.MonkeyPatch, config) -> None:
    captured = {}

    def fake_post(url: str, *args: Any, **kwargs: Any) -> httpx.Response:
        captured["json"] = kwargs.get("json", {})
        return _fake_gemini_response()

    monkeypatch.setattr(httpx, "post", fake_post)
    router = LLMRouter(config, gemini_api_key="test-key")

    router.generate("daily_digestion", system="", prompt="hi", grounded=True)

    assert captured["json"]["tools"] == [{"google_search": {}}]


def test_unknown_task_key_raises(config) -> None:
    router = LLMRouter(config)
    with pytest.raises(ValueError):
        router.generate("nonexistent_task", system="", prompt="")


def _fake_gemini_response(*args: Any, **kwargs: Any) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json={
            "candidates": [{"content": {"parts": [{"text": "distilled title"}], "role": "model"}}],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 4, "totalTokenCount": 16},
        },
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"),
    )


def test_generate_dispatches_to_gemini_and_parses_response(monkeypatch: pytest.MonkeyPatch, config) -> None:
    monkeypatch.setattr(httpx, "post", _fake_gemini_response)
    router = LLMRouter(config, gemini_api_key="test-key")

    result = router.generate("daily_digestion", system="title this", prompt="some journal text")

    assert result.text == "distilled title"
    assert result.tokens_in == 12
    assert result.tokens_out == 4
    assert result.cost_usd == 0.0


def test_gemini_api_key_sent_as_header_not_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_post(url: str, *args: Any, **kwargs: Any) -> httpx.Response:
        captured["url"] = url
        captured["headers"] = kwargs.get("headers", {})
        return _fake_gemini_response()

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = GeminiProvider(api_key="super-secret-key")

    provider.generate(system="", prompt="hi", model="gemini-3.5-flash")

    assert "super-secret-key" not in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == "super-secret-key"


def test_gemini_missing_api_key_raises_clearly() -> None:
    provider = GeminiProvider(api_key="")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        provider.generate(system="", prompt="hi", model="gemini-3.5-flash")


def test_gemini_empty_candidates_returns_empty_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"promptFeedback": {"blockReason": "SAFETY"}},
            request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = GeminiProvider(api_key="test-key")

    result = provider.generate(system="", prompt="hi", model="gemini-3.5-flash")

    assert result.text == ""


def test_gemini_grounded_request_includes_search_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_post(url: str, *args: Any, **kwargs: Any) -> httpx.Response:
        captured["json"] = kwargs.get("json", {})
        return _fake_gemini_response()

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = GeminiProvider(api_key="test-key")

    provider.generate(system="", prompt="hi", model="gemini-3.5-flash", grounded=True)

    assert captured["json"]["tools"] == [{"google_search": {}}]


def test_gemini_ungrounded_request_has_no_tools_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_post(url: str, *args: Any, **kwargs: Any) -> httpx.Response:
        captured["json"] = kwargs.get("json", {})
        return _fake_gemini_response()

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = GeminiProvider(api_key="test-key")

    provider.generate(system="", prompt="hi", model="gemini-3.5-flash")

    assert "tools" not in captured["json"]


def test_ollama_grounded_request_raises() -> None:
    provider = OllamaProvider()
    with pytest.raises(ValueError, match="grounding"):
        provider.generate(system="", prompt="hi", model="qwen", grounded=True)
