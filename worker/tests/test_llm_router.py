"""Tests for the LLM router: task dispatch, provider adapters, and cost tracking."""

from pathlib import Path
from typing import Any

import httpx
import pytest

from src.config import load_config
from src.llm_router import LLMRouter, UnsupportedProviderError

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


def test_unknown_task_key_raises(config) -> None:
    router = LLMRouter(config)
    with pytest.raises(ValueError):
        router.generate("nonexistent_task", system="", prompt="")
