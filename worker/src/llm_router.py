"""LLM router module: dispatches pipeline tasks to their configured provider/model."""

import os
from dataclasses import dataclass
from typing import Protocol

import httpx

from .config import Config

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


@dataclass(frozen=True)
class LLMResponse:
    """The result of a single LLM call: generated text plus token counts and estimated cost."""

    text: str
    tokens_in: int
    tokens_out: int
    cost_usd: float


class UnsupportedProviderError(Exception):
    """Raised when config.yaml routes a task to a provider with no implemented adapter."""


class LLMProvider(Protocol):
    """A backend capable of generating text for a given model name."""

    def generate(self, system: str, prompt: str, model: str) -> LLMResponse:
        """Generate a completion for the given system prompt, user prompt, and model name."""
        ...


class Router(Protocol):
    """Structural interface satisfied by LLMRouter; lets pipeline modules take a fake in tests."""

    def generate(self, task_key: str, system: str, prompt: str) -> LLMResponse:
        """Generate text for a pipeline task, routed to whatever provider/model backs task_key."""
        ...


class OllamaProvider:
    """Local-inference provider backed by a native Ollama server (zero marginal cost)."""

    def __init__(self, host: str = DEFAULT_OLLAMA_HOST, timeout: float = 120.0) -> None:
        self._host = host.rstrip("/")
        self._timeout = timeout

    def generate(self, system: str, prompt: str, model: str) -> LLMResponse:
        """Call Ollama's /api/generate endpoint and translate its response into an LLMResponse."""
        response = httpx.post(
            f"{self._host}/api/generate",
            json={"model": model, "system": system, "prompt": prompt, "stream": False},
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return LLMResponse(
            text=payload.get("response", ""),
            tokens_in=payload.get("prompt_eval_count", 0),
            tokens_out=payload.get("eval_count", 0),
            cost_usd=0.0,
        )


class LLMRouter:
    """Routes each pipeline task to its configured provider/model and tracks cumulative cost."""

    def __init__(self, config: Config, ollama_host: str = DEFAULT_OLLAMA_HOST) -> None:
        self._config = config
        self._providers: dict[str, LLMProvider] = {"ollama": OllamaProvider(host=ollama_host)}
        self._total_cost_usd = 0.0

    @property
    def total_cost_usd(self) -> float:
        """Cumulative estimated cost of every generate() call made through this router instance."""
        return self._total_cost_usd

    def generate(self, task_key: str, system: str, prompt: str) -> LLMResponse:
        """Generate text for a pipeline task, routed to the provider/model set in config.yaml."""
        task_cfg = self._config.model_for(task_key)
        provider = self._providers.get(task_cfg.provider)
        if provider is None:
            raise UnsupportedProviderError(
                f"task '{task_key}' routes to provider '{task_cfg.provider}', which has no implemented adapter yet"
            )
        result = provider.generate(system, prompt, task_cfg.model)
        if self._config.cost_tracking.enabled:
            self._total_cost_usd += result.cost_usd
        return result
