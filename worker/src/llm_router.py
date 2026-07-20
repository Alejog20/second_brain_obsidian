"""LLM router module: dispatches pipeline tasks to their configured provider/model."""

import os
from dataclasses import dataclass
from typing import Protocol

import httpx

from .config import Config

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DEFAULT_GEMINI_HOST = "https://generativelanguage.googleapis.com"


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


class GeminiProvider:
    """Cloud provider backed by Google's Gemini API.

    Cost is always reported as 0.0 - this targets the free tier (see config.yaml), not a
    calibrated per-token price. If you outgrow the free quota, that's no longer accurate;
    this deliberately doesn't hardcode a paid rate that would just go stale.
    """

    def __init__(self, api_key: str = DEFAULT_GEMINI_API_KEY, host: str = DEFAULT_GEMINI_HOST, timeout: float = 60.0) -> None:
        self._api_key = api_key
        self._host = host.rstrip("/")
        self._timeout = timeout

    def generate(self, system: str, prompt: str, model: str) -> LLMResponse:
        """Call Gemini's generateContent endpoint.

        The API key goes in the x-goog-api-key header, never the URL - Gemini also accepts
        it as a `?key=` query param, but query params are far more likely to end up copied
        into logs, error messages, or shell history than headers are.
        """
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY is not set - required for the 'gemini' provider")
        response = httpx.post(
            f"{self._host}/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": self._api_key, "Content-Type": "application/json"},
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "systemInstruction": {"parts": [{"text": system}]},
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        candidates = payload.get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        text = "".join(part.get("text", "") for part in parts)
        usage = payload.get("usageMetadata", {})
        return LLMResponse(
            text=text,
            tokens_in=usage.get("promptTokenCount", 0),
            tokens_out=usage.get("candidatesTokenCount", 0),
            cost_usd=0.0,
        )

    def list_models(self) -> list[str]:
        """List available Gemini models.

        A lightweight, free call (no generation quota consumed) used purely to verify the
        API key is set and valid - the CLI's `check` command uses this rather than a real
        generateContent call.
        """
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY is not set - required for the 'gemini' provider")
        response = httpx.get(
            f"{self._host}/v1beta/models",
            headers={"x-goog-api-key": self._api_key},
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return [model.get("name", "") for model in payload.get("models", [])]


class LLMRouter:
    """Routes each pipeline task to its configured provider/model and tracks cumulative cost."""

    def __init__(
        self,
        config: Config,
        ollama_host: str = DEFAULT_OLLAMA_HOST,
        gemini_api_key: str = DEFAULT_GEMINI_API_KEY,
    ) -> None:
        self._config = config
        self._providers: dict[str, LLMProvider] = {
            "ollama": OllamaProvider(host=ollama_host),
            "gemini": GeminiProvider(api_key=gemini_api_key),
        }
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
