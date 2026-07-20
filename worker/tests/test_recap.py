"""Tests for the morning reinforcement recap generator."""

from src.llm_router import LLMResponse
from src.recap import RecapEntry, RecapGenerator


class FakeRouter:
    """Duck-types the Router protocol; returns canned text per task_key, tracks calls."""

    def __init__(self, responses: dict[str, str]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, str, str]] = []

    def generate(self, task_key: str, system: str, prompt: str) -> LLMResponse:
        self.calls.append((task_key, system, prompt))
        return LLMResponse(text=self._responses.get(task_key, ""), tokens_in=1, tokens_out=1, cost_usd=0.0)


def test_build_returns_none_when_there_are_no_entries() -> None:
    router = FakeRouter({})
    generator = RecapGenerator(router)

    result = generator.build([], "07-20-2026")

    assert result is None
    assert router.calls == []


def test_build_includes_generated_text_and_date() -> None:
    router = FakeRouter({"daily_recap": "You explored Docker networking.\n\n1. What connects two containers?"})
    generator = RecapGenerator(router)

    result = generator.build([RecapEntry(title="Docker Networking", content="Bridge networks let containers talk.")], "07-20-2026")

    assert result is not None
    assert "Recap — 07-20-2026" in result
    assert "You explored Docker networking" in result
    assert "What connects two containers?" in result


def test_build_sends_all_entries_to_the_router() -> None:
    router = FakeRouter({"daily_recap": "recap text"})
    generator = RecapGenerator(router)

    generator.build(
        [
            RecapEntry(title="Note One", content="First idea."),
            RecapEntry(title="Note Two", content="Second idea."),
        ],
        "07-20-2026",
    )

    assert len(router.calls) == 1
    task_key, system, prompt = router.calls[0]
    assert task_key == "daily_recap"
    assert "Note One" in prompt
    assert "First idea." in prompt
    assert "Note Two" in prompt
    assert "Second idea." in prompt
