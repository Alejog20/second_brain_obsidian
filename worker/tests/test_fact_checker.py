"""Tests for the daily, grounded fact-check pass over newly digested notes."""

from pathlib import Path

from src.fact_checker import FactCheckEntry, FactChecker
from src.llm_router import LLMResponse
from src.vault_io import Note, VaultIO


class FakeRouter:
    """Duck-types the Router protocol; returns canned text, records whether grounding was requested."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: list[tuple[str, str, str, bool]] = []

    def generate(self, task_key: str, system: str, prompt: str, grounded: bool = False) -> LLMResponse:
        self.calls.append((task_key, system, prompt, grounded))
        return LLMResponse(text=self._response_text, tokens_in=1, tokens_out=1, cost_usd=0.0)


def _vault(tmp_path: Path) -> VaultIO:
    root = tmp_path / "vault"
    root.mkdir()
    return VaultIO(root)


def test_run_returns_empty_list_with_no_entries(tmp_path: Path) -> None:
    router = FakeRouter("")
    checker = FactChecker(router, _vault(tmp_path), dry_run=False)

    assert checker.run([], "07-20-2026") == []
    assert router.calls == []


def test_run_makes_exactly_one_call_for_multiple_entries(tmp_path: Path) -> None:
    router = FakeRouter("---NOTE: Note One---\nnothing to add\n---NOTE: Note Two---\nnothing to add")
    checker = FactChecker(router, _vault(tmp_path), dry_run=False)
    entries = [
        FactCheckEntry(title="Note One", content="First claim.", rel_path="note-one.md"),
        FactCheckEntry(title="Note Two", content="Second claim.", rel_path="note-two.md"),
    ]

    checker.run(entries, "07-20-2026")

    assert len(router.calls) == 1
    task_key, _, prompt, grounded = router.calls[0]
    assert task_key == "fact_check"
    assert grounded is True
    assert "Note One" in prompt and "First claim." in prompt
    assert "Note Two" in prompt and "Second claim." in prompt


def test_nothing_to_add_notes_are_not_flagged_or_written(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    vault.write_note("note-one.md", Note(metadata={"title": "Note One"}, content="First claim."))
    router = FakeRouter("---NOTE: Note One---\nnothing to add")
    checker = FactChecker(router, vault, dry_run=False)

    items = checker.run([FactCheckEntry(title="Note One", content="First claim.", rel_path="note-one.md")], "07-20-2026")

    assert items == []
    assert vault.read_note("note-one.md").content == "First claim."


def test_flagged_note_gets_a_callout_appended_in_apply_mode(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    vault.write_note("note-one.md", Note(metadata={"title": "Note One"}, content="The sky is green."))
    router = FakeRouter("---NOTE: Note One---\nThe sky is actually blue during the day; see any basic optics reference.")
    checker = FactChecker(router, vault, dry_run=False)

    items = checker.run([FactCheckEntry(title="Note One", content="The sky is green.", rel_path="note-one.md")], "07-20-2026")

    assert len(items) == 1
    assert "Note One" in items[0]["reason"]
    updated = vault.read_note("note-one.md")
    assert "The sky is green." in updated.content
    assert "[!ai-fact-check]" in updated.content
    assert "actually blue" in updated.content


def test_flagged_note_is_staged_not_written_live_in_dry_run(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    vault.write_note("note-one.md", Note(metadata={"title": "Note One"}, content="The sky is green."))
    router = FakeRouter("---NOTE: Note One---\nThe sky is actually blue during the day.")
    checker = FactChecker(router, vault, dry_run=True)

    checker.run([FactCheckEntry(title="Note One", content="The sky is green.", rel_path="note-one.md")], "07-20-2026")

    assert vault.read_note("note-one.md").content == "The sky is green."
    staged = vault.read_note("_staging/note-one.md")
    assert "[!ai-fact-check]" in staged.content


def test_entries_with_no_matching_block_in_the_response_are_skipped(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    vault.write_note("note-one.md", Note(metadata={"title": "Note One"}, content="Body."))
    router = FakeRouter("some malformed response with no delimiters")
    checker = FactChecker(router, vault, dry_run=False)

    items = checker.run([FactCheckEntry(title="Note One", content="Body.", rel_path="note-one.md")], "07-20-2026")

    assert items == []
    assert vault.read_note("note-one.md").content == "Body."
