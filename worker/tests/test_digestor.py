"""Tests for the digestor: chunk segmentation, merge-vs-create routing, and daily-note summary."""

from pathlib import Path

import pytest

from src.digestor import Digestor
from src.llm_router import LLMResponse
from src.vault_io import Note, VaultIO
from src.vector_store import EmbeddedChunk, VectorStore


class FakeRouter:
    """Duck-types the Router protocol; returns canned text per task_key."""

    def __init__(self, responses: dict[str, str]) -> None:
        self._responses = responses

    def generate(self, task_key: str, system: str, prompt: str) -> LLMResponse:
        return LLMResponse(text=self._responses.get(task_key, ""), tokens_in=1, tokens_out=1, cost_usd=0.0)


class FakeEmbedder:
    """Returns a fixed vector regardless of input, for deterministic vector-store tests."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def embed(self, text: str) -> list[float]:
        return self._vector


class VaryingEmbedder:
    """Returns a distinct, pre-registered vector per input text, so unrelated chunks don't merge."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    def embed(self, text: str) -> list[float]:
        return self._mapping[text]


@pytest.fixture
def vault(tmp_path: Path) -> VaultIO:
    root = tmp_path / "vault"
    root.mkdir()
    return VaultIO(root)


@pytest.fixture
def vector_store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "vector_store", embedding_dim=4)


def test_segment_splits_on_headings() -> None:
    text = "# First\nfirst body\n\n# Second\nsecond body"
    chunks = Digestor._segment(text)
    assert len(chunks) == 2
    assert chunks[0].startswith("# First")
    assert chunks[1].startswith("# Second")


def test_segment_falls_back_to_paragraphs_without_headings() -> None:
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = Digestor._segment(text)
    assert chunks == ["First paragraph.", "Second paragraph.", "Third paragraph."]


def test_create_new_note_for_unmatched_chunk(vault: VaultIO, vector_store: VectorStore) -> None:
    vault.write_note("01-Daily/07-16-2026.md", Note(metadata={}, content="# New Idea\nSome idea text."))
    router = FakeRouter({"daily_digestion": "My New Idea"})
    digestor = Digestor(router, FakeEmbedder([1.0, 0.0, 0.0, 0.0]), vector_store, vault, dry_run=False, default_folder="00-Inbox")

    results = digestor.digest("01-Daily/07-16-2026.md", "07-16-2026")

    assert len(results) == 1
    assert results[0].merged_into_existing is False
    assert results[0].rel_path == "00-Inbox/My-New-Idea.md"
    created = vault.read_note("00-Inbox/My-New-Idea.md")
    assert created.metadata["source"] == "[[07-16-2026]]"
    assert created.metadata["created"] == "07-16-2026"
    assert created.metadata["status"] == "draft"
    assert created.metadata["tags"] == []


def test_merge_into_existing_note_for_matched_chunk(vault: VaultIO, vector_store: VectorStore) -> None:
    vault.write_note("04-Reference/existing.md", Note(metadata={"title": "Existing Note"}, content="Existing content."))
    vector_store.upsert(
        EmbeddedChunk(id="04-Reference/existing.md", text="Existing content.", vector=[1.0, 0.0, 0.0, 0.0], path="04-Reference/existing.md", note_title="Existing Note")
    )
    vault.write_note("01-Daily/07-16-2026.md", Note(metadata={}, content="# Update\nMore about the existing topic."))
    router = FakeRouter({"daily_digestion": "Should not be called"})
    digestor = Digestor(router, FakeEmbedder([1.0, 0.0, 0.0, 0.0]), vector_store, vault, dry_run=False, default_folder="00-Inbox")

    results = digestor.digest("01-Daily/07-16-2026.md", "07-16-2026")

    assert len(results) == 1
    assert results[0].merged_into_existing is True
    assert results[0].rel_path == "04-Reference/existing.md"
    assert not vault.exists("00-Inbox")
    merged = vault.read_note("04-Reference/existing.md")
    assert "Existing content." in merged.content
    assert "More about the existing topic." in merged.content
    assert "[[07-16-2026]]" in merged.content


def test_daily_note_gets_notes_generated_section(vault: VaultIO, vector_store: VectorStore) -> None:
    vault.write_note("01-Daily/07-16-2026.md", Note(metadata={}, content="# New Idea\nSome idea text."))
    router = FakeRouter({"daily_digestion": "My New Idea"})
    digestor = Digestor(router, FakeEmbedder([1.0, 0.0, 0.0, 0.0]), vector_store, vault, dry_run=False, default_folder="00-Inbox")

    digestor.digest("01-Daily/07-16-2026.md", "07-16-2026")

    daily = vault.read_note("01-Daily/07-16-2026.md")
    assert "## Notes generated" in daily.content
    assert "[[My New Idea]] (created)" in daily.content


def test_dry_run_stages_new_note_instead_of_writing_live(vault: VaultIO, vector_store: VectorStore) -> None:
    vault.write_note("01-Daily/07-16-2026.md", Note(metadata={}, content="# New Idea\nSome idea text."))
    router = FakeRouter({"daily_digestion": "My New Idea"})
    digestor = Digestor(router, FakeEmbedder([1.0, 0.0, 0.0, 0.0]), vector_store, vault, dry_run=True, default_folder="00-Inbox")

    digestor.digest("01-Daily/07-16-2026.md", "07-16-2026")

    assert not vault.exists("00-Inbox/My-New-Idea.md")
    assert vault.exists("_staging/00-Inbox/My-New-Idea.md")


def test_default_folder_root_creates_note_at_vault_root(vault: VaultIO, vector_store: VectorStore) -> None:
    vault.write_note("01-Daily/07-16-2026.md", Note(metadata={}, content="# New Idea\nSome idea text."))
    router = FakeRouter({"daily_digestion": "My New Idea"})
    digestor = Digestor(router, FakeEmbedder([1.0, 0.0, 0.0, 0.0]), vector_store, vault, dry_run=False, default_folder="")

    results = digestor.digest("01-Daily/07-16-2026.md", "07-16-2026")

    assert results[0].rel_path == "My-New-Idea.md"
    assert vault.exists("My-New-Idea.md")


def test_no_chunks_leaves_daily_note_untouched(vault: VaultIO, vector_store: VectorStore) -> None:
    vault.write_note("01-Daily/07-16-2026.md", Note(metadata={}, content=""))
    router = FakeRouter({})
    digestor = Digestor(router, FakeEmbedder([1.0, 0.0, 0.0, 0.0]), vector_store, vault, dry_run=False, default_folder="00-Inbox")

    results = digestor.digest("01-Daily/07-16-2026.md", "07-16-2026")

    assert results == []
    assert "Notes generated" not in vault.read_note("01-Daily/07-16-2026.md").content


def test_two_unrelated_chunks_with_same_proposed_title_do_not_collide(vault: VaultIO, vector_store: VectorStore) -> None:
    daily_content = "# Idea One\nFirst distinct idea.\n\n# Idea Two\nSecond distinct idea."
    vault.write_note("01-Daily/07-16-2026.md", Note(metadata={}, content=daily_content))
    router = FakeRouter({"daily_digestion": "My New Idea"})
    embedder = VaryingEmbedder(
        {
            "# Idea One\nFirst distinct idea.": [1.0, 0.0, 0.0, 0.0],
            "# Idea Two\nSecond distinct idea.": [0.0, 1.0, 0.0, 0.0],
        }
    )
    digestor = Digestor(router, embedder, vector_store, vault, dry_run=False, default_folder="00-Inbox")

    results = digestor.digest("01-Daily/07-16-2026.md", "07-16-2026")

    assert len(results) == 2
    rel_paths = {r.rel_path for r in results}
    assert rel_paths == {"00-Inbox/My-New-Idea.md", "00-Inbox/My-New-Idea-2.md"}
    assert vault.read_note("00-Inbox/My-New-Idea.md").content.strip() == "# Idea One\nFirst distinct idea."
    assert vault.read_note("00-Inbox/My-New-Idea-2.md").content.strip() == "# Idea Two\nSecond distinct idea."


def test_collision_check_respects_dry_run_staging_path(vault: VaultIO, vector_store: VectorStore) -> None:
    daily_content = "# Idea One\nFirst distinct idea.\n\n# Idea Two\nSecond distinct idea."
    vault.write_note("01-Daily/07-16-2026.md", Note(metadata={}, content=daily_content))
    router = FakeRouter({"daily_digestion": "My New Idea"})
    embedder = VaryingEmbedder(
        {
            "# Idea One\nFirst distinct idea.": [1.0, 0.0, 0.0, 0.0],
            "# Idea Two\nSecond distinct idea.": [0.0, 1.0, 0.0, 0.0],
        }
    )
    digestor = Digestor(router, embedder, vector_store, vault, dry_run=True, default_folder="00-Inbox")

    results = digestor.digest("01-Daily/07-16-2026.md", "07-16-2026")

    rel_paths = {r.rel_path for r in results}
    assert rel_paths == {"00-Inbox/My-New-Idea.md", "00-Inbox/My-New-Idea-2.md"}
    assert vault.exists("_staging/00-Inbox/My-New-Idea.md")
    assert vault.exists("_staging/00-Inbox/My-New-Idea-2.md")


def test_collision_check_at_vault_root(vault: VaultIO, vector_store: VectorStore) -> None:
    daily_content = "# Idea One\nFirst distinct idea.\n\n# Idea Two\nSecond distinct idea."
    vault.write_note("01-Daily/07-16-2026.md", Note(metadata={}, content=daily_content))
    router = FakeRouter({"daily_digestion": "My New Idea"})
    embedder = VaryingEmbedder(
        {
            "# Idea One\nFirst distinct idea.": [1.0, 0.0, 0.0, 0.0],
            "# Idea Two\nSecond distinct idea.": [0.0, 1.0, 0.0, 0.0],
        }
    )
    digestor = Digestor(router, embedder, vector_store, vault, dry_run=False, default_folder="")

    results = digestor.digest("01-Daily/07-16-2026.md", "07-16-2026")

    rel_paths = {r.rel_path for r in results}
    assert rel_paths == {"My-New-Idea.md", "My-New-Idea-2.md"}
