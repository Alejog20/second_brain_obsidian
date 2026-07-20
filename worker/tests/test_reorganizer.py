"""Tests for the reorganizer: title, grammar/clarity, taxonomy, and link suggestions."""

from pathlib import Path

import pytest

from src.llm_router import LLMResponse
from src.reorganizer import Reorganizer
from src.vault_io import Note
from src.vector_store import EmbeddedChunk, VectorStore

CLEAR_GRAMMAR_RESPONSE = "---CORRECTED---\nFixed text.\n---CLARITY---\nclear"
UNCLEAR_GRAMMAR_RESPONSE = "---CORRECTED---\nFixed text.\n---CLARITY---\nunclear: rambling, no conclusion"
MALFORMED_GRAMMAR_RESPONSE = "I fixed it for you!"


class FakeRouter:
    """Duck-types the Router protocol; returns canned text per task_key, tracks calls."""

    def __init__(self, responses: dict[str, str]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def generate(self, task_key: str, system: str, prompt: str) -> LLMResponse:
        self.calls.append(task_key)
        return LLMResponse(text=self._responses.get(task_key, ""), tokens_in=1, tokens_out=1, cost_usd=0.0)


class FakeEmbedder:
    """Returns a fixed vector regardless of input, for deterministic vector-store tests."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def embed(self, text: str) -> list[float]:
        return self._vector


@pytest.fixture
def vector_store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "vector_store", embedding_dim=4)


def test_title_kept_when_already_descriptive(vector_store: VectorStore) -> None:
    router = FakeRouter({"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE})
    reorganizer = Reorganizer(router, FakeEmbedder([0.0, 0.0, 0.0, 0.0]), vector_store)

    result = reorganizer.reorganize("04-Reference/docker.md", Note(metadata={"title": "Docker Networking Basics"}, content="body"))

    assert result.title == "Docker Networking Basics"
    assert "title_and_tagging" not in router.calls


def test_title_proposed_when_missing(vector_store: VectorStore) -> None:
    router = FakeRouter({"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE, "title_and_tagging": "Docker Networking Basics"})
    reorganizer = Reorganizer(router, FakeEmbedder([0.0, 0.0, 0.0, 0.0]), vector_store)

    result = reorganizer.reorganize("00-Inbox/note.md", Note(metadata={}, content="body"))

    assert result.title == "Docker Networking Basics"
    assert "title_and_tagging" in router.calls


def test_title_proposed_when_generic(vector_store: VectorStore) -> None:
    router = FakeRouter({"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE, "title_and_tagging": "Better Title"})
    reorganizer = Reorganizer(router, FakeEmbedder([0.0, 0.0, 0.0, 0.0]), vector_store)

    result = reorganizer.reorganize("00-Inbox/note.md", Note(metadata={"title": "Untitled"}, content="body"))

    assert result.title == "Better Title"


def test_clear_grammar_pass_has_no_flags(vector_store: VectorStore) -> None:
    router = FakeRouter({"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE})
    reorganizer = Reorganizer(router, FakeEmbedder([0.0, 0.0, 0.0, 0.0]), vector_store)

    result = reorganizer.reorganize("note.md", Note(metadata={"title": "T"}, content="teh cat sat"))

    assert result.content == "Fixed text."
    assert result.flags == ()


def test_unclear_grammar_pass_flags_and_keeps_correction(vector_store: VectorStore) -> None:
    router = FakeRouter({"bulk_grammar_pass": UNCLEAR_GRAMMAR_RESPONSE})
    reorganizer = Reorganizer(router, FakeEmbedder([0.0, 0.0, 0.0, 0.0]), vector_store)

    result = reorganizer.reorganize("note.md", Note(metadata={"title": "T"}, content="rambly text"))

    assert result.content == "Fixed text."
    assert len(result.flags) == 1
    assert "rambling, no conclusion" in result.flags[0]


def test_malformed_grammar_response_falls_back_to_original_and_flags(vector_store: VectorStore) -> None:
    router = FakeRouter({"bulk_grammar_pass": MALFORMED_GRAMMAR_RESPONSE})
    reorganizer = Reorganizer(router, FakeEmbedder([0.0, 0.0, 0.0, 0.0]), vector_store)

    result = reorganizer.reorganize("note.md", Note(metadata={"title": "T"}, content="original text"))

    assert result.content == "original text"
    assert result.flags == ("low_clarity: grammar_pass_response_unparseable",)


def test_taxonomy_suggested_when_close_neighbor_in_other_folder(vector_store: VectorStore) -> None:
    vector_store.upsert(
        EmbeddedChunk(id="02-Areas/security.md", text="security notes", vector=[1.0, 0.0, 0.0, 0.0], path="02-Areas/security.md", note_title="Security")
    )
    router = FakeRouter({"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE})
    reorganizer = Reorganizer(router, FakeEmbedder([1.0, 0.0, 0.0, 0.0]), vector_store)

    result = reorganizer.reorganize("00-Inbox/note.md", Note(metadata={"title": "T"}, content="body"))

    assert result.suggested_folder == "02-Areas"


def test_no_taxonomy_suggestion_when_neighbor_in_same_folder(vector_store: VectorStore) -> None:
    vector_store.upsert(
        EmbeddedChunk(id="00-Inbox/other.md", text="x", vector=[1.0, 0.0, 0.0, 0.0], path="00-Inbox/other.md", note_title="Other")
    )
    router = FakeRouter({"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE})
    reorganizer = Reorganizer(router, FakeEmbedder([1.0, 0.0, 0.0, 0.0]), vector_store)

    result = reorganizer.reorganize("00-Inbox/note.md", Note(metadata={"title": "T"}, content="body"))

    assert result.suggested_folder is None


def test_no_taxonomy_suggestion_when_current_note_is_at_root(vector_store: VectorStore) -> None:
    """A root-level note being reorganized has no real "current folder" - a close neighbor that
    lives in a real folder should still be suggested (this is the normal, correct case)."""
    vector_store.upsert(
        EmbeddedChunk(id="02-Areas/security.md", text="x", vector=[1.0, 0.0, 0.0, 0.0], path="02-Areas/security.md", note_title="Security")
    )
    router = FakeRouter({"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE})
    reorganizer = Reorganizer(router, FakeEmbedder([1.0, 0.0, 0.0, 0.0]), vector_store)

    result = reorganizer.reorganize("note-at-root.md", Note(metadata={"title": "T"}, content="body"))

    assert result.suggested_folder == "02-Areas"


def test_no_taxonomy_suggestion_when_neighbor_is_also_at_root(vector_store: VectorStore) -> None:
    """Regression test: a root-level neighbor's filename must never be suggested as a folder.

    Path("neighbor-at-root.md").parts[0] degenerates to the filename itself for a single-part
    path - treating that as a real folder would make move_note() try to create a note "inside"
    another note's filename, which crashes (or worse, silently misfiles content) since that
    path segment is actually a file, not a directory. This happened for real against an actual
    vault where daily notes sit at the vault root alongside regular notes.
    """
    vector_store.upsert(
        EmbeddedChunk(id="neighbor-at-root.md", text="x", vector=[1.0, 0.0, 0.0, 0.0], path="neighbor-at-root.md", note_title="Neighbor")
    )
    router = FakeRouter({"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE})
    reorganizer = Reorganizer(router, FakeEmbedder([1.0, 0.0, 0.0, 0.0]), vector_store)

    result = reorganizer.reorganize("00-Inbox/note.md", Note(metadata={"title": "T"}, content="body"))

    assert result.suggested_folder is None


def test_link_suggested_for_moderately_close_neighbor(vector_store: VectorStore) -> None:
    vector_store.upsert(
        EmbeddedChunk(id="04-Reference/related.md", text="x", vector=[0.9, 0.1, 0.0, 0.0], path="04-Reference/related.md", note_title="Related Note")
    )
    router = FakeRouter({"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE})
    reorganizer = Reorganizer(router, FakeEmbedder([1.0, 0.0, 0.0, 0.0]), vector_store)

    result = reorganizer.reorganize("00-Inbox/note.md", Note(metadata={"title": "T"}, content="body"))

    assert "Related Note" in result.suggested_links


def test_reorganize_upserts_self_into_vector_store(vector_store: VectorStore) -> None:
    router = FakeRouter({"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE})
    reorganizer = Reorganizer(router, FakeEmbedder([1.0, 0.0, 0.0, 0.0]), vector_store)

    reorganizer.reorganize("00-Inbox/note.md", Note(metadata={"title": "T"}, content="body"))

    assert vector_store.count() == 1


class ExplodingEmbedder:
    """Raises if embed() is ever called - proves empty-content notes never reach the embedder."""

    def embed(self, text: str) -> list[float]:
        raise AssertionError("embed() should not be called for empty/whitespace-only content")


def test_empty_content_after_grammar_pass_skips_embedding(vector_store: VectorStore) -> None:
    """Regression test: a real Ollama embedding call for empty text can return an empty vector,
    which crashes LanceDB several layers down (IndexError on vector[0]) - so empty/whitespace
    content must never reach the embedder at all."""
    router = FakeRouter({"bulk_grammar_pass": "---CORRECTED---\n   \n---CLARITY---\nclear"})
    reorganizer = Reorganizer(router, ExplodingEmbedder(), vector_store)

    result = reorganizer.reorganize("00-Inbox/blank.md", Note(metadata={"title": "T"}, content=""))

    assert result.content.strip() == ""
    assert result.suggested_folder is None
    assert result.suggested_links == ()
    assert vector_store.count() == 0
