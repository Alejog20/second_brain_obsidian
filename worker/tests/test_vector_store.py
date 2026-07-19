"""Tests for the embedding client and LanceDB-backed vector store."""

from pathlib import Path
from typing import Any

import httpx
import pytest

from src.vector_store import EmbeddedChunk, OllamaEmbeddingClient, VectorStore


def test_ollama_embedding_client_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"embedding": [0.1, 0.2, 0.3]},
            request=httpx.Request("POST", "http://localhost:11434/api/embeddings"),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OllamaEmbeddingClient()
    assert client.embed("some note text") == [0.1, 0.2, 0.3]


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "vector_store", embedding_dim=4)


def test_upsert_and_search_returns_closest_match(store: VectorStore) -> None:
    store.upsert(EmbeddedChunk(id="a", text="docker networking basics", vector=[1.0, 0.0, 0.0, 0.0], path="p1.md", note_title="Docker Networking"))
    store.upsert(EmbeddedChunk(id="b", text="unrelated cooking recipe", vector=[0.0, 1.0, 0.0, 0.0], path="p2.md", note_title="Recipe"))

    matches = store.search([1.0, 0.0, 0.0, 0.0], limit=1)

    assert len(matches) == 1
    assert matches[0].chunk_id == "a"
    assert matches[0].note_title == "Docker Networking"


def test_upsert_replaces_existing_chunk(store: VectorStore) -> None:
    store.upsert(EmbeddedChunk(id="a", text="version 1", vector=[1.0, 0.0, 0.0, 0.0], path="p1.md", note_title="T"))
    store.upsert(EmbeddedChunk(id="a", text="version 2", vector=[1.0, 0.0, 0.0, 0.0], path="p1.md", note_title="T"))

    assert store.count() == 1
    matches = store.search([1.0, 0.0, 0.0, 0.0], limit=5)
    assert matches[0].text == "version 2"


def test_delete_by_path_removes_all_matching_chunks(store: VectorStore) -> None:
    store.upsert(EmbeddedChunk(id="a", text="chunk 1", vector=[1.0, 0.0, 0.0, 0.0], path="p1.md", note_title="T"))
    store.upsert(EmbeddedChunk(id="b", text="chunk 2", vector=[0.0, 1.0, 0.0, 0.0], path="p1.md", note_title="T"))
    store.upsert(EmbeddedChunk(id="c", text="chunk 3", vector=[0.0, 0.0, 1.0, 0.0], path="p2.md", note_title="T2"))

    store.delete_by_path("p1.md")

    assert store.count() == 1


def test_reopening_existing_store_preserves_data(tmp_path: Path) -> None:
    path = tmp_path / "vector_store"
    first = VectorStore(path, embedding_dim=4)
    first.upsert(EmbeddedChunk(id="a", text="persisted", vector=[1.0, 0.0, 0.0, 0.0], path="p1.md", note_title="T"))

    second = VectorStore(path, embedding_dim=4)
    assert second.count() == 1


def test_id_with_quote_does_not_break_upsert(store: VectorStore) -> None:
    chunk = EmbeddedChunk(id="o'brien", text="edge case id", vector=[1.0, 0.0, 0.0, 0.0], path="p1.md", note_title="T")
    store.upsert(chunk)
    store.upsert(chunk)  # replace path must survive the quote in the id
    assert store.count() == 1
