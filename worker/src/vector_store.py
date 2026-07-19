"""Vector store module: local embeddings (Ollama) backed by an embedded LanceDB table."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx
import lancedb
import pyarrow as pa

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class Embedder(Protocol):
    """Structural interface satisfied by OllamaEmbeddingClient; lets pipeline modules take a fake in tests."""

    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for a piece of text."""
        ...


@dataclass(frozen=True)
class EmbeddedChunk:
    """A vault text chunk paired with its embedding vector and source metadata."""

    id: str
    text: str
    vector: list[float]
    path: str
    note_title: str


@dataclass(frozen=True)
class SimilarityMatch:
    """A vector-store search hit: the matched chunk and its similarity distance."""

    chunk_id: str
    path: str
    note_title: str
    text: str
    distance: float


class OllamaEmbeddingClient:
    """Generates text embeddings via a local Ollama server."""

    def __init__(self, model: str = "nomic-embed-text", host: str = DEFAULT_OLLAMA_HOST, timeout: float = 30.0) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout

    def embed(self, text: str) -> list[float]:
        """Request an embedding vector for a single piece of text."""
        response = httpx.post(
            f"{self._host}/api/embeddings",
            json={"model": self._model, "prompt": text},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()["embedding"]


def _escape_literal(value: str) -> str:
    """Escape a string for safe interpolation into a LanceDB predicate (no bind-param API exists)."""
    return value.replace("'", "''")


class VectorStore:
    """Embedded LanceDB table storing note-chunk embeddings for similarity search."""

    TABLE_NAME = "note_chunks"

    def __init__(self, store_path: Path, embedding_dim: int = 768) -> None:
        store_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(store_path))
        self._table = self._db.create_table(self.TABLE_NAME, schema=self._schema(embedding_dim), exist_ok=True)

    @staticmethod
    def _schema(embedding_dim: int) -> pa.Schema:
        """The fixed Arrow schema for the note-chunks table."""
        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), embedding_dim)),
                pa.field("path", pa.string()),
                pa.field("note_title", pa.string()),
            ]
        )

    def upsert(self, chunk: EmbeddedChunk) -> None:
        """Insert or replace a chunk's embedding, keyed by chunk id."""
        self._table.delete(f"id = '{_escape_literal(chunk.id)}'")
        self._table.add(
            [
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "vector": chunk.vector,
                    "path": chunk.path,
                    "note_title": chunk.note_title,
                }
            ]
        )

    def delete_by_path(self, path: str) -> None:
        """Remove all chunks belonging to a note path, e.g. before re-embedding an edited note."""
        self._table.delete(f"path = '{_escape_literal(path)}'")

    def search(self, query_vector: list[float], limit: int = 5) -> list[SimilarityMatch]:
        """Find the closest chunks to a query embedding."""
        results = self._table.search(query_vector).limit(limit).to_list()
        return [
            SimilarityMatch(
                chunk_id=row["id"],
                path=row["path"],
                note_title=row["note_title"],
                text=row["text"],
                distance=row["_distance"],
            )
            for row in results
        ]

    def count(self) -> int:
        """Number of chunks currently stored."""
        return self._table.count_rows()
