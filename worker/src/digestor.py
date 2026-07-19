"""Digestor module: splits a daily note into atomic chunks and merges/creates notes (requirement 2)."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .llm_router import Router
from .vault_io import Note, VaultIO
from .vector_store import EmbeddedChunk, Embedder, SimilarityMatch, VectorStore

# Placeholder confidence bar on LanceDB L2 distance; not yet calibrated against a real vault.
MERGE_DISTANCE_THRESHOLD = 0.25
HEADING_RE = re.compile(r"^#{1,6}\s+.+$")

DIGESTION_SYSTEM_PROMPT = (
    "You title atomic notes distilled from a personal journal entry. Read the fragment and "
    "respond with only a short, descriptive title in plain text - no quotes, no trailing "
    "punctuation, no explanation."
)


@dataclass(frozen=True)
class DigestedChunk:
    """One atomic idea distilled from a daily note, and where it landed."""

    title: str
    content: str
    rel_path: str
    merged_into_existing: bool


class Digestor:
    """Splits a daily note into atomic chunks, merging into existing notes or creating new ones."""

    def __init__(
        self,
        router: Router,
        embedder: Embedder,
        vector_store: VectorStore,
        vault: VaultIO,
        dry_run: bool,
        default_folder: str = "",
    ) -> None:
        self._router = router
        self._embedder = embedder
        self._vector_store = vector_store
        self._vault = vault
        self._dry_run = dry_run
        self._default_folder = default_folder

    def digest(self, daily_note_rel_path: str, daily_note_date: str) -> list[DigestedChunk]:
        """Read, segment, and route every atomic idea in a daily note; updates the daily note's summary."""
        daily_note = self._vault.read_note(daily_note_rel_path)
        chunks = self._segment(daily_note.content)

        results = [self._route_chunk(chunk_text, daily_note_date) for chunk_text in chunks]

        if results:
            self._update_daily_note_summary(daily_note_rel_path, daily_note, results)

        return results

    @staticmethod
    def _segment(text: str) -> list[str]:
        """Split into atomic chunks by heading; falls back to paragraph breaks for unstructured entries."""
        lines = text.splitlines()
        if any(HEADING_RE.match(line) for line in lines):
            chunks: list[str] = []
            current: list[str] = []
            for line in lines:
                if HEADING_RE.match(line) and current:
                    chunks.append("\n".join(current).strip())
                    current = [line]
                else:
                    current.append(line)
            if current:
                chunks.append("\n".join(current).strip())
        else:
            chunks = [p.strip() for p in text.split("\n\n")]
        return [c for c in chunks if c]

    def _route_chunk(self, chunk_text: str, daily_note_date: str) -> DigestedChunk:
        """Embed a chunk and either merge it into a close existing note or create a new one."""
        vector = self._embedder.embed(chunk_text)
        match = self._closest_match(vector)
        if match is not None:
            return self._merge_into_existing(chunk_text, match, daily_note_date, vector)
        return self._create_new_note(chunk_text, daily_note_date, vector)

    def _closest_match(self, vector: list[float]) -> Optional[SimilarityMatch]:
        """Find the single closest existing chunk, if it clears the merge-confidence bar."""
        results = self._vector_store.search(vector, limit=1)
        if results and results[0].distance <= MERGE_DISTANCE_THRESHOLD:
            return results[0]
        return None

    def _merge_into_existing(
        self, chunk_text: str, match: SimilarityMatch, daily_note_date: str, vector: list[float]
    ) -> DigestedChunk:
        """Append a chunk to an existing note, with a link back to the daily note it came from."""
        existing = self._vault.read_note(match.path)
        merged_content = f"{existing.content.rstrip()}\n\n---\n\n{chunk_text}\n\nSource: [[{daily_note_date}]]"
        merged_note = Note(metadata=existing.metadata, content=merged_content)
        self._vault.write_or_stage(match.path, merged_note, dry_run=self._dry_run)
        self._vector_store.upsert(
            EmbeddedChunk(id=match.path, text=merged_content, vector=vector, path=match.path, note_title=match.note_title)
        )
        return DigestedChunk(title=match.note_title, content=chunk_text, rel_path=match.path, merged_into_existing=True)

    def _create_new_note(self, chunk_text: str, daily_note_date: str, vector: list[float]) -> DigestedChunk:
        """Create a new atomic note carrying the standard source/created/tags/status frontmatter."""
        title = self._propose_title(chunk_text)
        filename = f"{self._slugify(title)}.md"
        target = f"{self._default_folder}/{filename}" if self._default_folder else filename
        rel_path = self._available_path(target)
        note = Note(
            metadata={
                "title": title,
                "created": daily_note_date,
                "tags": [],
                "source": f"[[{daily_note_date}]]",
                "status": "draft",
            },
            content=chunk_text,
        )
        self._vault.write_or_stage(rel_path, note, dry_run=self._dry_run)
        self._vector_store.upsert(
            EmbeddedChunk(id=rel_path, text=chunk_text, vector=vector, path=rel_path, note_title=title)
        )
        return DigestedChunk(title=title, content=chunk_text, rel_path=rel_path, merged_into_existing=False)

    def _propose_title(self, chunk_text: str) -> str:
        """Ask the daily_digestion model for a short descriptive title for one atomic chunk."""
        response = self._router.generate("daily_digestion", system=DIGESTION_SYSTEM_PROMPT, prompt=chunk_text)
        return response.text.strip().strip('"').strip() or "Untitled Note"

    def _available_path(self, rel_path: str) -> str:
        """Append a numeric suffix if rel_path is already taken, so same-titled chunks never collide.

        Two chunks landing on the same slug (e.g. the model returns "Untitled Note" for both)
        would otherwise silently overwrite each other - the second write clobbering the first
        with no error and no trace in the report.
        """
        candidate = Path(rel_path)
        target = candidate
        counter = 2
        while self._target_exists(str(target)):
            target = candidate.with_name(f"{candidate.stem}-{counter}{candidate.suffix}")
            counter += 1
        return str(target)

    def _target_exists(self, rel_path: str) -> bool:
        """Check the path this run would actually write to, respecting dry-run staging."""
        if self._dry_run:
            return self._vault.exists(str(Path("_staging") / rel_path))
        return self._vault.exists(rel_path)

    @staticmethod
    def _slugify(title: str) -> str:
        """Turn a proposed title into a filesystem-safe filename stem."""
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-")
        return slug or "untitled-note"

    def _update_daily_note_summary(self, daily_note_rel_path: str, daily_note: Note, results: list[DigestedChunk]) -> None:
        """Append a 'Notes generated' map-of-content section to the daily note."""
        lines = ["", "## Notes generated", ""]
        for result in results:
            verb = "merged into" if result.merged_into_existing else "created"
            lines.append(f"- [[{result.title}]] ({verb})")
        updated_content = daily_note.content.rstrip() + "\n" + "\n".join(lines) + "\n"
        self._vault.write_or_stage(
            daily_note_rel_path, Note(metadata=daily_note.metadata, content=updated_content), dry_run=self._dry_run
        )
