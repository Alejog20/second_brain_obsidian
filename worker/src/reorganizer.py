"""Reorganizer module: per-note title, grammar/clarity, taxonomy, and link pipeline (requirement 1)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .llm_router import Router
from .vault_io import Note
from .vector_store import EmbeddedChunk, Embedder, VectorStore

GENERIC_TITLES = {"untitled", "note", "new note", "daily note", ""}
SEARCH_LIMIT = 5
# Placeholder confidence bars on LanceDB L2 distance; not yet calibrated against a real vault.
TAXONOMY_CONFIDENCE_DISTANCE = 0.35
LINK_SUGGESTION_DISTANCE = 0.45

GRAMMAR_SYSTEM_PROMPT = (
    "You copyedit personal notes. Fix spelling and grammar only - preserve the author's voice, "
    "structure, and meaning exactly. Never add content that isn't already implied by the text.\n"
    "Respond in exactly this format:\n"
    "---CORRECTED---\n<corrected note text>\n---CLARITY---\nclear\n"
    "or, if the note's core idea doesn't come through clearly:\n"
    "---CORRECTED---\n<corrected note text>\n---CLARITY---\nunclear: <one short reason>"
)

TITLE_SYSTEM_PROMPT = (
    "You title personal knowledge-base notes. Read the note body and respond with only a short, "
    "descriptive title in plain text - no quotes, no trailing punctuation, no explanation."
)


@dataclass(frozen=True)
class ReorganizeResult:
    """Proposed edits and review flags for a single note; nothing here has touched the vault yet."""

    title: str
    content: str
    suggested_folder: Optional[str] = None
    suggested_links: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()


class Reorganizer:
    """Computes title/grammar/taxonomy/link proposals for one note; never writes to the vault."""

    def __init__(self, router: Router, embedder: Embedder, vector_store: VectorStore) -> None:
        self._router = router
        self._embedder = embedder
        self._vector_store = vector_store

    def reorganize(self, rel_path: str, note: Note) -> ReorganizeResult:
        """Run the full pipeline for one note and return proposed changes."""
        flags: list[str] = []

        title = self._improve_title(note.metadata.get("title"), note.content)
        content, is_clear, clarity_reason = self._improve_grammar(note.content)
        if not is_clear:
            flags.append(f"low_clarity: {clarity_reason}" if clarity_reason else "low_clarity")

        if not content.strip():
            # Nothing meaningful to embed - an empty/whitespace-only note has no taxonomy or
            # link signal, and some embedding providers return an empty vector for empty input,
            # which would otherwise crash the vector-store search several layers down.
            return ReorganizeResult(title=title, content=content, flags=tuple(flags))

        vector = self._embedder.embed(content)
        folder = self._suggest_taxonomy(rel_path, vector)
        links = self._suggest_links(rel_path, vector)

        self._vector_store.upsert(
            EmbeddedChunk(id=rel_path, text=content, vector=vector, path=rel_path, note_title=title)
        )

        return ReorganizeResult(
            title=title,
            content=content,
            suggested_folder=folder,
            suggested_links=tuple(links),
            flags=tuple(flags),
        )

    def _improve_title(self, current_title: Optional[str], content: str) -> str:
        """Propose a better title only if the current one is missing or generic."""
        if current_title and current_title.strip().lower() not in GENERIC_TITLES:
            return current_title
        response = self._router.generate("title_and_tagging", system=TITLE_SYSTEM_PROMPT, prompt=content)
        proposed = response.text.strip().strip('"').strip()
        return proposed or (current_title or "Untitled")

    def _improve_grammar(self, content: str) -> tuple[str, bool, Optional[str]]:
        """Fix genuine grammar errors without rewriting voice; flag (don't fix) unclear notes."""
        response = self._router.generate("bulk_grammar_pass", system=GRAMMAR_SYSTEM_PROMPT, prompt=content)
        return self._parse_grammar_response(response.text, fallback=content)

    @staticmethod
    def _parse_grammar_response(raw: str, fallback: str) -> tuple[str, bool, Optional[str]]:
        """Parse the corrected-text/clarity-verdict format; fail safe to the original text if malformed."""
        if "---CORRECTED---" not in raw or "---CLARITY---" not in raw:
            return fallback, False, "grammar_pass_response_unparseable"
        _, _, rest = raw.partition("---CORRECTED---")
        corrected, _, clarity_block = rest.partition("---CLARITY---")
        corrected = corrected.strip()
        clarity_block = clarity_block.strip().lower()
        if not corrected:
            return fallback, False, "grammar_pass_response_unparseable"
        if clarity_block.startswith("clear"):
            return corrected, True, None
        reason = clarity_block[len("unclear:"):].strip() if clarity_block.startswith("unclear:") else None
        return corrected, False, reason or "unclear"

    def _suggest_taxonomy(self, rel_path: str, vector: list[float]) -> Optional[str]:
        """Suggest a different top-level folder only when a close neighbor already lives in one.

        A note's own filename is not a folder: for a root-level file, Path(path).parts has only
        one element (the filename itself), which _top_level_folder deliberately treats as "no
        folder" rather than degenerating into it - otherwise this could suggest moving a note
        "into" another note's filename as if it were a directory.
        """
        current_folder = self._top_level_folder(rel_path)
        for match in self._vector_store.search(vector, limit=SEARCH_LIMIT):
            if match.path == rel_path:
                continue
            neighbor_folder = self._top_level_folder(match.path)
            if neighbor_folder and neighbor_folder != current_folder and match.distance <= TAXONOMY_CONFIDENCE_DISTANCE:
                return neighbor_folder
        return None

    @staticmethod
    def _top_level_folder(rel_path: str) -> Optional[str]:
        """The note's top-level folder, or None if it lives at the vault root (no real folder)."""
        parts = Path(rel_path).parts
        return parts[0] if len(parts) > 1 else None

    def _suggest_links(self, rel_path: str, vector: list[float]) -> list[str]:
        """Propose wikilinks to semantically related notes; never inserted into content automatically."""
        return [
            match.note_title
            for match in self._vector_store.search(vector, limit=SEARCH_LIMIT)
            if match.path != rel_path and match.distance <= LINK_SUGGESTION_DISTANCE
        ]
