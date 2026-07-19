"""Vault I/O module: atomic writes, path containment, and frontmatter parsing for vault notes."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Union

import frontmatter


class PathEscapesVaultError(Exception):
    """Raised when a resolved path would fall outside the vault root."""


@dataclass(frozen=True)
class Note:
    """A parsed Markdown note: YAML frontmatter metadata plus body content."""

    metadata: dict[str, Any]
    content: str


class VaultIO:
    """Safe, atomic read/write access to Markdown notes rooted at a single vault directory."""

    def __init__(self, vault_root: Path) -> None:
        self._vault_root = vault_root.resolve()
        if not self._vault_root.is_dir():
            raise FileNotFoundError(f"vault root does not exist or is not a directory: {self._vault_root}")

    def resolve(self, rel_path: Union[str, Path]) -> Path:
        """Resolve a vault-relative path and verify the result stays inside the vault root.

        Joining an absolute rel_path onto vault_root silently discards the vault_root prefix
        (a pathlib quirk), so containment is checked on the resolved result, not assumed
        from the input shape.
        """
        candidate = (self._vault_root / rel_path).resolve()
        if candidate != self._vault_root and self._vault_root not in candidate.parents:
            raise PathEscapesVaultError(f"path escapes vault root: {rel_path}")
        return candidate

    def exists(self, rel_path: Union[str, Path]) -> bool:
        """Check whether a vault-relative path exists, without raising on an escaping path."""
        try:
            return self.resolve(rel_path).is_file()
        except PathEscapesVaultError:
            return False

    def read_note(self, rel_path: Union[str, Path]) -> Note:
        """Read and parse a note's frontmatter and body."""
        full_path = self.resolve(rel_path)
        post = frontmatter.load(full_path)
        return Note(metadata=dict(post.metadata), content=post.content)

    def write_note(self, rel_path: Union[str, Path], note: Note) -> None:
        """Atomically write a note's frontmatter and body, creating parent folders as needed.

        A note with no metadata is written as plain content with no frontmatter block:
        python-frontmatter's dumps() otherwise renders an empty mapping as a literal '{}'
        between the delimiters, which is valid YAML but reads as a bug to a human in Obsidian.
        """
        full_path = self.resolve(rel_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        if note.metadata:
            post = frontmatter.Post(note.content, **note.metadata)
            serialized = frontmatter.dumps(post)
        else:
            serialized = note.content
        self._atomic_write(full_path, serialized)

    def stage_note(self, rel_path: Union[str, Path], note: Note, staging_folder: str = "_staging") -> Path:
        """Write a proposed note under _staging/ instead of the live vault path, for dry-run review."""
        staged_rel = Path(staging_folder) / rel_path
        self.write_note(staged_rel, note)
        return self.resolve(staged_rel)

    def write_or_stage(self, rel_path: Union[str, Path], note: Note, dry_run: bool, staging_folder: str = "_staging") -> Path:
        """Write live when dry_run is False, otherwise write the same content as a staged proposal."""
        if dry_run:
            return self.stage_note(rel_path, note, staging_folder=staging_folder)
        self.write_note(rel_path, note)
        return self.resolve(rel_path)

    def move_note(
        self,
        old_rel_path: Union[str, Path],
        new_rel_path: Union[str, Path],
        note: Note,
        dry_run: bool,
        staging_folder: str = "_staging",
    ) -> Path:
        """Relocate a note to a new vault path.

        In dry-run mode this only stages the proposed destination and leaves the original in
        place for review. In apply mode it writes the new location and removes the old file -
        the note's content is never lost, only relocated, and the whole thing is one git-visible
        change (a rename), not a silent deletion.
        """
        if dry_run:
            return self.stage_note(new_rel_path, note, staging_folder=staging_folder)
        self.write_note(new_rel_path, note)
        old_full = self.resolve(old_rel_path)
        if old_full.exists():
            old_full.unlink()
        return self.resolve(new_rel_path)

    def read_raw(self, rel_path: Union[str, Path]) -> str:
        """Read a vault-relative file as plain text, with no frontmatter parsing."""
        return self.resolve(rel_path).read_text(encoding="utf-8")

    def write_raw(self, rel_path: Union[str, Path], content: str) -> None:
        """Atomically write plain text (no frontmatter) to a vault-relative path, always live.

        Used for pipeline-generated artifacts (AGENTS.md, reports) that aren't user notes and
        so aren't subject to the dry-run staging policy.
        """
        full_path = self.resolve(rel_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(full_path, content)

    def iter_notes(self, excluded_folders: frozenset[str] = frozenset()) -> Iterator[tuple[str, Note]]:
        """Yield (rel_path, Note) for every Markdown note in the vault, skipping excluded top-level folders."""
        for full_path in sorted(self._vault_root.rglob("*.md")):
            rel_path = full_path.relative_to(self._vault_root).as_posix()
            if excluded_folders.intersection(Path(rel_path).parts):
                continue
            try:
                yield rel_path, self.read_note(rel_path)
            except (OSError, UnicodeDecodeError):
                continue

    def _atomic_write(self, full_path: Path, content: str) -> None:
        """Write content to a temp file in the same directory, then atomically replace the target."""
        tmp_path = full_path.with_name(full_path.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, full_path)
