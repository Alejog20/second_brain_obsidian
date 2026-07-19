"""Git safety module: commit-before/after bracket for any apply-mode vault mutation."""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from git import InvalidGitRepositoryError, Repo

logger = logging.getLogger(__name__)


class VaultNotAGitRepoError(Exception):
    """Raised when safety.require_git is true but the vault isn't a git repository."""


class GitSafety:
    """Ensures every apply-mode vault mutation is bracketed by a git commit, for instant rollback."""

    def __init__(self, vault_root: Path, require_git: bool = True) -> None:
        self._vault_root = vault_root
        self._require_git = require_git
        self._repo: Optional[Repo] = self._open_repo()

    def _open_repo(self) -> Optional[Repo]:
        """Open the vault as a git repo, or return None if require_git is false and it isn't one."""
        try:
            return Repo(self._vault_root)
        except InvalidGitRepositoryError:
            if self._require_git:
                raise VaultNotAGitRepoError(
                    f"vault at {self._vault_root} is not a git repository and safety.require_git is true"
                ) from None
            return None

    def has_uncommitted_changes(self) -> bool:
        """Check whether the vault has staged, unstaged, or untracked changes right now."""
        if self._repo is None:
            return False
        return self._repo.is_dirty(untracked_files=True)

    def commit(self, message: str) -> Optional[str]:
        """Stage all changes and commit them; returns the new commit hexsha, or None if nothing changed."""
        if self._repo is None or not self.has_uncommitted_changes():
            return None
        self._repo.git.add(all=True)
        commit = self._repo.index.commit(message)
        logger.info("Committed %s: %s", commit.hexsha[:8], message)
        return commit.hexsha

    @contextmanager
    def bracket(self, run_label: str) -> Iterator[None]:
        """Commit before and after the wrapped block, so any run is a single, revertible unit."""
        self.commit(f"second-brain: pre-run snapshot before {run_label}")
        try:
            yield
        finally:
            self.commit(f"second-brain: {run_label}")
