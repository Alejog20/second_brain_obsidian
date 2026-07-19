"""Worktree module: isolates the nightly review branch from the vault the user has open."""

from pathlib import Path

from git import Repo


class NightlyWorktree:
    """Maintains a disposable git worktree on a dedicated branch, rebuilt fresh from the base branch each run.

    The vault's primary checkout (what Obsidian has open) is never touched - this opens a
    second, independent checkout of the same repo at a different path, on a different branch.
    """

    def __init__(self, vault_root: Path, worktree_path: Path, branch: str, base_branch: str, remote: str = "origin") -> None:
        self._vault_root = vault_root
        self._worktree_path = worktree_path
        self._branch = branch
        self._base_branch = base_branch
        self._remote = remote
        self._repo = Repo(vault_root)

    def sync(self) -> Path:
        """Fetch the remote, then (re)create the worktree fresh from the base branch's current tip."""
        self._repo.remotes[self._remote].fetch()
        if self._worktree_path.exists():
            self._repo.git.worktree("remove", "--force", str(self._worktree_path))
        self._worktree_path.parent.mkdir(parents=True, exist_ok=True)
        self._repo.git.worktree(
            "add", "-B", self._branch, str(self._worktree_path), f"{self._remote}/{self._base_branch}"
        )
        return self._worktree_path

    def push(self) -> None:
        """Force-with-lease push the worktree's branch, since it's always fully rebuilt from base_branch."""
        worktree_repo = Repo(self._worktree_path)
        worktree_repo.git.push(self._remote, self._branch, force_with_lease=True, set_upstream=True)
