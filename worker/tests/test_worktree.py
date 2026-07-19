"""Tests for the nightly review-branch worktree, against a local bare repo (no GitHub needed)."""

from pathlib import Path

import pytest
from git import Repo

from src.worktree import NightlyWorktree


@pytest.fixture
def remote_repo(tmp_path: Path) -> Path:
    """A bare repo standing in for the GitHub remote."""
    remote_path = tmp_path / "remote.git"
    Repo.init(remote_path, bare=True)
    return remote_path


@pytest.fixture
def vault_repo(tmp_path: Path, remote_repo: Path) -> Path:
    """A clone of the bare remote, standing in for the vault's live checkout, with one commit on main."""
    vault_path = tmp_path / "vault"
    repo = Repo.clone_from(str(remote_repo), str(vault_path))
    repo.git.checkout("-B", "main")
    (vault_path / "note.md").write_text("hello", encoding="utf-8")
    repo.git.add(all=True)
    repo.index.commit("initial commit")
    repo.git.push("origin", "main", set_upstream=True)
    return vault_path


def _worktree(tmp_path: Path, vault_repo: Path) -> NightlyWorktree:
    return NightlyWorktree(
        vault_root=vault_repo,
        worktree_path=tmp_path / "worktree",
        branch="second-brain/nightly",
        base_branch="main",
    )


def test_sync_creates_worktree_on_dedicated_branch_with_base_content(tmp_path: Path, vault_repo: Path) -> None:
    nw = _worktree(tmp_path, vault_repo)

    result_path = nw.sync()

    assert result_path == tmp_path / "worktree"
    assert (result_path / "note.md").read_text(encoding="utf-8") == "hello"
    assert Repo(result_path).active_branch.name == "second-brain/nightly"


def test_vault_live_checkout_is_never_touched(tmp_path: Path, vault_repo: Path) -> None:
    nw = _worktree(tmp_path, vault_repo)

    nw.sync()

    vault = Repo(vault_repo)
    assert vault.active_branch.name == "main"
    assert not (vault_repo / "second-brain").exists()


def test_resync_rebuilds_cleanly_discarding_prior_run_content(tmp_path: Path, vault_repo: Path) -> None:
    nw = _worktree(tmp_path, vault_repo)
    worktree_path = nw.sync()
    (worktree_path / "leftover.md").write_text("from a previous run", encoding="utf-8")
    Repo(worktree_path).git.add(all=True)
    Repo(worktree_path).index.commit("simulated prior proposal")

    nw.sync()

    assert not (worktree_path / "leftover.md").exists()
    assert (worktree_path / "note.md").exists()


def test_resync_picks_up_new_commits_on_base_branch(tmp_path: Path, vault_repo: Path) -> None:
    nw = _worktree(tmp_path, vault_repo)
    nw.sync()

    vault = Repo(vault_repo)
    (vault_repo / "new-note.md").write_text("added after first sync", encoding="utf-8")
    vault.git.add(all=True)
    vault.index.commit("add new note")
    vault.git.push("origin", "main")

    worktree_path = nw.sync()

    assert (worktree_path / "new-note.md").read_text(encoding="utf-8") == "added after first sync"


def test_push_lands_branch_on_remote(tmp_path: Path, vault_repo: Path, remote_repo: Path) -> None:
    nw = _worktree(tmp_path, vault_repo)
    worktree_path = nw.sync()
    (worktree_path / "proposal.md").write_text("agent proposal", encoding="utf-8")
    worktree_repo = Repo(worktree_path)
    worktree_repo.git.add(all=True)
    worktree_repo.index.commit("nightly proposal")

    nw.push()

    remote = Repo(remote_repo)
    assert "second-brain/nightly" in [h.name for h in remote.heads]
    tree = remote.commit("second-brain/nightly").tree
    assert "proposal.md" in [item.path for item in tree.traverse()]


def test_second_run_force_pushes_over_first(tmp_path: Path, vault_repo: Path, remote_repo: Path) -> None:
    nw = _worktree(tmp_path, vault_repo)

    worktree_path = nw.sync()
    (worktree_path / "night-one.md").write_text("first run", encoding="utf-8")
    Repo(worktree_path).git.add(all=True)
    Repo(worktree_path).index.commit("night one")
    nw.push()

    worktree_path = nw.sync()
    (worktree_path / "night-two.md").write_text("second run", encoding="utf-8")
    Repo(worktree_path).git.add(all=True)
    Repo(worktree_path).index.commit("night two")
    nw.push()

    remote = Repo(remote_repo)
    filenames = [item.path for item in remote.commit("second-brain/nightly").tree.traverse()]
    assert "night-two.md" in filenames
    assert "night-one.md" not in filenames
