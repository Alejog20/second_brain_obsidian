"""Tests for the git commit-before/after safety bracket."""

from pathlib import Path

import pytest
from git import Repo

from src.git_safety import GitSafety, VaultNotAGitRepoError


@pytest.fixture
def git_vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    root.mkdir()
    Repo.init(root)
    return root


def test_requires_git_repo_by_default(tmp_path: Path) -> None:
    non_repo = tmp_path / "not-a-repo"
    non_repo.mkdir()
    with pytest.raises(VaultNotAGitRepoError):
        GitSafety(non_repo, require_git=True)


def test_require_git_false_allows_non_repo(tmp_path: Path) -> None:
    non_repo = tmp_path / "not-a-repo"
    non_repo.mkdir()
    safety = GitSafety(non_repo, require_git=False)
    assert safety.has_uncommitted_changes() is False
    assert safety.commit("no-op") is None


def test_commit_returns_hexsha_when_changes_exist(git_vault: Path) -> None:
    (git_vault / "note.md").write_text("hello", encoding="utf-8")
    safety = GitSafety(git_vault)
    assert safety.has_uncommitted_changes() is True
    hexsha = safety.commit("add note")
    assert hexsha is not None
    assert safety.has_uncommitted_changes() is False


def test_commit_returns_none_when_nothing_changed(git_vault: Path) -> None:
    safety = GitSafety(git_vault)
    assert safety.commit("nothing to commit") is None


def test_bracket_commits_before_and_after(git_vault: Path) -> None:
    safety = GitSafety(git_vault)
    (git_vault / "before.md").write_text("before", encoding="utf-8")

    with safety.bracket("test run"):
        (git_vault / "during.md").write_text("during", encoding="utf-8")

    repo = Repo(git_vault)
    messages = [c.message for c in repo.iter_commits()]
    assert any("pre-run snapshot before test run" in m for m in messages)
    assert any(m.strip() == "second-brain: test run" for m in messages)
    assert safety.has_uncommitted_changes() is False


def test_bracket_still_commits_after_exception(git_vault: Path) -> None:
    safety = GitSafety(git_vault)

    with pytest.raises(RuntimeError):
        with safety.bracket("failing run"):
            (git_vault / "partial.md").write_text("partial", encoding="utf-8")
            raise RuntimeError("boom")

    assert safety.has_uncommitted_changes() is False
