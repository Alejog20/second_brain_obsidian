"""Tests for the manifest diff scanner."""

from pathlib import Path

import pytest

from src.manifest import ManifestManager, VaultDelta


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    vault_root = tmp_path / "vault"
    _write(vault_root / "01-Daily" / "07-16-2026.md", "# Today\ncontent")
    _write(vault_root / "_staging" / "ignored.md", "should be excluded")
    return vault_root


@pytest.fixture
def manager(vault: Path, tmp_path: Path) -> ManifestManager:
    db_path = tmp_path / "data" / "manifest.sqlite"
    return ManifestManager(vault_path=str(vault), excluded_folders=["_staging", "_reports"], db_path=db_path)


def test_first_scan_reports_all_files_as_added(manager: ManifestManager) -> None:
    delta = manager.get_delta()
    assert delta.added == [Path("01-Daily/07-16-2026.md")]
    assert delta.modified == []
    assert delta.deleted == []


def test_excluded_folders_are_skipped(manager: ManifestManager) -> None:
    delta = manager.get_delta()
    assert all("_staging" not in str(p) for p in delta.added)


def test_commit_then_rescan_reports_no_changes(manager: ManifestManager) -> None:
    manager.get_delta()
    manager.commit()
    delta = manager.get_delta()
    assert delta == VaultDelta()


def test_modified_file_is_detected_after_commit(manager: ManifestManager, vault: Path) -> None:
    manager.get_delta()
    manager.commit()
    _write(vault / "01-Daily" / "07-16-2026.md", "# Today\nchanged content")
    delta = manager.get_delta()
    assert delta.modified == [Path("01-Daily/07-16-2026.md")]
    assert delta.added == []


def test_deleted_file_is_detected_after_commit(manager: ManifestManager, vault: Path) -> None:
    manager.get_delta()
    manager.commit()
    (vault / "01-Daily" / "07-16-2026.md").unlink()
    delta = manager.get_delta()
    assert delta.deleted == [Path("01-Daily/07-16-2026.md")]


def test_missing_vault_root_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError):
        ManifestManager(vault_path=str(missing), db_path=tmp_path / "manifest.sqlite")
