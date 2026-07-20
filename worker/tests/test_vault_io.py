"""Tests for vault I/O: atomic writes, path containment, and frontmatter round-tripping."""

from pathlib import Path

import pytest

from src.vault_io import Note, PathBlockedByFileError, PathEscapesVaultError, VaultIO


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    root.mkdir()
    return root


def test_write_then_read_note_roundtrips(vault: Path) -> None:
    io = VaultIO(vault)
    note = Note(metadata={"title": "Docker Networking", "tags": ["docker"]}, content="Body text.")
    io.write_note("04-Reference/docker-networking.md", note)

    read_back = io.read_note("04-Reference/docker-networking.md")
    assert read_back.metadata["title"] == "Docker Networking"
    assert read_back.metadata["tags"] == ["docker"]
    assert read_back.content == "Body text."


def test_write_note_with_empty_metadata_has_no_frontmatter_block(vault: Path) -> None:
    io = VaultIO(vault)
    io.write_note("note.md", Note(metadata={}, content="just body text"))
    raw = io.read_raw("note.md")
    assert raw == "just body text"
    assert "{}" not in raw


def test_write_creates_parent_folders(vault: Path) -> None:
    io = VaultIO(vault)
    io.write_note("a/b/c/note.md", Note(metadata={}, content="x"))
    assert (vault / "a" / "b" / "c" / "note.md").is_file()


def test_stage_note_writes_under_staging_folder(vault: Path) -> None:
    io = VaultIO(vault)
    staged_path = io.stage_note("04-Reference/note.md", Note(metadata={}, content="proposed"))
    assert staged_path == vault / "_staging" / "04-Reference" / "note.md"
    assert staged_path.is_file()
    assert not (vault / "04-Reference" / "note.md").exists()


def test_relative_path_escape_is_rejected(vault: Path) -> None:
    io = VaultIO(vault)
    with pytest.raises(PathEscapesVaultError):
        io.resolve("../outside.md")


def test_absolute_path_escape_is_rejected(vault: Path) -> None:
    io = VaultIO(vault)
    with pytest.raises(PathEscapesVaultError):
        io.resolve("/etc/passwd")


def test_no_leftover_tmp_file_after_write(vault: Path) -> None:
    io = VaultIO(vault)
    io.write_note("note.md", Note(metadata={}, content="x"))
    assert not (vault / "note.md.tmp").exists()


def test_missing_vault_root_raises() -> None:
    with pytest.raises(FileNotFoundError):
        VaultIO(Path("/nonexistent/vault/root"))


def test_exists_true_for_present_file(vault: Path) -> None:
    io = VaultIO(vault)
    io.write_note("note.md", Note(metadata={}, content="x"))
    assert io.exists("note.md") is True


def test_exists_false_for_missing_file(vault: Path) -> None:
    io = VaultIO(vault)
    assert io.exists("nope.md") is False


def test_exists_false_for_escaping_path_does_not_raise(vault: Path) -> None:
    io = VaultIO(vault)
    assert io.exists("../outside.md") is False


def test_write_or_stage_dry_run_writes_to_staging(vault: Path) -> None:
    io = VaultIO(vault)
    path = io.write_or_stage("note.md", Note(metadata={}, content="proposed"), dry_run=True)
    assert path == vault / "_staging" / "note.md"
    assert not (vault / "note.md").exists()


def test_write_or_stage_apply_writes_live(vault: Path) -> None:
    io = VaultIO(vault)
    path = io.write_or_stage("note.md", Note(metadata={}, content="live"), dry_run=False)
    assert path == vault / "note.md"
    assert io.read_note("note.md").content == "live"


def test_move_note_dry_run_stages_destination_and_keeps_original(vault: Path) -> None:
    io = VaultIO(vault)
    io.write_note("00-Inbox/note.md", Note(metadata={}, content="original"))
    note = Note(metadata={}, content="original")

    result_path = io.move_note("00-Inbox/note.md", "02-Areas/note.md", note, dry_run=True)

    assert result_path == vault / "_staging" / "02-Areas" / "note.md"
    assert io.exists("00-Inbox/note.md")
    assert not io.exists("02-Areas/note.md")


def test_move_note_apply_relocates_and_removes_original(vault: Path) -> None:
    io = VaultIO(vault)
    io.write_note("00-Inbox/note.md", Note(metadata={}, content="original"))
    note = Note(metadata={}, content="original")

    result_path = io.move_note("00-Inbox/note.md", "02-Areas/note.md", note, dry_run=False)

    assert result_path == vault / "02-Areas" / "note.md"
    assert not io.exists("00-Inbox/note.md")
    assert io.read_note("02-Areas/note.md").content == "original"


def test_read_write_raw_roundtrip(vault: Path) -> None:
    io = VaultIO(vault)
    io.write_raw("AGENTS.md", "# Hello\n\nplain text, no frontmatter")
    assert io.read_raw("AGENTS.md") == "# Hello\n\nplain text, no frontmatter"


def test_iter_notes_skips_excluded_folders(vault: Path) -> None:
    io = VaultIO(vault)
    io.write_note("00-Inbox/a.md", Note(metadata={"title": "A"}, content="a"))
    io.write_note("_staging/b.md", Note(metadata={"title": "B"}, content="b"))

    found = dict(io.iter_notes(excluded_folders=frozenset({"_staging"})))

    assert "00-Inbox/a.md" in found
    assert "_staging/b.md" not in found


def test_iter_notes_yields_parsed_metadata(vault: Path) -> None:
    io = VaultIO(vault)
    io.write_note("note.md", Note(metadata={"title": "T", "tags": ["x"]}, content="body"))

    found = dict(io.iter_notes())

    assert found["note.md"].metadata["title"] == "T"
    assert found["note.md"].content == "body"


def test_write_note_raises_clear_error_when_target_folder_is_actually_a_file(vault: Path) -> None:
    """Regression test: writing "existing-file.md/new-note.md" must fail clearly, not with a
    cryptic mkdir FileExistsError - this happened for real when a taxonomy bug suggested
    treating a note's own filename as a destination folder."""
    io = VaultIO(vault)
    io.write_note("existing-file.md", Note(metadata={}, content="I am a file, not a folder"))

    with pytest.raises(PathBlockedByFileError, match="existing-file.md"):
        io.write_note("existing-file.md/new-note.md", Note(metadata={}, content="new"))


def test_move_note_raises_clear_error_when_destination_folder_is_actually_a_file(vault: Path) -> None:
    io = VaultIO(vault)
    io.write_note("existing-file.md", Note(metadata={}, content="I am a file, not a folder"))
    io.write_note("source.md", Note(metadata={}, content="to be moved"))

    with pytest.raises(PathBlockedByFileError):
        io.move_note("source.md", "existing-file.md/source.md", Note(metadata={}, content="to be moved"), dry_run=False)
