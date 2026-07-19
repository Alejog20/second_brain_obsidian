"""Tests for the AGENTS.md builder: taxonomy scanning and manual-notes preservation."""

from pathlib import Path

import pytest

from src.agents_md_builder import AgentsBuilder, MANUAL_NOTES_MARKER
from src.vault_io import Note, VaultIO


@pytest.fixture
def vault(tmp_path: Path) -> VaultIO:
    root = tmp_path / "vault"
    root.mkdir()
    return VaultIO(root)


def test_analyze_counts_folders_tags_and_frontmatter_keys(vault: VaultIO) -> None:
    vault.write_note("02-Areas/security.md", Note(metadata={"title": "Security", "tags": ["security", "networking"]}, content="x"))
    vault.write_note("02-Areas/docker.md", Note(metadata={"title": "Docker", "tags": ["networking"]}, content="x"))
    vault.write_note("03-Projects/proj.md", Note(metadata={"title": "Proj", "status": "draft"}, content="x"))

    taxonomy = AgentsBuilder(vault).analyze()

    assert taxonomy.note_count == 3
    assert set(taxonomy.folders) == {"02-Areas", "03-Projects"}
    tag_counts = dict(taxonomy.tags)
    assert tag_counts["networking"] == 2
    assert tag_counts["security"] == 1
    key_counts = dict(taxonomy.frontmatter_keys)
    assert key_counts["title"] == 3
    assert key_counts["status"] == 1


def test_analyze_skips_excluded_folders(vault: VaultIO) -> None:
    vault.write_note("02-Areas/note.md", Note(metadata={"title": "T"}, content="x"))
    vault.write_note("_staging/draft.md", Note(metadata={"title": "Draft"}, content="x"))

    taxonomy = AgentsBuilder(vault, excluded_folders=frozenset({"_staging"})).analyze()

    assert taxonomy.note_count == 1
    assert taxonomy.folders == ("02-Areas",)


def test_build_renders_folders_and_note_count(vault: VaultIO) -> None:
    vault.write_note("02-Areas/security.md", Note(metadata={"title": "Security"}, content="x"))

    content = AgentsBuilder(vault).build(vault_name="My Vault", generated_date="2026-07-19")

    assert "AGENTS.md — My Vault second brain" in content
    assert "- `02-Areas/`" in content
    assert "1 notes across 1 top-level folders" in content
    assert MANUAL_NOTES_MARKER in content


def test_build_preserves_manual_notes_from_existing_file(vault: VaultIO) -> None:
    existing = f"# AGENTS.md — old\n\nstale content\n\n---\n{MANUAL_NOTES_MARKER}\nMy hand-written rules here.\n"

    content = AgentsBuilder(vault).build(vault_name="My Vault", generated_date="2026-07-19", existing_agents_md=existing)

    assert "My hand-written rules here." in content
    assert content.count(MANUAL_NOTES_MARKER) == 2  # once in the fresh template, once from the preserved block


def test_build_with_no_prior_agents_md_has_empty_manual_section(vault: VaultIO) -> None:
    content = AgentsBuilder(vault).build(vault_name="My Vault", generated_date="2026-07-19")
    assert content.rstrip().endswith(MANUAL_NOTES_MARKER)
