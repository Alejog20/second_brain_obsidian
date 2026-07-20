"""Tests for the typed config loader."""

from pathlib import Path

import pytest

from src.config import load_config

VALID_YAML = """
vault:
  path: /vault
  daily_notes_folder: "01-Daily"
  daily_note_date_format: "MM-DD-YYYY"
  excluded_folders:
    - "_staging"
    - "_reports"

safety:
  mode: dry_run
  require_git: true
  materiality_threshold: structural

embeddings:
  provider: ollama
  model: nomic-embed-text
  store: lancedb
  store_path: /data/vector_store

models:
  bulk_grammar_pass:
    provider: ollama
    model: qwen3.5:9b

cost_tracking:
  enabled: true
  currency: USD

report:
  path: "_reports/Review-{date}.md"
  full_diff_log_path: "_reports/{date}-full-diff.json"
  include_cost_summary: true
"""


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_config_parses_all_sections(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, VALID_YAML))

    assert config.vault.path == "/vault"
    assert set(config.vault.excluded_folders) == {"_staging", "_reports", ".obsidian"}
    assert config.safety.mode == "dry_run"
    assert config.safety.is_apply is False
    assert config.embeddings.model == "nomic-embed-text"
    assert config.cost_tracking.enabled is True
    assert config.report.path == "_reports/Review-{date}.md"
    assert config.fact_check.enabled is True


def test_model_for_returns_task_config(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, VALID_YAML))
    task_cfg = config.model_for("bulk_grammar_pass")
    assert task_cfg.provider == "ollama"
    assert task_cfg.model == "qwen3.5:9b"


def test_model_for_unknown_task_raises(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, VALID_YAML))
    with pytest.raises(ValueError):
        config.model_for("nonexistent_task")


def test_missing_vault_path_raises(tmp_path: Path) -> None:
    bad_yaml = "vault:\n  daily_notes_folder: '01-Daily'\n"
    with pytest.raises(ValueError):
        load_config(_write_config(tmp_path, bad_yaml))


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does-not-exist.yaml")


def test_apply_mode_flag(tmp_path: Path) -> None:
    apply_yaml = VALID_YAML.replace("mode: dry_run", "mode: apply")
    config = load_config(_write_config(tmp_path, apply_yaml))
    assert config.safety.is_apply is True


def test_staging_and_reports_excluded_even_when_field_is_empty(tmp_path: Path) -> None:
    """Regression test: _staging/_reports/.obsidian must be protected even if a config.yaml
    explicitly sets excluded_folders to an empty list - this happened for real (a scratch
    config with no excluded_folders let the reorganizer move a live note into _reports/)."""
    yaml_content = VALID_YAML.replace('excluded_folders:\n    - "_staging"\n    - "_reports"', "excluded_folders: []")
    config = load_config(_write_config(tmp_path, yaml_content))
    assert set(config.vault.excluded_folders) >= {"_staging", "_reports", ".obsidian"}


def test_staging_and_reports_excluded_when_field_is_missing_entirely(tmp_path: Path) -> None:
    yaml_content = "vault:\n  path: /vault\n"
    config = load_config(_write_config(tmp_path, yaml_content))
    assert set(config.vault.excluded_folders) == {"_staging", "_reports", ".obsidian"}


def test_user_excluded_folders_are_added_not_replaced(tmp_path: Path) -> None:
    yaml_content = VALID_YAML.replace(
        'excluded_folders:\n    - "_staging"\n    - "_reports"',
        'excluded_folders:\n    - "attachments"',
    )
    config = load_config(_write_config(tmp_path, yaml_content))
    assert set(config.vault.excluded_folders) == {"_staging", "_reports", ".obsidian", "attachments"}


def test_fact_check_defaults_to_enabled_when_section_is_missing(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, VALID_YAML))
    assert config.fact_check.enabled is True


def test_fact_check_can_be_disabled(tmp_path: Path) -> None:
    yaml_content = VALID_YAML + "\nfact_check:\n  enabled: false\n"
    config = load_config(_write_config(tmp_path, yaml_content))
    assert config.fact_check.enabled is False
