"""Tests for the CLI: command wiring, flag precedence, and error handling."""

from pathlib import Path
from typing import Any, Optional

import httpx
import pytest
from typer.testing import CliRunner

import src.cli as cli_module
from src.cli import app

runner = CliRunner()

CONFIG_YAML = """
vault:
  path: {vault_path}
  daily_notes_folder: "01-Daily"
  default_new_note_folder: "00-Inbox"
safety:
  mode: dry_run
  require_git: true
embeddings:
  provider: ollama
  model: nomic-embed-text
models:
  bulk_grammar_pass:
    provider: ollama
    model: qwen3.5:9b
  daily_digestion:
    provider: gemini
    model: gemini-3.5-flash
report:
  path: "_reports/Review-{{date}}.md"
git_review:
  enabled: false
  branch: second-brain/nightly
  base_branch: main
"""


@pytest.fixture
def vault_dir(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


@pytest.fixture
def config_path(tmp_path: Path, vault_dir: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_YAML.format(vault_path=vault_dir), encoding="utf-8")
    return path


class FakeNightlyRun:
    """Stands in for NightlyRun; records construction args and returns a canned report."""

    calls: list[dict[str, Any]] = []

    def __init__(self, config: Any, dry_run: bool, vault_root: Optional[Path] = None, full_scan: bool = False, **kwargs: Any) -> None:
        FakeNightlyRun.calls.append({"config": config, "dry_run": dry_run, "vault_root": vault_root, "full_scan": full_scan})
        self.dry_run = dry_run

    def run(self) -> str:
        return "# Review — fake\n\nfake report body"


class FakeWorktree:
    """Stands in for NightlyWorktree; records sync/push calls without touching real git."""

    instances: list["FakeWorktree"] = []

    def __init__(self, vault_root: Path, worktree_path: Path, branch: str, base_branch: str, remote: str = "origin") -> None:
        self.vault_root = vault_root
        self.worktree_path = worktree_path
        self.branch = branch
        self.synced = False
        self.pushed = False
        FakeWorktree.instances.append(self)

    def sync(self) -> Path:
        self.synced = True
        return self.worktree_path

    def push(self) -> None:
        self.pushed = True


@pytest.fixture(autouse=True)
def _reset_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeNightlyRun.calls.clear()
    FakeWorktree.instances.clear()
    # cli.py loads .env at import time (see cli.py's top-of-file comment); an ambient .env in
    # the real repo (VAULT_PATH, GEMINI_API_KEY, etc.) must not leak into these tests, which
    # need a clean, predictable environment regardless of what's set on the machine running them.
    for var in ("VAULT_PATH", "GEMINI_API_KEY", "CONFIG_PATH", "DRY_RUN", "OLLAMA_HOST"):
        monkeypatch.delenv(var, raising=False)


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "status" in result.output
    assert "check" in result.output
    assert "report" in result.output


def test_status_shows_config_summary(config_path: Path) -> None:
    result = runner.invoke(app, ["status", "--config", str(config_path)])
    assert result.exit_code == 0
    assert "Safety mode" in result.output
    assert "daily_digestion" in result.output
    assert "gemini" in result.output


def test_status_missing_config_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["status", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 1
    assert "Config error" in result.output


def test_check_reports_missing_vault_and_gemini_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_YAML.format(vault_path=tmp_path / "does-not-exist"), encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def fake_get(*args: Any, **kwargs: Any) -> httpx.Response:
        return httpx.Response(status_code=200, json={"models": []}, request=httpx.Request("GET", "http://localhost:11434/api/tags"))

    monkeypatch.setattr(httpx, "get", fake_get)

    result = runner.invoke(app, ["check", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "MISSING" in result.output


def _fake_get_by_url(gemini_status: int = 200, gemini_models: Optional[list[str]] = None):
    """Routes fake httpx.get responses by URL, so Ollama and Gemini can be tested independently."""

    def fake_get(url: str, *args: Any, **kwargs: Any) -> httpx.Response:
        request = httpx.Request("GET", url)
        if "generativelanguage.googleapis.com" in url:
            body = {"models": [{"name": m} for m in (gemini_models or [])]}
            return httpx.Response(status_code=gemini_status, json=body, request=request)
        return httpx.Response(status_code=200, json={"models": []}, request=request)

    return fake_get


def test_check_all_ok(config_path: Path, vault_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from git import Repo

    Repo.init(vault_dir)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(httpx, "get", _fake_get_by_url(gemini_models=["gemini-3.5-flash"]))

    result = runner.invoke(app, ["check", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "OK" in result.output
    assert "1 models visible" in result.output


def test_check_gemini_key_invalid(config_path: Path, vault_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from git import Repo

    Repo.init(vault_dir)
    monkeypatch.setenv("GEMINI_API_KEY", "a-bad-key")
    monkeypatch.setattr(httpx, "get", _fake_get_by_url(gemini_status=401))

    result = runner.invoke(app, ["check", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "INVALID KEY" in result.output


def test_check_gemini_unreachable(config_path: Path, vault_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from git import Repo

    Repo.init(vault_dir)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def fake_get(url: str, *args: Any, **kwargs: Any) -> httpx.Response:
        if "generativelanguage.googleapis.com" in url:
            raise httpx.ConnectError("connection refused", request=httpx.Request("GET", url))
        return httpx.Response(status_code=200, json={"models": []}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)

    result = runner.invoke(app, ["check", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "UNREACHABLE" in result.output


def test_run_rejects_apply_and_dry_run_together(config_path: Path) -> None:
    result = runner.invoke(app, ["run", "--config", str(config_path), "--apply", "--dry-run"])
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


def test_run_apply_flag_overrides_config(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "NightlyRun", FakeNightlyRun)

    result = runner.invoke(app, ["run", "--config", str(config_path), "--apply"])

    assert result.exit_code == 0
    assert FakeNightlyRun.calls[0]["dry_run"] is False
    assert "APPLIED" in result.output


def test_run_full_flag_passes_full_scan(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "NightlyRun", FakeNightlyRun)

    result = runner.invoke(app, ["run", "--config", str(config_path), "--dry-run", "--full"])

    assert result.exit_code == 0
    assert FakeNightlyRun.calls[0]["full_scan"] is True
    assert "Scope" in result.output


def test_run_without_full_flag_defaults_to_incremental(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "NightlyRun", FakeNightlyRun)

    result = runner.invoke(app, ["run", "--config", str(config_path), "--dry-run"])

    assert result.exit_code == 0
    assert FakeNightlyRun.calls[0]["full_scan"] is False


def test_run_defaults_to_dry_run_from_config(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "NightlyRun", FakeNightlyRun)
    monkeypatch.delenv("DRY_RUN", raising=False)

    result = runner.invoke(app, ["run", "--config", str(config_path)])

    assert result.exit_code == 0
    assert FakeNightlyRun.calls[0]["dry_run"] is True
    assert "DRY RUN" in result.output


def test_run_shows_report_by_default(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "NightlyRun", FakeNightlyRun)

    result = runner.invoke(app, ["run", "--config", str(config_path), "--dry-run"])

    assert "fake report body" in result.output


def test_run_no_show_report_suppresses_output(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "NightlyRun", FakeNightlyRun)

    result = runner.invoke(app, ["run", "--config", str(config_path), "--dry-run", "--no-show-report"])

    assert "fake report body" not in result.output


def test_run_git_review_flag_syncs_and_pushes_worktree(config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "NightlyRun", FakeNightlyRun)
    monkeypatch.setattr(cli_module, "NightlyWorktree", FakeWorktree)

    result = runner.invoke(app, ["run", "--config", str(config_path), "--git-review"])

    assert result.exit_code == 0
    assert len(FakeWorktree.instances) == 1
    assert FakeWorktree.instances[0].synced is True
    assert FakeWorktree.instances[0].pushed is True
    # git_review mode always writes live - the branch itself is the review gate
    assert FakeNightlyRun.calls[0]["dry_run"] is False


def test_run_without_git_review_flag_never_touches_worktree(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "NightlyRun", FakeNightlyRun)
    monkeypatch.setattr(cli_module, "NightlyWorktree", FakeWorktree)

    result = runner.invoke(app, ["run", "--config", str(config_path), "--dry-run"])

    assert result.exit_code == 0
    assert FakeWorktree.instances == []


def test_report_command_shows_existing_report(config_path: Path, vault_dir: Path) -> None:
    from datetime import datetime

    (vault_dir / "_reports").mkdir()
    today = datetime.now().strftime("%m-%d-%Y")
    (vault_dir / "_reports" / f"Review-{today}.md").write_text("# Review — today\n\nAll good.", encoding="utf-8")

    result = runner.invoke(app, ["report", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "All good" in result.output


def test_report_command_missing_report_exits_nonzero(config_path: Path, vault_dir: Path) -> None:
    result = runner.invoke(app, ["report", "--config", str(config_path), "--date", "01-01-2020"])
    assert result.exit_code == 1
    assert "No review found" in result.output


def test_report_command_rejects_bad_type(config_path: Path, vault_dir: Path) -> None:
    result = runner.invoke(app, ["report", "--config", str(config_path), "--type", "nonsense"])
    assert result.exit_code == 1
    assert "--type" in result.output
    assert "review" in result.output
    assert "recap" in result.output


def test_report_command_recap_type(config_path: Path, vault_dir: Path) -> None:
    from datetime import datetime

    (vault_dir / "_reports").mkdir()
    today = datetime.now().strftime("%m-%d-%Y")
    (vault_dir / "_reports" / f"Recap-{today}.md").write_text("# Recap — today\n\nYou learned some things.", encoding="utf-8")

    result = runner.invoke(app, ["report", "--config", str(config_path), "--type", "recap"])

    assert result.exit_code == 0
    assert "You learned some things" in result.output
