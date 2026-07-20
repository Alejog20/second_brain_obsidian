"""Integration tests for the nightly orchestrator: manifest -> reorganize -> digest -> AGENTS.md -> report."""

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest
from git import Repo

from src.config import Config, CostTrackingConfig, EmbeddingsConfig, FactCheckConfig, GitReviewConfig, ModelTaskConfig, ReportConfig, SafetyConfig, VaultConfig
from src.git_safety import GitSafety
from src.jobs.nightly_run import NightlyRun, _build_default_embedder, resolve_dry_run
from src.llm_router import LLMResponse
from src.manifest import ManifestManager
from src.vault_io import Note, VaultIO
from src.vector_store import EmbeddedChunk, GeminiEmbeddingClient, OllamaEmbeddingClient, VectorStore

CLEAR_GRAMMAR_RESPONSE = "---CORRECTED---\nFixed text.\n---CLARITY---\nclear"


class FakeRouter:
    """Duck-types the Router protocol; returns canned text per task_key."""

    def __init__(self, responses: dict[str, str]) -> None:
        self._responses = responses
        self.total_cost_usd = 0.0

    def generate(self, task_key: str, system: str, prompt: str, grounded: bool = False) -> LLMResponse:
        return LLMResponse(text=self._responses.get(task_key, ""), tokens_in=1, tokens_out=1, cost_usd=0.0)


class FakeEmbedder:
    """Returns a fixed vector regardless of input, for deterministic vector-store tests."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def embed(self, text: str) -> list[float]:
        return self._vector


def make_config(
    vault_root: Path,
    mode: str = "dry_run",
    materiality: str = "structural",
    daily_notes_folder: str = "01-Daily",
    default_new_note_folder: str = "00-Inbox",
) -> Config:
    return Config(
        vault=VaultConfig(
            path=str(vault_root),
            daily_notes_folder=daily_notes_folder,
            daily_note_date_format="MM-DD-YYYY",
            default_new_note_folder=default_new_note_folder,
            excluded_folders=("_staging", "_reports"),
        ),
        safety=SafetyConfig(mode=mode, require_git=True, materiality_threshold=materiality),
        embeddings=EmbeddingsConfig(provider="ollama", model="nomic-embed-text", store="lancedb", store_path="/data/vector_store"),
        models={
            "title_and_tagging": ModelTaskConfig(provider="ollama", model="qwen"),
            "bulk_grammar_pass": ModelTaskConfig(provider="ollama", model="qwen"),
            "daily_digestion": ModelTaskConfig(provider="ollama", model="qwen"),
            "fact_check": ModelTaskConfig(provider="ollama", model="qwen"),
        },
        cost_tracking=CostTrackingConfig(enabled=True, currency="USD"),
        report=ReportConfig(path="_reports/Review-{date}.md", full_diff_log_path="_reports/{date}-full-diff.json", include_cost_summary=True),
        fact_check=FactCheckConfig(enabled=True),
        git_review=GitReviewConfig(),
    )


def test_build_default_embedder_selects_gemini(tmp_path: Path) -> None:
    config = replace(make_config(tmp_path), embeddings=EmbeddingsConfig(provider="gemini", model="gemini-embedding-001", store="lancedb", store_path="/data/vector_store"))
    assert isinstance(_build_default_embedder(config), GeminiEmbeddingClient)


def test_build_default_embedder_selects_ollama(tmp_path: Path) -> None:
    config = replace(make_config(tmp_path), embeddings=EmbeddingsConfig(provider="ollama", model="nomic-embed-text", store="lancedb", store_path="/data/vector_store"))
    assert isinstance(_build_default_embedder(config), OllamaEmbeddingClient)


def test_build_default_embedder_rejects_unknown_provider(tmp_path: Path) -> None:
    config = replace(make_config(tmp_path), embeddings=EmbeddingsConfig(provider="voyage", model="x", store="lancedb", store_path="/data/vector_store"))
    with pytest.raises(ValueError, match="unsupported embeddings provider"):
        _build_default_embedder(config)


@pytest.fixture
def vault_root(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    root.mkdir()
    Repo.init(root)
    return root


def _make_run(
    tmp_path: Path,
    vault_root: Path,
    mode: str,
    router_responses: dict[str, str],
    embed_vector: list[float],
    materiality: str = "structural",
    daily_notes_folder: str = "01-Daily",
    default_new_note_folder: str = "00-Inbox",
    full_scan: bool = False,
) -> NightlyRun:
    config = make_config(
        vault_root,
        mode=mode,
        materiality=materiality,
        daily_notes_folder=daily_notes_folder,
        default_new_note_folder=default_new_note_folder,
    )
    vault = VaultIO(vault_root)
    git = GitSafety(vault_root, require_git=True)
    router = FakeRouter(router_responses)
    embedder = FakeEmbedder(embed_vector)
    vector_store = VectorStore(tmp_path / "vector_store", embedding_dim=4)
    manifest = ManifestManager(vault_path=str(vault_root), excluded_folders=["_staging", "_reports"], db_path=tmp_path / "manifest.sqlite")
    return NightlyRun(
        config,
        dry_run=(mode != "apply"),
        vault=vault,
        git=git,
        router=router,
        embedder=embedder,
        vector_store=vector_store,
        manifest=manifest,
        full_scan=full_scan,
    )


def test_resolve_dry_run_defaults_to_true(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DRY_RUN", raising=False)
    config = make_config(tmp_path, mode="apply")
    assert resolve_dry_run(config) is True


def test_resolve_dry_run_requires_both_signals(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DRY_RUN", "false")
    assert resolve_dry_run(make_config(tmp_path, mode="dry_run")) is True
    assert resolve_dry_run(make_config(tmp_path, mode="apply")) is False


def test_dry_run_stages_proposal_and_leaves_live_note_untouched(tmp_path: Path, vault_root: Path) -> None:
    vault = VaultIO(vault_root)
    vault.write_note("00-Inbox/idea.md", Note(metadata={"title": "Untitled"}, content="teh idea"))

    run = _make_run(tmp_path, vault_root, "dry_run", {"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE, "title_and_tagging": "A Real Title"}, [1.0, 0.0, 0.0, 0.0])
    report = run.run()

    assert vault.read_note("00-Inbox/idea.md").content == "teh idea"
    assert vault.exists("_staging/00-Inbox/idea.md")
    assert "1 notes scanned" in report


def test_apply_mode_moves_note_to_high_confidence_taxonomy_match(tmp_path: Path, vault_root: Path) -> None:
    vault = VaultIO(vault_root)
    vault.write_note("00-Inbox/note.md", Note(metadata={"title": "Existing Title"}, content="body"))

    config = make_config(vault_root, mode="apply")
    vector_store = VectorStore(tmp_path / "vector_store", embedding_dim=4)
    vector_store.upsert(EmbeddedChunk(id="02-Areas/security.md", text="x", vector=[1.0, 0.0, 0.0, 0.0], path="02-Areas/security.md", note_title="Security"))
    git = GitSafety(vault_root, require_git=True)
    manifest = ManifestManager(vault_path=str(vault_root), excluded_folders=["_staging", "_reports"], db_path=tmp_path / "manifest.sqlite")

    run = NightlyRun(
        config,
        dry_run=False,
        vault=vault,
        git=git,
        router=FakeRouter({"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE}),
        embedder=FakeEmbedder([1.0, 0.0, 0.0, 0.0]),
        vector_store=vector_store,
        manifest=manifest,
    )
    report = run.run()

    assert not vault.exists("00-Inbox/note.md")
    assert vault.exists("02-Areas/note.md")
    assert "Moved note.md to 02-Areas" in report
    assert vault.exists("AGENTS.md")


def test_daily_digestion_creates_note_and_updates_report(tmp_path: Path, vault_root: Path) -> None:
    vault = VaultIO(vault_root)
    today = datetime.now().strftime("%m-%d-%Y")
    vault.write_note(f"01-Daily/{today}.md", Note(metadata={}, content="# New Idea\nSomething worth keeping."))

    run = _make_run(tmp_path, vault_root, "apply", {"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE, "daily_digestion": "New Idea Note"}, [1.0, 0.0, 0.0, 0.0])
    report = run.run()

    assert vault.exists("00-Inbox/New-Idea-Note.md")
    assert "New Idea Note" in report
    daily = vault.read_note(f"01-Daily/{today}.md")
    assert "## Notes generated" in daily.content


def test_daily_note_itself_is_not_reorganized(tmp_path: Path, vault_root: Path) -> None:
    vault = VaultIO(vault_root)
    today = datetime.now().strftime("%m-%d-%Y")
    vault.write_note(f"01-Daily/{today}.md", Note(metadata={"title": "Untitled"}, content="# Idea\nRaw journal text."))

    run = _make_run(tmp_path, vault_root, "apply", {"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE, "title_and_tagging": "Should Not Be Used", "daily_digestion": "Idea"}, [1.0, 0.0, 0.0, 0.0])
    run.run()

    daily = vault.read_note(f"01-Daily/{today}.md")
    assert daily.metadata.get("title") == "Untitled"


def test_root_based_vault_digests_daily_note_to_root(tmp_path: Path, vault_root: Path) -> None:
    """Mirrors a real vault with no 01-Daily/ or 00-Inbox/ - daily notes and new notes both land at root."""
    vault = VaultIO(vault_root)
    today = datetime.now().strftime("%m-%d-%Y")
    vault.write_note(f"{today}.md", Note(metadata={}, content="# New Idea\nSomething worth keeping."))

    run = _make_run(
        tmp_path,
        vault_root,
        "apply",
        {"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE, "daily_digestion": "New Idea Note"},
        [1.0, 0.0, 0.0, 0.0],
        daily_notes_folder="",
        default_new_note_folder="",
    )
    report = run.run()

    assert vault.exists("New-Idea-Note.md")
    assert "New Idea Note" in report
    daily = vault.read_note(f"{today}.md")
    assert "## Notes generated" in daily.content


def test_root_based_vault_reorganizes_topic_folder_notes_normally(tmp_path: Path, vault_root: Path) -> None:
    """A topic-folder note (e.g. AI/) is a regular note under a root-based layout too - it still gets reorganized."""
    vault = VaultIO(vault_root)
    vault.write_note("AI/existing-topic-note.md", Note(metadata={"title": "Untitled"}, content="teh existing note"))

    run = _make_run(
        tmp_path,
        vault_root,
        "apply",
        {"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE, "title_and_tagging": "Existing Topic Note"},
        [1.0, 0.0, 0.0, 0.0],
        daily_notes_folder="",
        default_new_note_folder="",
    )
    run.run()

    updated = vault.read_note("AI/existing-topic-note.md")
    assert updated.metadata["title"] == "Existing Topic Note"
    assert updated.content == "Fixed text."


def test_root_level_non_daily_note_is_still_reorganized(tmp_path: Path, vault_root: Path) -> None:
    """A root-level file that ISN'T a dated daily note (e.g. a canvas-adjacent stray note) is still a regular note."""
    vault = VaultIO(vault_root)
    vault.write_note("random-root-note.md", Note(metadata={"title": "Untitled"}, content="teh idea"))

    run = _make_run(
        tmp_path,
        vault_root,
        "apply",
        {"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE, "title_and_tagging": "A Real Title"},
        [1.0, 0.0, 0.0, 0.0],
        daily_notes_folder="",
        default_new_note_folder="",
    )
    report = run.run()

    assert vault.read_note("random-root-note.md").metadata["title"] == "A Real Title"
    assert "1 notes scanned" in report


def test_second_run_with_no_external_changes_reports_zero_scanned(tmp_path: Path, vault_root: Path) -> None:
    vault = VaultIO(vault_root)
    vault.write_note("00-Inbox/idea.md", Note(metadata={"title": "Untitled"}, content="teh idea"))

    run = _make_run(tmp_path, vault_root, "apply", {"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE, "title_and_tagging": "A Real Title"}, [1.0, 0.0, 0.0, 0.0])
    first_report = run.run()
    second_report = run.run()

    assert "1 notes scanned" in first_report
    assert "0 notes scanned" in second_report


def test_materiality_any_reports_minor_edits_as_significant(tmp_path: Path, vault_root: Path) -> None:
    vault = VaultIO(vault_root)
    vault.write_note("00-Inbox/idea.md", Note(metadata={"title": "Already Descriptive"}, content="teh idea"))

    run = _make_run(tmp_path, vault_root, "apply", {"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE}, [1.0, 0.0, 0.0, 0.0], materiality="any")
    report = run.run()

    assert "Grammar/title tidy-up" in report


def test_apply_run_produces_git_commits(tmp_path: Path, vault_root: Path) -> None:
    vault = VaultIO(vault_root)
    vault.write_note("00-Inbox/idea.md", Note(metadata={"title": "Untitled"}, content="teh idea"))

    run = _make_run(tmp_path, vault_root, "apply", {"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE, "title_and_tagging": "A Real Title"}, [1.0, 0.0, 0.0, 0.0])
    run.run()

    repo = Repo(vault_root)
    messages = [c.message for c in repo.iter_commits()]
    assert any("pre-run snapshot" in m for m in messages)
    assert any("nightly run" in m for m in messages)


def test_report_file_written_to_reports_folder(tmp_path: Path, vault_root: Path) -> None:
    vault = VaultIO(vault_root)
    vault.write_note("00-Inbox/idea.md", Note(metadata={"title": "Untitled"}, content="teh idea"))

    run = _make_run(tmp_path, vault_root, "dry_run", {"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE, "title_and_tagging": "A Real Title"}, [1.0, 0.0, 0.0, 0.0])
    report = run.run()

    today = datetime.now().strftime("%m-%d-%Y")
    assert vault.read_raw(f"_reports/Review-{today}.md") == report


def test_vault_root_override_propagates_to_default_constructed_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A regression test: NightlyRun's own ManifestManager must scan vault_root, not config.vault.path.

    config.vault.path points somewhere that doesn't exist; if the manifest scanned that
    instead of the vault_root override, ManifestManager's constructor would raise
    FileNotFoundError immediately.
    """
    import src.manifest as manifest_module

    monkeypatch.setattr(manifest_module, "DEFAULT_DB_PATH", tmp_path / "manifest.sqlite")

    bogus_config_path = tmp_path / "does-not-exist"
    real_vault = tmp_path / "actual-vault"
    real_vault.mkdir()
    Repo.init(real_vault)
    vault = VaultIO(real_vault)
    vault.write_note("00-Inbox/idea.md", Note(metadata={"title": "Untitled"}, content="teh idea"))

    config = make_config(bogus_config_path, mode="apply")
    run = NightlyRun(
        config,
        dry_run=False,
        vault_root=real_vault,
        vault=vault,
        git=GitSafety(real_vault, require_git=True),
        router=FakeRouter({"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE, "title_and_tagging": "A Real Title"}),
        embedder=FakeEmbedder([1.0, 0.0, 0.0, 0.0]),
        vector_store=VectorStore(tmp_path / "vector_store", embedding_dim=4),
        # manifest intentionally NOT injected - exercises NightlyRun's own default construction
    )
    report = run.run()

    assert not bogus_config_path.exists()
    assert vault.read_note("00-Inbox/idea.md").metadata["title"] == "A Real Title"
    assert "1 notes scanned" in report


def test_default_constructed_vector_store_uses_data_path_env_var_not_config(
    tmp_path: Path, vault_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regression test: the default VectorStore must come from $DATA_PATH, not config.yaml's
    embeddings.store_path (which is /data/vector_store - unwritable outside a container)."""
    import src.jobs.nightly_run as nightly_run_module

    data_path = tmp_path / "local-data"
    monkeypatch.setattr(nightly_run_module, "DEFAULT_DATA_PATH", data_path)

    vault = VaultIO(vault_root)
    vault.write_note("00-Inbox/idea.md", Note(metadata={"title": "Untitled"}, content="teh idea"))

    config = make_config(vault_root, mode="apply")
    run = NightlyRun(
        config,
        dry_run=False,
        vault=vault,
        git=GitSafety(vault_root, require_git=True),
        router=FakeRouter({"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE, "title_and_tagging": "A Real Title"}),
        embedder=FakeEmbedder([1.0] + [0.0] * 767),  # matches VectorStore's default embedding_dim=768
        manifest=ManifestManager(vault_path=str(vault_root), excluded_folders=["_staging", "_reports"], db_path=tmp_path / "manifest.sqlite"),
        # vector_store intentionally NOT injected - exercises NightlyRun's own default construction
    )
    run.run()

    assert (data_path / "vector_store").is_dir()


def test_full_scan_processes_unchanged_notes(tmp_path: Path, vault_root: Path) -> None:
    """A note the manifest already knows about (unchanged since baseline) is still
    reprocessed under --full, unlike a normal incremental run."""
    vault = VaultIO(vault_root)
    vault.write_note("00-Inbox/idea.md", Note(metadata={"title": "Untitled"}, content="teh idea"))

    baseline_run = _make_run(tmp_path, vault_root, "apply", {"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE, "title_and_tagging": "A Real Title"}, [1.0, 0.0, 0.0, 0.0])
    baseline_run.run()  # establishes the manifest baseline; nothing left "changed" afterward

    full_run = _make_run(tmp_path, vault_root, "apply", {"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE, "title_and_tagging": "A Real Title"}, [1.0, 0.0, 0.0, 0.0], full_scan=True)
    report = full_run.run()

    assert "1 notes scanned" in report


def test_incremental_scan_skips_unchanged_notes_for_comparison(tmp_path: Path, vault_root: Path) -> None:
    vault = VaultIO(vault_root)
    vault.write_note("00-Inbox/idea.md", Note(metadata={"title": "Untitled"}, content="teh idea"))

    baseline_run = _make_run(tmp_path, vault_root, "apply", {"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE, "title_and_tagging": "A Real Title"}, [1.0, 0.0, 0.0, 0.0])
    baseline_run.run()

    incremental_run = _make_run(tmp_path, vault_root, "apply", {"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE, "title_and_tagging": "A Real Title"}, [1.0, 0.0, 0.0, 0.0])
    report = incremental_run.run()

    assert "0 notes scanned" in report


def test_full_scan_still_excludes_daily_notes(tmp_path: Path, vault_root: Path) -> None:
    vault = VaultIO(vault_root)
    today = datetime.now().strftime("%m-%d-%Y")
    vault.write_note(f"01-Daily/{today}.md", Note(metadata={"title": "Untitled"}, content="# Idea\nRaw journal text."))

    run = _make_run(tmp_path, vault_root, "apply", {"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE, "title_and_tagging": "Should Not Be Used", "daily_digestion": "Idea"}, [1.0, 0.0, 0.0, 0.0], full_scan=True)
    run.run()

    daily = vault.read_note(f"01-Daily/{today}.md")
    assert daily.metadata.get("title") == "Untitled"


def test_recap_written_after_digesting_daily_note(tmp_path: Path, vault_root: Path) -> None:
    vault = VaultIO(vault_root)
    today = datetime.now().strftime("%m-%d-%Y")
    vault.write_note(f"01-Daily/{today}.md", Note(metadata={}, content="# New Idea\nSomething worth keeping."))

    run = _make_run(tmp_path, vault_root, "apply", {"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE, "daily_digestion": "New Idea Note", "daily_recap": "Yesterday you explored a new idea.\n\n1. What was the idea about?"}, [1.0, 0.0, 0.0, 0.0])
    run.run()

    assert vault.exists(f"_reports/Recap-{today}.md")
    recap = vault.read_raw(f"_reports/Recap-{today}.md")
    assert "Yesterday you explored a new idea" in recap


def test_no_recap_written_without_a_daily_note(tmp_path: Path, vault_root: Path) -> None:
    vault = VaultIO(vault_root)
    vault.write_note("00-Inbox/idea.md", Note(metadata={"title": "Untitled"}, content="teh idea"))
    today = datetime.now().strftime("%m-%d-%Y")

    run = _make_run(tmp_path, vault_root, "apply", {"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE, "title_and_tagging": "A Real Title"}, [1.0, 0.0, 0.0, 0.0])
    run.run()

    assert not vault.exists(f"_reports/Recap-{today}.md")


def test_fact_check_appends_callout_to_newly_digested_note(tmp_path: Path, vault_root: Path) -> None:
    vault = VaultIO(vault_root)
    today = datetime.now().strftime("%m-%d-%Y")
    vault.write_note(f"01-Daily/{today}.md", Note(metadata={}, content="# New Idea\nSomething worth keeping."))

    run = _make_run(
        tmp_path,
        vault_root,
        "apply",
        {
            "bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE,
            "daily_digestion": "New Idea Note",
            "fact_check": "---NOTE: New Idea Note---\nWorth double-checking one detail here; see a primary source.",
        },
        [1.0, 0.0, 0.0, 0.0],
    )
    report = run.run()

    new_note = vault.read_note("00-Inbox/New-Idea-Note.md")
    assert "[!ai-fact-check]" in new_note.content
    assert "Fact-check flagged New Idea Note" in report


def test_fact_check_skipped_when_disabled_in_config(tmp_path: Path, vault_root: Path) -> None:
    vault = VaultIO(vault_root)
    today = datetime.now().strftime("%m-%d-%Y")
    vault.write_note(f"01-Daily/{today}.md", Note(metadata={}, content="# New Idea\nSomething worth keeping."))

    config = replace(
        make_config(vault_root, mode="apply"),
        fact_check=FactCheckConfig(enabled=False),
    )
    git = GitSafety(vault_root, require_git=True)
    router = FakeRouter(
        {
            "bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE,
            "daily_digestion": "New Idea Note",
            "fact_check": "---NOTE: New Idea Note---\nShould never be requested.",
        }
    )
    manifest = ManifestManager(vault_path=str(vault_root), excluded_folders=["_staging", "_reports"], db_path=tmp_path / "manifest.sqlite")
    run = NightlyRun(
        config,
        dry_run=False,
        vault=vault,
        git=git,
        router=router,
        embedder=FakeEmbedder([1.0, 0.0, 0.0, 0.0]),
        vector_store=VectorStore(tmp_path / "vector_store", embedding_dim=4),
        manifest=manifest,
    )
    report = run.run()

    new_note = vault.read_note("00-Inbox/New-Idea-Note.md")
    assert "[!ai-fact-check]" not in new_note.content
    assert "Fact-check flagged" not in report


def test_fact_check_failure_is_flagged_not_fatal(tmp_path: Path, vault_root: Path) -> None:
    import httpx

    class RaisingRouter(FakeRouter):
        def generate(self, task_key: str, system: str, prompt: str, grounded: bool = False) -> LLMResponse:
            if task_key == "fact_check":
                raise httpx.HTTPError("rate limited")
            return super().generate(task_key, system, prompt, grounded=grounded)

    vault = VaultIO(vault_root)
    today = datetime.now().strftime("%m-%d-%Y")
    vault.write_note(f"01-Daily/{today}.md", Note(metadata={}, content="# New Idea\nSomething worth keeping."))

    config = make_config(vault_root, mode="apply")
    git = GitSafety(vault_root, require_git=True)
    router = RaisingRouter({"bulk_grammar_pass": CLEAR_GRAMMAR_RESPONSE, "daily_digestion": "New Idea Note"})
    manifest = ManifestManager(vault_path=str(vault_root), excluded_folders=["_staging", "_reports"], db_path=tmp_path / "manifest.sqlite")
    run = NightlyRun(
        config,
        dry_run=False,
        vault=vault,
        git=git,
        router=router,
        embedder=FakeEmbedder([1.0, 0.0, 0.0, 0.0]),
        vector_store=VectorStore(tmp_path / "vector_store", embedding_dim=4),
        manifest=manifest,
    )
    report = run.run()  # must not raise

    assert vault.exists("00-Inbox/New-Idea-Note.md")
