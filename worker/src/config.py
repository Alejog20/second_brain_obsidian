"""Config module: loads config.yaml once into typed, immutable config objects."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "config.yaml"))


@dataclass(frozen=True)
class VaultConfig:
    """Vault location, daily-note conventions, and folders excluded from processing."""

    path: str
    daily_notes_folder: str = "01-Daily"
    daily_note_date_format: str = "MM-DD-YYYY"
    excluded_folders: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SafetyConfig:
    """Dry-run/apply mode and the git and materiality guardrails around it."""

    mode: str = "dry_run"
    require_git: bool = True
    materiality_threshold: str = "structural"

    @property
    def is_apply(self) -> bool:
        """True once the pipeline is trusted to write directly to the vault."""
        return self.mode == "apply"


@dataclass(frozen=True)
class EmbeddingsConfig:
    """Embedding provider/model and where the vector store lives on disk."""

    provider: str = "ollama"
    model: str = "nomic-embed-text"
    store: str = "lancedb"
    store_path: str = "/data/vector_store"


@dataclass(frozen=True)
class ModelTaskConfig:
    """The provider/model routed to a single pipeline task."""

    provider: str
    model: str


@dataclass(frozen=True)
class CostTrackingConfig:
    """Whether to accumulate and report estimated spend."""

    enabled: bool = True
    currency: str = "USD"


@dataclass(frozen=True)
class ReportConfig:
    """Output paths for the nightly morning report and its full diff log."""

    path: str = "_reports/Review-{date}.md"
    full_diff_log_path: str = "_reports/{date}-full-diff.json"
    include_cost_summary: bool = True


@dataclass(frozen=True)
class Config:
    """The fully parsed, typed contents of config.yaml for a single pipeline run."""

    vault: VaultConfig
    safety: SafetyConfig
    embeddings: EmbeddingsConfig
    models: dict[str, ModelTaskConfig]
    cost_tracking: CostTrackingConfig
    report: ReportConfig

    def model_for(self, task_key: str) -> ModelTaskConfig:
        """Look up the provider/model configured for a pipeline task."""
        try:
            return self.models[task_key]
        except KeyError:
            raise ValueError(f"no model configured for task '{task_key}'") from None


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load and validate config.yaml into a typed, immutable Config. Call once per process."""
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    vault_raw = raw.get("vault", {})
    try:
        vault = VaultConfig(
            path=vault_raw["path"],
            daily_notes_folder=vault_raw.get("daily_notes_folder", "01-Daily"),
            daily_note_date_format=vault_raw.get("daily_note_date_format", "MM-DD-YYYY"),
            excluded_folders=tuple(vault_raw.get("excluded_folders", [])),
        )
    except KeyError:
        raise ValueError(f"{path} is missing required field: vault.path") from None

    safety_raw = raw.get("safety", {})
    safety = SafetyConfig(
        mode=safety_raw.get("mode", "dry_run"),
        require_git=safety_raw.get("require_git", True),
        materiality_threshold=safety_raw.get("materiality_threshold", "structural"),
    )

    embeddings_raw = raw.get("embeddings", {})
    embeddings = EmbeddingsConfig(
        provider=embeddings_raw.get("provider", "ollama"),
        model=embeddings_raw.get("model", "nomic-embed-text"),
        store=embeddings_raw.get("store", "lancedb"),
        store_path=embeddings_raw.get("store_path", "/data/vector_store"),
    )

    models_raw = raw.get("models", {})
    try:
        models = {
            task_key: ModelTaskConfig(provider=cfg["provider"], model=cfg["model"])
            for task_key, cfg in models_raw.items()
        }
    except KeyError:
        raise ValueError(f"{path}: every entry under 'models' needs a provider and a model") from None

    cost_tracking_raw = raw.get("cost_tracking", {})
    cost_tracking = CostTrackingConfig(
        enabled=cost_tracking_raw.get("enabled", True),
        currency=cost_tracking_raw.get("currency", "USD"),
    )

    report_raw = raw.get("report", {})
    report = ReportConfig(
        path=report_raw.get("path", "_reports/Review-{date}.md"),
        full_diff_log_path=report_raw.get("full_diff_log_path", "_reports/{date}-full-diff.json"),
        include_cost_summary=report_raw.get("include_cost_summary", True),
    )

    return Config(
        vault=vault,
        safety=safety,
        embeddings=embeddings,
        models=models,
        cost_tracking=cost_tracking,
        report=report,
    )
