"""Manifest module: SHA-256 hash-based change detection for Markdown vault files."""

import hashlib
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "config.yaml"))
DEFAULT_DB_PATH = Path(os.environ.get("DATA_PATH", "data")) / "manifest.sqlite"


@dataclass(frozen=True)
class VaultDelta:
    """Represents added/modified/deleted vault-relative paths between two manifest states."""

    added: list[Path] = field(default_factory=list)
    modified: list[Path] = field(default_factory=list)
    deleted: list[Path] = field(default_factory=list)


class ManifestManager:
    """Scans Markdown files in the vault and diffs their hashes against SQLite-manifest state."""

    def __init__(
        self,
        vault_path: Optional[str] = None,
        excluded_folders: Optional[list[str]] = None,
        db_path: Optional[Path] = None,
        config_path: Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        config = self._load_config(config_path)
        vault_cfg = config.get("vault", {})

        resolved_vault = vault_path or vault_cfg.get("path")
        if not resolved_vault:
            raise ValueError("vault path must be provided or set via config.yaml: vault.path")
        self._vault_root = Path(resolved_vault).resolve()
        if not self._vault_root.is_dir():
            raise FileNotFoundError(f"vault root does not exist or is not a directory: {self._vault_root}")

        self._excluded_folders = set(excluded_folders if excluded_folders is not None else vault_cfg.get("excluded_folders", []))
        self._db_path = db_path or DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @staticmethod
    def _load_config(config_path: Path) -> dict:
        """Load config.yaml, returning an empty dict if the file doesn't exist."""
        if not config_path.exists():
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _get_hash(file: Path) -> str:
        """Compute the SHA-256 hash of a file's contents."""
        sha256 = hashlib.sha256()
        with open(file, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _init_db(self) -> None:
        """Create the manifest table if it doesn't already exist."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS manifest (
                    path TEXT PRIMARY KEY,
                    hash TEXT NOT NULL,
                    mtime REAL NOT NULL
                )
                """
            )
            conn.commit()

    def _is_excluded(self, rel_path: str) -> bool:
        """Check whether a vault-relative path falls under an excluded top-level folder."""
        return bool(self._excluded_folders.intersection(Path(rel_path).parts))

    def _scan_vault(self) -> dict[str, dict]:
        """Walk the vault and return a map of relative path -> hash/mtime/full path."""
        current_files: dict[str, dict] = {}
        for root, _dirs, files in os.walk(self._vault_root):
            for name in files:
                if not name.endswith(".md"):
                    continue
                full_path = Path(root) / name
                rel_path = full_path.relative_to(self._vault_root).as_posix()
                if self._is_excluded(rel_path):
                    continue
                try:
                    current_files[rel_path] = {
                        "path": full_path,
                        "hash": self._get_hash(full_path),
                        "mtime": full_path.stat().st_mtime,
                    }
                except FileNotFoundError:
                    # Editors (Obsidian included) save via temp-file-then-rename; a file
                    # listed by os.walk can vanish before it's hashed. Skip it this run.
                    continue
        return current_files

    def get_delta(self) -> VaultDelta:
        """Scan the vault and return the delta against the last committed manifest state."""
        current_files = self._scan_vault()

        with sqlite3.connect(str(self._db_path)) as conn:
            db_records = {
                row[0]: {"hash": row[1], "mtime": row[2]}
                for row in conn.execute("SELECT path, hash, mtime FROM manifest")
            }

        added: list[Path] = []
        modified: list[Path] = []
        deleted: list[Path] = []

        for rel_path, info in current_files.items():
            record = db_records.get(rel_path)
            if record is None:
                added.append(Path(rel_path))
            elif info["hash"] != record["hash"]:
                modified.append(Path(rel_path))

        for rel_path in db_records:
            if rel_path not in current_files:
                deleted.append(Path(rel_path))

        return VaultDelta(added=added, modified=modified, deleted=deleted)

    def commit(self) -> None:
        """Persist a fresh scan as the new baseline.

        Always rescans rather than reusing get_delta()'s snapshot: callers (e.g. nightly_run)
        typically mutate the vault - grammar fixes, new notes, a regenerated AGENTS.md - in the
        window between get_delta() and commit(). A cached pre-mutation snapshot would persist
        stale hashes for files the pipeline itself just wrote, making them look "changed" again
        on the very next run.
        """
        current_files = self._scan_vault()
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("DELETE FROM manifest")
            conn.executemany(
                "INSERT INTO manifest (path, hash, mtime) VALUES (?, ?, ?)",
                [(rel, info["hash"], info["mtime"]) for rel, info in current_files.items()],
            )
            conn.commit()
