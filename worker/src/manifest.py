# -*- coding: utf-8 -*-
"""Manifest module for SHA-256 hash-based change detection of Markdown vault files."""

import hashlib, sqlite3, os, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

@dataclass(frozen=True)
class VaultDelta:
    """Represent changes between two manifest states (vault-relative paths only)."""
    added: list[Path] = field(default_factory=list, metadata={"type": "relative_paths"})
    modified: list[Path] = field(default_factory=list, metadata={"type": "relative_paths"})
    deleted: list[Path] = field(default_factory=list, metadata={"type": "relative_paths"})

class ManifestManager:
    """Handles scanning Markdown files and computing diffs against SQLite-manifest state."""

    DEFAULT_DB_PATH = Path("data/manifest.sqlite")
    
    def __init__(self, vault_path: Optional[str] = None):
        # If vault_path is provided, it's the absolute path to the folder containing 'Second Brain' or the root of the vault as defined in config.
        # According to architecture, we usually look for a subfolder or use the direct mount.
        self._vault_root = (Path(vault_path) / "Second Brain").resolve() if vault_path else None
        if not self._vault_root and not os.path.isdir(Path(".").resolve()):
            # Fallback to current directory if no path is provided, though config should provide it.
            pass

    @staticmethod
    def _get_hash(file: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        try:
            with open(file, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except FileNotFoundError:
            return ""

    def _init_db(self):
        """Initialize the SQLite database for tracking file hashes."""
        os.makedirs(self.DEFAULT_DB_PATH.parent, exist_ok=True)
        conn = sqlite3.connect(str(self.DEFAULT_DB_path)) # Wait, using local logic or config? 
                                                                # For now, use the class attribute but handle pathing safely.
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS manifest (
                path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                mtime REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def get_delta(self) -> VaultDelta:
        """
        Scans the vault and returns a VaultDelta comparing current state to the database.
        """
        current_files = {}
        # Scan all .md files in the vault (excluding specific directories if needed)
        # In practice, this would use a glob or similar.
        if not self._vault_root:
            return VaultDelta()

        # We'll assume we are scanning the vault rooted at self._vault_root
        for root, dirs, files in os.walk(self._vault_root):
            for file in files:
                if file.endswith(".md"):
                    full_path = Path(root) / file
                    rel_path = full_path.relative_to(self._vault_root).as_posix()
                    # Check if it's a "real" note (not in excluded folders)
                    if any(x in rel_path for x in ["_staging", "_reports"]):
                        continue
                    
                    current_files[rel_path] = {
                        "path": full_path,
                        "hash": self._get_hash(full_path),
                        "mtime": os.path.getmtime(full_path)
                    }

        conn = sqlite3.connect(str(self.DEFAULT_DB_PATH))
        cursor = conn.cursor()
        
        db_records = {}
        cursor.execute("SELECT path, hash, mtime FROM manifest")
        for row in cursor.fetchall():
            db_records[row[0]] = {"hash": row[1], "mtime": row[2]}

        added = []
        modified = []
        deleted = []

        # Identify added and modified
        for rel_path, info in current_files.items():
            if rel_path not in db_records:
                added.append(Path(rel_path))
            else:
                db_record = db_records[rel_path]
                if info["hash"] != db_record["hash"]:
                    modified.append(Path(rel_path))

        # Identify deleted
        for rel_path in db_records:
            if rel_path not in current_files:
                deleted.append(Path(rel_path))

        conn.close()
        return VaultDelta(added=added, modified=modified, deleted=deleted)
</write_to_file>
<task_progress>
- [x] Explore project structure and documentation
- [x] Read core documentation (Architecture, AGENTS.md)
- [x] Understand architecture
- [x] Identify immediate technical setup items/missing configs
- [ ] Implement Step 1: manifest.py (Diff Scanner)
- [ ] Implement Step 2: llm_router.py & config.yaml (Model Routing)
- [ ] Implement Step 3: reorganizer.py (Dry-run edits)
</task_progress>
</write_to_file>