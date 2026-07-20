"""Nightly run job: orchestrates manifest diff -> reorganize -> digest -> AGENTS.md -> report."""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

from ..agents_md_builder import AgentsBuilder
from ..config import Config, load_config
from ..digestor import Digestor, DigestedChunk
from ..fact_checker import FactChecker, FactCheckEntry
from ..git_safety import GitSafety
from ..llm_router import LLMRouter, Router
from ..manifest import ManifestManager
from ..recap import RecapEntry, RecapGenerator
from ..reorganizer import Reorganizer
from ..report import generate_morning_report
from ..vault_io import Note, PathBlockedByFileError, VaultIO
from ..vector_store import Embedder, GeminiEmbeddingClient, OllamaEmbeddingClient, VectorStore
from ..worktree import NightlyWorktree

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path(os.environ.get("DATA_PATH", "data"))

_DATE_FORMAT_PATTERNS = {
    "MM-DD-YYYY": "%m-%d-%Y",
    "YYYY-MM-DD": "%Y-%m-%d",
    "DD-MM-YYYY": "%d-%m-%Y",
}

_DATE_FORMAT_REGEXES = {
    "MM-DD-YYYY": re.compile(r"^\d{2}-\d{2}-\d{4}$"),
    "YYYY-MM-DD": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    "DD-MM-YYYY": re.compile(r"^\d{2}-\d{2}-\d{4}$"),
}


def strftime_pattern(date_format: str) -> str:
    """Translate config.yaml's human date-format string into a strftime pattern."""
    try:
        return _DATE_FORMAT_PATTERNS[date_format]
    except KeyError:
        raise ValueError(f"unsupported daily_note_date_format: {date_format}") from None


def _join_folder(folder: str, filename: str) -> str:
    """Join an optional folder prefix with a filename; folder="" means the vault root."""
    return f"{folder}/{filename}" if folder else filename


def _build_default_embedder(config: Config) -> Embedder:
    """Construct the embedder config.yaml's embeddings.provider actually asks for."""
    if config.embeddings.provider == "gemini":
        return GeminiEmbeddingClient(model=config.embeddings.model)
    if config.embeddings.provider == "ollama":
        return OllamaEmbeddingClient(model=config.embeddings.model)
    raise ValueError(f"unsupported embeddings provider: {config.embeddings.provider}")


def resolve_dry_run(config: Config) -> bool:
    """Run live only if both config.yaml's safety.mode and the DRY_RUN env var agree to apply."""
    env_allows_apply = os.environ.get("DRY_RUN", "true").strip().lower() == "false"
    return not (config.safety.is_apply and env_allows_apply)


class NightlyRun:
    """Runs the full nightly pipeline once, end to end, and returns the morning report markdown."""

    def __init__(
        self,
        config: Config,
        dry_run: bool,
        vault_root: Optional[Path] = None,
        vault: Optional[VaultIO] = None,
        git: Optional[GitSafety] = None,
        router: Optional[Router] = None,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
        manifest: Optional[ManifestManager] = None,
        full_scan: bool = False,
    ) -> None:
        self._config = config
        self._dry_run = dry_run
        self._full_scan = full_scan
        self._vault_root = vault_root or Path(config.vault.path)
        self._vault = vault or VaultIO(self._vault_root)
        self._git = git or GitSafety(self._vault_root, require_git=config.safety.require_git)
        self._router = router or LLMRouter(config)
        self._embedder = embedder or _build_default_embedder(config)
        self._vector_store = vector_store or VectorStore(DEFAULT_DATA_PATH / "vector_store")
        self._manifest = manifest or ManifestManager(
            vault_path=str(self._vault_root), excluded_folders=list(config.vault.excluded_folders)
        )
        self._reorganizer = Reorganizer(self._router, self._embedder, self._vector_store)
        self._digestor = Digestor(
            self._router,
            self._embedder,
            self._vector_store,
            self._vault,
            dry_run=dry_run,
            default_folder=config.vault.default_new_note_folder,
        )
        self._agents_builder = AgentsBuilder(self._vault, excluded_folders=frozenset(config.vault.excluded_folders))
        self._recap_generator = RecapGenerator(self._router)
        self._fact_checker = FactChecker(self._router, self._vault, dry_run=dry_run)
        self._materiality_any = config.safety.materiality_threshold == "any"
        self._taxonomy_changed = False

    def run(self) -> str:
        """Execute the full nightly pipeline and return the generated morning report markdown."""
        start = datetime.now()
        significant_items: list[dict[str, Any]] = []
        new_notes: list[dict[str, Any]] = []
        flags: list[dict[str, Any]] = []
        minor_changes = 0
        scanned_count = 0

        logger.info("Starting nightly run against %s (%s)", self._vault_root, "dry-run" if self._dry_run else "apply")

        run_label = start.strftime("nightly run %Y-%m-%d %H:%M")
        with self._git.bracket(run_label):
            note_paths = self._scan_note_paths()
            scanned_count = len(note_paths)
            if self._full_scan:
                logger.info("Full scan: %d note(s) in the vault", scanned_count)
            else:
                logger.info("%d note(s) changed since the last run", scanned_count)

            for rel_path in note_paths:
                logger.debug("Reorganizing %s", rel_path)
                significant_item, flag, is_minor = self._process_note(str(rel_path))
                if significant_item:
                    logger.info("%s", significant_item["reason"])
                    significant_items.append(significant_item)
                if flag:
                    logger.warning("Flagged %s: %s", flag["title"], flag["reason"])
                    flags.append(flag)
                if is_minor:
                    minor_changes += 1

            logger.info("Digesting today's daily note, if present")
            digested_new, digested_merges, chunks = self._digest_today()
            new_notes.extend(digested_new)
            significant_items.extend(digested_merges)
            if digested_new:
                logger.info("Created %d new note(s) from today's daily note", len(digested_new))
            if digested_merges:
                logger.info("Merged %d chunk(s) into existing notes", len(digested_merges))

            date_str = start.strftime(strftime_pattern(self._config.vault.daily_note_date_format))
            recap_entries = [RecapEntry(title=c.title, content=c.content) for c in chunks]
            self._maybe_write_recap(recap_entries, date_str)

            if self._config.fact_check.enabled:
                fact_check_entries = [
                    FactCheckEntry(title=c.title, content=c.content, rel_path=c.rel_path) for c in chunks
                ]
                significant_items.extend(self._run_fact_check(fact_check_entries, date_str))

            self._maybe_rebuild_agents_md()
            self._manifest.commit()

            # Report generation and write happen inside the bracket so the post-run commit
            # captures the report itself - otherwise a `git revert` of a bad night would
            # leave a stale, uncommitted report file behind describing a run that got undone.
            end = datetime.now()
            stats = {
                "start_time": start.strftime("%H:%M"),
                "end_time": end.strftime("%H:%M"),
                "scanned": scanned_count,
                "changed": len(significant_items) + minor_changes,
                "new": len(new_notes),
                "cost": round(self._router_cost(), 2),
                "minor_changes": minor_changes,
            }
            report = generate_morning_report(stats, significant_items, new_notes, flags)
            self._write_report(report, end)

        logger.info("Nightly run complete: %d changed, %d new, %d flagged", stats["changed"], stats["new"], len(flags))
        return report

    def _is_daily_note(self, rel_path: Path) -> bool:
        """Identify a daily note by location + filename pattern.

        daily_notes_folder may be "" (vault root), in which case daily notes sit directly
        alongside topic folders and can only be told apart by filename - so this checks both
        the location and that the filename actually matches the configured date format,
        rather than assuming every root-level file is a daily note.
        """
        daily_folder = self._config.vault.daily_notes_folder
        parts = rel_path.parts
        if daily_folder:
            if len(parts) < 2 or parts[0] != daily_folder:
                return False
        elif len(parts) != 1:
            return False

        date_regex = _DATE_FORMAT_REGEXES.get(self._config.vault.daily_note_date_format)
        return bool(date_regex and date_regex.match(rel_path.stem))

    def _scan_note_paths(self) -> list[Path]:
        """List the notes to process this run: every note in a full scan, else just what changed.

        Either way the daily-notes-folder/AGENTS.md exclusion is the same - a full scan isn't an
        excuse to reorganize daily notes, it's just "check everything else, not only what's new."
        """
        if self._full_scan:
            candidates = (Path(rel_path) for rel_path, _ in self._vault.iter_notes(frozenset(self._config.vault.excluded_folders)))
        else:
            delta = self._manifest.get_delta()
            candidates = iter(delta.added + delta.modified)
        return [p for p in candidates if not self._is_daily_note(p) and str(p) != "AGENTS.md"]

    def _router_cost(self) -> float:
        """Read cumulative cost from the router if it tracks one; fakes injected in tests may not."""
        return getattr(self._router, "total_cost_usd", 0.0)

    def _process_note(self, rel_path: str) -> tuple[Optional[dict], Optional[dict], bool]:
        """Reorganize one non-daily note and apply the result; returns (significant_item, flag, is_minor)."""
        if not self._vault.exists(rel_path):
            return None, None, False

        note = self._vault.read_note(rel_path)
        try:
            result = self._reorganizer.reorganize(rel_path, note)
        except httpx.HTTPError as exc:
            return None, {"title": rel_path, "reason": f"reorganize failed: {exc}"}, False

        updated_note = Note(metadata={**note.metadata, "title": result.title}, content=result.content)

        try:
            if result.suggested_folder:
                new_rel_path = f"{result.suggested_folder}/{Path(rel_path).name}"
                self._vault.move_note(rel_path, new_rel_path, updated_note, dry_run=self._dry_run)
                self._taxonomy_changed = True
                item = {
                    "reason": f"Moved {Path(rel_path).name} to {result.suggested_folder}",
                    "detail": f"High embedding similarity to existing notes already in {result.suggested_folder}.",
                }
                return item, None, False

            if result.flags:
                self._vault.write_or_stage(rel_path, updated_note, dry_run=self._dry_run)
                flag = {"title": result.title, "reason": "; ".join(result.flags)}
                return None, flag, False

            changed = updated_note.content != note.content or result.title != note.metadata.get("title")
            if changed:
                self._vault.write_or_stage(rel_path, updated_note, dry_run=self._dry_run)
                if self._materiality_any:
                    item = {"reason": f"Grammar/title tidy-up: {rel_path}", "detail": f"Title: {result.title}"}
                    return item, None, False
                return None, None, True

            return None, None, False
        except PathBlockedByFileError as exc:
            return None, {"title": rel_path, "reason": f"could not apply changes: {exc}"}, False

    def _digest_today(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[DigestedChunk]]:
        """Digest today's daily note if it exists; returns (new_note_items, merge_items, chunks).

        chunks is the digestor's raw output - the single source of truth run() derives both
        the recap and the fact-check scope from, so "what counts as today's new material"
        can't drift between the two.
        """
        date_str = datetime.now().strftime(strftime_pattern(self._config.vault.daily_note_date_format))
        daily_rel_path = _join_folder(self._config.vault.daily_notes_folder, f"{date_str}.md")
        if not self._vault.exists(daily_rel_path):
            return [], [], []

        try:
            chunks = self._digestor.digest(daily_rel_path, date_str)
        except httpx.HTTPError as exc:
            return [], [{"reason": "Daily digestion failed", "detail": str(exc)}], []

        new_notes = [{"title": c.title, "source": date_str} for c in chunks if not c.merged_into_existing]
        merge_items = [
            {
                "reason": f"Merged today's note into {c.title}",
                "detail": f"New content appended with a link back to [[{date_str}]].",
            }
            for c in chunks
            if c.merged_into_existing
        ]
        return new_notes, merge_items, chunks

    def _maybe_rebuild_agents_md(self) -> None:
        """Rebuild AGENTS.md only when the taxonomy changed this run, or it doesn't exist yet."""
        agents_md_exists = self._vault.exists("AGENTS.md")
        if not (self._taxonomy_changed or not agents_md_exists):
            return
        logger.info("Rebuilding AGENTS.md (%s)", "taxonomy changed" if self._taxonomy_changed else "first run")
        existing = self._vault.read_raw("AGENTS.md") if agents_md_exists else None
        content = self._agents_builder.build(
            vault_name=self._vault_root.name,
            generated_date=datetime.now().strftime("%Y-%m-%d"),
            existing_agents_md=existing,
        )
        self._vault.write_raw("AGENTS.md", content)

    def _write_report(self, report: str, end: datetime) -> None:
        """Write the morning report to _reports/, always live regardless of dry-run mode."""
        date_str = end.strftime(strftime_pattern(self._config.vault.daily_note_date_format))
        report_rel_path = self._config.report.path.format(date=date_str)
        self._vault.write_raw(report_rel_path, report)

    def _maybe_write_recap(self, entries: list[RecapEntry], date_str: str) -> None:
        """Write a reinforcement recap of today's digested notes, if there were any."""
        try:
            recap = self._recap_generator.build(entries, date_str)
        except httpx.HTTPError as exc:
            logger.warning("Recap generation failed: %s", exc)
            return
        if recap is None:
            return
        logger.info("Writing morning recap")
        recap_rel_path = self._config.report.recap_path.format(date=date_str)
        self._vault.write_raw(recap_rel_path, recap)

    def _run_fact_check(self, entries: list[FactCheckEntry], date_str: str) -> list[dict[str, Any]]:
        """Run the grounded fact-check pass over today's digested notes, if there were any.

        Failure here (rate limit, network) degrades the same way recap generation does -
        logged and skipped, never crashing the run - since this is a nice-to-have addition
        on top of notes the digestor already successfully wrote.
        """
        try:
            items = self._fact_checker.run(entries, date_str)
        except httpx.HTTPError as exc:
            logger.warning("Fact-check pass failed: %s", exc)
            return []
        if items:
            logger.info("Fact-check flagged %d note(s)", len(items))
        return items


def main() -> None:
    """Entry point: python -m src.jobs.nightly_run."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    config = load_config()
    dry_run = resolve_dry_run(config)
    vault_root = Path(config.vault.path)

    worktree = None
    if config.git_review.enabled:
        worktree = NightlyWorktree(
            vault_root=vault_root,
            worktree_path=Path(config.git_review.worktree_path),
            branch=config.git_review.branch,
            base_branch=config.git_review.base_branch,
            remote=config.git_review.remote,
        )
        logger.info("Syncing review worktree onto %s (from %s/%s)", config.git_review.branch, config.git_review.remote, config.git_review.base_branch)
        vault_root = worktree.sync()
        dry_run = False  # the review branch itself is the staging area, not local _staging/

    NightlyRun(config, dry_run=dry_run, vault_root=vault_root).run()

    if worktree is not None:
        logger.info("Pushing review branch %s", config.git_review.branch)
        worktree.push()


if __name__ == "__main__":
    main()
