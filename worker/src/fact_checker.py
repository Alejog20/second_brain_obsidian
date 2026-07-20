"""Fact-checker module: a daily, web-search-grounded fact-check + simple-explanation pass.

Scoped to only the notes digested during the current run (see nightly_run.py's
_digest_today) - never the whole vault - so this is small and cheap enough to run
automatically every night. Every note in scope goes into a single grounded call, not
one call per note, to keep this well under Gemini's free-tier rate limit (see
llm_router.py's GeminiProvider.generate docstring for what "grounded" adds to the
request).
"""

from dataclasses import dataclass
from typing import Any

from .llm_router import Router
from .markers import format_callout
from .vault_io import Note, VaultIO

FACT_CHECK_SYSTEM_PROMPT = (
    "You fact-check notes someone just wrote, using web search to verify factual claims. For "
    "each note below, respond with one block in exactly this format:\n"
    "---NOTE: <title, copied exactly as given>---\n"
    "<your findings>\n"
    "In the findings: call out any claim that's wrong, outdated, or unverifiable, citing what "
    "search found and where; if a concept is stated without a simple explanation or concrete "
    "example, add one brief example to make it stick. Keep each note's findings to at most a "
    "few sentences - this is a quick daily check, not a rewrite, and token budget is limited. "
    "If a note has nothing worth flagging or adding, its block's body must be exactly: "
    "nothing to add. Only state something as fact if your search actually supports it."
)

_NOTHING_TO_ADD = "nothing to add"


@dataclass(frozen=True)
class FactCheckEntry:
    """One digested note's title, content, and vault path - the path is where a finding gets written."""

    title: str
    content: str
    rel_path: str


class FactChecker:
    """Runs one batched, grounded fact-check pass over a run's newly digested notes."""

    def __init__(self, router: Router, vault: VaultIO, dry_run: bool) -> None:
        self._router = router
        self._vault = vault
        self._dry_run = dry_run

    def run(self, entries: list[FactCheckEntry], date_str: str) -> list[dict[str, Any]]:
        """Fact-check every entry in one grounded call; returns report items for flagged notes."""
        if not entries:
            return []

        material = "\n\n".join(f"## {entry.title}\n{entry.content}" for entry in entries)
        response = self._router.generate(
            "fact_check", system=FACT_CHECK_SYSTEM_PROMPT, prompt=material, grounded=True
        )
        findings = self._parse_response(response.text)

        report_items: list[dict[str, Any]] = []
        for entry in entries:
            finding = findings.get(entry.title)
            if not finding:
                continue
            self._append_finding(entry.rel_path, finding, date_str)
            report_items.append({"reason": f"Fact-check flagged {entry.title}", "detail": finding})
        return report_items

    def _append_finding(self, rel_path: str, finding: str, date_str: str) -> None:
        """Append a fact-check callout to the end of the note it concerns."""
        note = self._vault.read_note(rel_path)
        callout = format_callout("ai-fact-check", date_str, finding)
        updated_content = f"{note.content.rstrip()}\n\n{callout}\n"
        self._vault.write_or_stage(rel_path, Note(metadata=note.metadata, content=updated_content), dry_run=self._dry_run)

    @staticmethod
    def _parse_response(raw: str) -> dict[str, str]:
        """Split the batched response back into per-note findings, keyed by title.

        Notes the model reported as "nothing to add" are dropped entirely, as is any block
        the model didn't title-match to something we recognize as a real delimiter.
        """
        results: dict[str, str] = {}
        for block in raw.split("---NOTE:")[1:]:
            title_line, _, body = block.partition("---")
            title = title_line.strip()
            finding = body.strip()
            if title and finding and finding.lower() != _NOTHING_TO_ADD:
                results[title] = finding
        return results
