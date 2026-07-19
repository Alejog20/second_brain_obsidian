"""Report module: builds the markdown morning report (requirement 5)."""

import datetime
from typing import Any


class ReportGenerator:
    """Formats run stats, significant changes, new notes, and flags into the morning report."""

    def generate_report(
        self,
        stats: dict[str, Any],
        items: list[dict[str, Any]],
        new_notes: list[dict[str, Any]],
        flags: list[dict[str, Any]],
    ) -> str:
        """Build the full markdown report string."""
        date_str = datetime.datetime.now().strftime("%m-%d-%Y")
        lines = [f"# Review — {date_str}", "", self._summary_line(stats), ""]
        lines += self._significant_changes_section(items)
        lines += self._new_notes_section(new_notes)
        lines += self._minor_changes_section(stats, date_str)
        lines += self._flagged_section(flags)
        return "\n".join(lines)

    def _summary_line(self, stats: dict[str, Any]) -> str:
        """Format the one-line run summary."""
        return (
            f"Run: {stats.get('start_time', '?')}–{stats.get('end_time', '?')} · "
            f"{stats.get('scanned', 0)} notes scanned · "
            f"{stats.get('changed', 0)} changed · "
            f"{stats.get('new', 0)} new · "
            f"est. cost ${stats.get('cost', 0.0):.2f}"
        )

    def _significant_changes_section(self, items: list[dict[str, Any]]) -> list[str]:
        """Format renames/merges/taxonomy-moves/deletions above the materiality threshold."""
        lines = ["## Significant changes"]
        if not items:
            lines.append("(No major structural changes detected.)")
        else:
            for item in items:
                lines.append(f"### {item.get('reason', 'Update')}")
                lines.append(f"{item.get('detail', '')}\n")
        return lines

    def _new_notes_section(self, new_notes: list[dict[str, Any]]) -> list[str]:
        """Format notes created by today's daily-note digestion."""
        lines = ["## New notes from today's daily digestion"]
        if not new_notes:
            lines.append("No new notes generated today.")
        else:
            for note in new_notes:
                lines.append(f"- [[{note['title']}]] ← from [[{note['source']}]]")
        return lines

    def _minor_changes_section(self, stats: dict[str, Any], date_str: str) -> list[str]:
        """Format the minor-changes count and a link to the full diff log."""
        lines = ["## Minor changes"]
        minor_count = stats.get("minor_changes", 0)
        if minor_count > 0:
            lines.append(f"{minor_count} total — see full log.")
            lines.append(f"[View full diff log](_reports/{date_str}-full-diff.json)")
        else:
            lines.append("No minor changes recorded.")
        return lines

    def _flagged_section(self, flags: list[dict[str, Any]]) -> list[str]:
        """Format notes flagged for manual review, left untouched by the pipeline."""
        lines = ["## Flagged for your review"]
        if not flags:
            lines.append("All items successfully processed.")
        else:
            for flag in flags:
                lines.append(f"- [[{flag['title']}]] — {flag.get('reason', 'needs review')}")
        return lines


report_generator = ReportGenerator()


def generate_morning_report(
    stats: dict[str, Any],
    items: list[dict[str, Any]],
    new_notes: list[dict[str, Any]],
    flags: list[dict[str, Any]],
) -> str:
    """Module-level convenience wrapper around the shared ReportGenerator instance."""
    return report_generator.generate_report(stats, items, new_notes, flags)
