"""Tests for the morning report generator."""

from src.report import generate_morning_report


def test_report_includes_summary_and_sections() -> None:
    stats = {
        "start_time": "02:00",
        "end_time": "02:14",
        "scanned": 10,
        "changed": 2,
        "new": 1,
        "cost": 0.11,
        "minor_changes": 3,
    }
    items = [{"reason": "Merged X into Y", "detail": "Some detail."}]
    new_notes = [{"title": "Qwen3 Local Model Notes", "source": "07-15-2026"}]
    flags = [{"title": "Cloud Cost Optimization Draft", "reason": "clarity score low"}]

    report = generate_morning_report(stats, items, new_notes, flags)

    assert "10 notes scanned" in report
    assert "### Merged X into Y" in report
    assert "- [[Qwen3 Local Model Notes]] ← from [[07-15-2026]]" in report
    assert "- [[Cloud Cost Optimization Draft]] — clarity score low" in report


def test_report_handles_empty_sections_gracefully() -> None:
    stats = {"scanned": 0, "changed": 0, "new": 0}
    report = generate_morning_report(stats, [], [], [])

    assert "(No major structural changes detected.)" in report
    assert "No new notes generated today." in report
    assert "No minor changes recorded." in report
    assert "All items successfully processed." in report
