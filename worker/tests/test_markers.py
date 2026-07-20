"""Tests for the shared Obsidian-callout formatter used to mark AI-added content."""

from src.markers import format_callout


def test_format_callout_wraps_single_line_body() -> None:
    result = format_callout("ai-fact-check", "07-20-2026", "Everything checks out.")

    assert result == "> [!ai-fact-check] Second Brain — 07-20-2026\n> Everything checks out."


def test_format_callout_prefixes_every_line_including_blank_ones() -> None:
    result = format_callout("ai-fact-check", "07-20-2026", "Line one.\n\nLine two.")

    lines = result.splitlines()
    assert lines[0] == "> [!ai-fact-check] Second Brain — 07-20-2026"
    assert lines[1] == "> Line one."
    assert lines[2] == ">"
    assert lines[3] == "> Line two."


def test_format_callout_strips_surrounding_whitespace_from_body() -> None:
    result = format_callout("ai-fact-check", "07-20-2026", "\n  padded body  \n")

    assert result == "> [!ai-fact-check] Second Brain — 07-20-2026\n> padded body"
